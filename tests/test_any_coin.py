"""Tests for arbitrary Binance USDT-M perpetual coin support (no network; mocked HTTP)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import requests
from streamlit.testing.v1 import AppTest

from riskguard.analytics.margin import lookup_margin_tier_with_source
from riskguard.analytics.portfolio import analyze_portfolio
from riskguard.analytics.positions import analyze_position, analyze_positions
from riskguard.analytics.sizing import pre_trade_check
from riskguard.data.cache import TTLCache
from riskguard.data.indicators import compute_daily_atr
from riskguard.data.providers import (
    DEFAULT_DEMO_SYMBOLS,
    MarketDataProvider,
    match_binance_usdtm_symbol,
    validate_symbol_format,
)
from riskguard.models import AccountState, PositionInput, RuleConfig

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DIR = ROOT / "data" / "sample"


class _FakeResponse:
    def __init__(self, status_code: int, payload: Any = None, text: str = "") -> None:
        self.status_code = status_code
        self._payload = payload
        self.headers: dict[str, str] = {}
        self.text = text or ""

    def json(self) -> Any:
        if self._payload is None:
            raise ValueError("no JSON")
        return self._payload


class _FakeSession:
    def __init__(self, handler) -> None:
        self._handler = handler
        self.calls: list[tuple[str, str, dict[str, Any] | None, Any]] = []

    def request(self, method: str, url: str, params=None, json=None, timeout=None):
        self.calls.append((method, url, params, json))
        return self._handler(method, url, params=params, json_body=json)


def _make_binance_klines(
    prices: list[float],
    start_dt: datetime = datetime(2025, 10, 1, 0, 0, tzinfo=timezone.utc),
    spread: float = 0.01,
) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for i, close_px in enumerate(prices):
        open_ms = int((start_dt + timedelta(hours=i)).timestamp() * 1000)
        close_ms = open_ms + 3_599_999
        open_px = prices[i - 1] if i > 0 else close_px
        high_px = max(open_px, close_px) * (1.0 + spread)
        low_px = min(open_px, close_px) * (1.0 - spread)
        rows.append(
            [
                open_ms,
                f"{open_px:.8f}",
                f"{high_px:.8f}",
                f"{low_px:.8f}",
                f"{close_px:.8f}",
                "1000.0",
                close_ms,
                f"{close_px * 1000.0:.4f}",
                100,
                "500.0",
                f"{close_px * 500.0:.4f}",
                "0",
            ]
        )
    return rows


def _synthetic_price_paths(
    hours: int = 960,
    btc_start: float = 100_000.0,
    link_start: float = 15.0,
    pepe_start: float = 0.012,
    link_beta: float = 1.6,
    pepe_beta: float = 2.2,
) -> dict[str, list[float]]:
    rng = np.random.default_rng(1234)
    btc_r = rng.normal(0.0, 0.004, hours - 1)
    paths: dict[str, list[float]] = {"BTCUSDT": [btc_start], "LINKUSDT": [link_start], "1000PEPEUSDT": [pepe_start]}
    for r in btc_r:
        paths["BTCUSDT"].append(paths["BTCUSDT"][-1] * (1.0 + float(r)))
        paths["LINKUSDT"].append(paths["LINKUSDT"][-1] * (1.0 + link_beta * float(r)))
        paths["1000PEPEUSDT"].append(paths["1000PEPEUSDT"][-1] * (1.0 + pepe_beta * float(r)))
    for sym in ("ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "BNBUSDT"):
        p = [100.0]
        for r in btc_r:
            p.append(p[-1] * (1.0 + 1.1 * float(r)))
        paths[sym] = p
    return paths


def _exchange_info_payload(extra_symbols: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    symbols = [
        {"symbol": s, "quoteAsset": "USDT", "contractType": "PERPETUAL", "status": "TRADING"}
        for s in [*DEFAULT_DEMO_SYMBOLS, "LINKUSDT", "1000PEPEUSDT", "1000000MOGUSDT"]
    ]
    # Non-perpetual / non-USDT / non-trading contracts must be filtered out
    symbols.extend(
        [
            {"symbol": "BTCUSDC", "quoteAsset": "USDC", "contractType": "PERPETUAL", "status": "TRADING"},
            {"symbol": "BTCUSDT_260327", "quoteAsset": "USDT", "contractType": "CURRENT_QUARTER", "status": "TRADING"},
            {"symbol": "OLDCOINUSDT", "quoteAsset": "USDT", "contractType": "PERPETUAL", "status": "SETTLING"},
        ]
    )
    if extra_symbols:
        symbols.extend(extra_symbols)
    return {"symbols": symbols}


def _mock_binance_handler(paths: dict[str, list[float]], exchange_payload: dict[str, Any] | None = None):
    ex_info = exchange_payload if exchange_payload is not None else _exchange_info_payload()
    klines_by_sym = {sym: _make_binance_klines(pxs) for sym, pxs in paths.items()}
    premium = [
        {
            "symbol": sym,
            "markPrice": f"{pxs[-1]:.8f}",
            "indexPrice": f"{pxs[-1]:.8f}",
            "lastFundingRate": "0.0002" if sym == "LINKUSDT" else ("0.00025" if sym == "1000PEPEUSDT" else "0.0001"),
        }
        for sym, pxs in paths.items()
    ]
    ticker = [{"symbol": sym, "priceChangePercent": "2.5"} for sym in paths]

    def handler(method: str, url: str, params=None, json_body=None, **kwargs):
        if "exchangeInfo" in url:
            return _FakeResponse(200, ex_info)
        if "premiumIndex" in url:
            return _FakeResponse(200, premium)
        if "ticker/24hr" in url:
            return _FakeResponse(200, ticker)
        if "klines" in url:
            sym = (params or {}).get("symbol", "")
            limit = int((params or {}).get("limit", 1500))
            end_time = (params or {}).get("endTime")
            rows = klines_by_sym.get(sym, [])
            if end_time is not None:
                rows = [r for r in rows if int(r[0]) <= int(end_time)]
            return _FakeResponse(200, rows[-limit:])
        raise AssertionError(f"Unexpected HTTP call in mock handler: {method} {url}")

    return handler


# ---------------------------------------------------------------------------
# 1. Symbol normalisation / validation (mocked exchangeInfo & offline regex)
# ---------------------------------------------------------------------------
def test_symbol_normalization_validation_and_exchange_info_cache() -> None:
    payload = _exchange_info_payload()
    session = _FakeSession(_mock_binance_handler({"BTCUSDT": [100_000.0]}, exchange_payload=payload))
    provider = MarketDataProvider(
        sample_dir=SAMPLE_DIR,
        session=session,  # type: ignore[arg-type]
        cache=TTLCache(default_ttl_seconds=60.0),
    )

    # 'pepe' resolves to '1000PEPEUSDT' when 1000PEPEUSDT is listed
    res_pepe = provider.resolve_symbol("pepe", mode="auto")
    assert res_pepe.symbol == "1000PEPEUSDT"
    assert res_pepe.validated_against_live_list is True

    # Cached exchangeInfo: subsequent calls do NOT hit /fapi/v1/exchangeInfo again
    res_link = provider.resolve_symbol("link-usdt", mode="auto")
    assert res_link.symbol == "LINKUSDT"
    res_mog = provider.resolve_symbol("mog", mode="auto")
    assert res_mog.symbol == "1000000MOGUSDT"
    assert sum(1 for _, u, _, _ in session.calls if "exchangeInfo" in u) == 1

    # Filtered-out contracts (non-USDT, quarterly, non-TRADING) & unknown coins are rejected
    with pytest.raises(ValueError):
        provider.resolve_symbol("BTCUSDC", mode="auto")
    with pytest.raises(ValueError):
        provider.resolve_symbol("OLDCOIN", mode="auto")
    with pytest.raises(ValueError):
        provider.resolve_symbol("DOESNOTEXIST", mode="auto")

    # Prefer exact PEPEUSDT when listed alongside/instead of 1000PEPEUSDT
    assert match_binance_usdtm_symbol("pepe", ["BTCUSDT", "PEPEUSDT", "1000PEPEUSDT"]) == "PEPEUSDT"
    assert match_binance_usdtm_symbol("1000pepe", ["BTCUSDT", "1000PEPEUSDT"]) == "1000PEPEUSDT"


def test_offline_demo_mode_validates_regex_without_network() -> None:
    def forbid(*args, **kwargs):
        raise AssertionError("Demo mode must not call exchangeInfo!")

    provider = MarketDataProvider(
        sample_dir=SAMPLE_DIR,
        session=_FakeSession(forbid),  # type: ignore[arg-type]
        cache=TTLCache(default_ttl_seconds=60.0),
    )

    assert provider.resolve_symbol("pepe", mode="demo").symbol == "PEPEUSDT"
    assert provider.resolve_symbol("op", mode="demo").symbol == "OPUSDT"
    assert provider.resolve_symbol("1000pepe", mode="demo").symbol == "1000PEPEUSDT"
    assert validate_symbol_format("link/usdt") == "LINKUSDT"

    for bad in ("", "A", "BTCUSDC", "PEPE!", "A" * 21):
        with pytest.raises(ValueError):
            provider.resolve_symbol(bad, mode="demo")


def test_exchange_info_451_falls_back_to_format_validation() -> None:
    def handler(method, url, **kwargs):
        return _FakeResponse(451, text="Geo-blocked")

    provider = MarketDataProvider(
        sample_dir=SAMPLE_DIR,
        session=_FakeSession(handler),  # type: ignore[arg-type]
        cache=TTLCache(default_ttl_seconds=60.0),
    )
    res = provider.resolve_symbol("LINK", mode="auto")
    assert res.symbol == "LINKUSDT"
    assert res.validated_against_live_list is False
    assert res.used_fallback_format_only is True


# ---------------------------------------------------------------------------
# 2. Pre-trade with a non-sample coin in live mode -> GO / REDUCE SIZE / NO-GO
# ---------------------------------------------------------------------------
def test_pre_trade_non_sample_coin_live_mode_verdicts() -> None:
    paths = _synthetic_price_paths(hours=960, link_beta=1.6)
    session = _FakeSession(_mock_binance_handler(paths))
    provider = MarketDataProvider(
        sample_dir=SAMPLE_DIR,
        session=session,  # type: ignore[arg-type]
        cache=TTLCache(default_ttl_seconds=60.0),
    )

    sym_res = provider.resolve_symbol("link", mode="auto")
    assert sym_res.symbol == "LINKUSDT"

    bundle = provider.get_market_data([sym_res.symbol], mode="auto", lookback_days=40)
    assert bundle.source == "binance"
    assert bundle.is_live is True
    assert "LINKUSDT" in bundle.snapshots
    assert len(bundle.candles["LINKUSDT"]) == 960

    link_snap = bundle.snapshots["LINKUSDT"]
    expected_atr = compute_daily_atr(bundle.candles["LINKUSDT"])
    assert link_snap.atr_1d == pytest.approx(expected_atr, rel=1e-9)
    assert link_snap.funding_rate_8h == pytest.approx(0.0002)

    account = AccountState(equity=10_000.0)
    rules = RuleConfig(max_risk_per_trade_pct=2.0, max_total_open_risk_pct=6.0, max_leverage=10.0, mandatory_stop=True)
    mark = link_snap.mark_price

    # 1. GO: 1% risk = $100 with 5% stop distance -> notional $2,000, lev 3x
    go_trade = PositionInput(
        symbol="LINKUSDT",
        side="long",
        entry_price=mark,
        notional=2_000.0,
        leverage=3.0,
        margin_mode="isolated",
        stop_price=mark * 0.95,
    )
    res_go = pre_trade_check(go_trade, [], account, bundle, rules)
    assert res_go.verdict == "GO"
    assert res_go.after_var_95_usd > 0.0
    assert res_go.after_btc_beta_exposure_usd == pytest.approx(2_000.0 * 1.6, rel=0.05)
    assert res_go.proposed_analysis.atr_1d == pytest.approx(expected_atr, rel=1e-9)
    assert res_go.proposed_analysis.margin_tier_source == "_DEFAULT"
    assert len(res_go.sizing_options) == 4

    # 2. REDUCE SIZE: 5% risk = $500 > 2% limit
    red_trade = PositionInput(
        symbol="LINKUSDT",
        side="long",
        entry_price=mark,
        notional=10_000.0,
        leverage=3.0,
        margin_mode="isolated",
        stop_price=mark * 0.95,
    )
    res_red = pre_trade_check(red_trade, [], account, bundle, rules)
    assert res_red.verdict == "REDUCE SIZE"
    assert res_red.recommended_quantity is not None
    assert res_red.recommended_quantity * mark <= 4_000.0 + 1e-6

    # 3. NO-GO: 25x isolated long with stop 10% below entry -> stop beyond LP
    nogo_trade = PositionInput(
        symbol="LINKUSDT",
        side="long",
        entry_price=mark,
        notional=2_000.0,
        leverage=25.0,
        margin_mode="isolated",
        stop_price=mark * 0.90,
    )
    res_nogo = pre_trade_check(nogo_trade, [], account, bundle, rules)
    assert res_nogo.verdict == "NO-GO"
    assert res_nogo.proposed_analysis.stop_beyond_liquidation is True


# ---------------------------------------------------------------------------
# 3. Positions with a non-sample coin in live mode get ATR/beta & _DEFAULT tier
# ---------------------------------------------------------------------------
def test_positions_non_sample_coin_live_mode_atr_beta_and_default_tier() -> None:
    paths = _synthetic_price_paths(hours=960, link_beta=1.6, pepe_beta=2.2)
    session = _FakeSession(_mock_binance_handler(paths))
    provider = MarketDataProvider(
        sample_dir=SAMPLE_DIR,
        session=session,  # type: ignore[arg-type]
        cache=TTLCache(default_ttl_seconds=60.0),
    )

    bundle = provider.get_market_data(["LINKUSDT", "1000PEPEUSDT"], mode="auto", lookback_days=40)
    assert set(bundle.candles.keys()) >= {"BTCUSDT", "LINKUSDT", "1000PEPEUSDT"}

    account = AccountState(equity=10_000.0)
    link_mark = bundle.snapshots["LINKUSDT"].mark_price
    pepe_mark = bundle.snapshots["1000PEPEUSDT"].mark_price
    positions = [
        PositionInput(
            symbol="LINKUSDT",
            side="long",
            entry_price=link_mark * 0.98,
            notional=3_000.0,
            leverage=4.0,
            stop_price=link_mark * 0.90,
        ),
        PositionInput(
            symbol="1000PEPEUSDT",
            side="long",
            entry_price=pepe_mark * 0.99,
            notional=2_000.0,
            leverage=4.0,
            stop_price=pepe_mark * 0.90,
        ),
    ]
    analyses = analyze_positions(positions, account, market=bundle)
    assert analyses[0].used_entry_proxy is False
    assert analyses[0].margin_tier_source == "_DEFAULT"
    assert analyses[0].atr_1d == pytest.approx(compute_daily_atr(bundle.candles["LINKUSDT"]), rel=1e-9)
    assert analyses[1].margin_tier_source == "_DEFAULT"

    tier, src = lookup_margin_tier_with_source("LINKUSDT", 3_000.0)
    assert src == "_DEFAULT"
    assert tier.mmr == 0.005

    port = analyze_portfolio(analyses, account, market=bundle)
    assert port.betas_vs_btc["LINKUSDT"] == pytest.approx(1.6, abs=1e-3)
    assert port.betas_vs_btc["1000PEPEUSDT"] == pytest.approx(2.2, abs=1e-3)
    assert port.corr_30d["LINKUSDT"]["1000PEPEUSDT"] > 0.95
    assert port.var_95_hist_usd > 0.0


def test_hyperliquid_fallback_maps_1000pepe_to_kpepe() -> None:
    hl_meta = [
        {"universe": [{"name": "BTC"}, {"name": "kPEPE"}]},
        [
            {"funding": "0.00001", "prevDayPx": "100000", "oraclePx": "101000", "markPx": "101000"},
            {"funding": "0.00002", "prevDayPx": "0.011", "oraclePx": "0.012", "markPx": "0.012"},
        ],
    ]
    t0 = 1761868800000
    hl_candles = [
        {"t": t0 + i * 3600000, "o": "0.011", "c": f"{0.011 + i * 0.0002}", "h": "0.013", "l": "0.010", "v": "100", "n": 10}
        for i in range(48)
    ]

    def handler(method, url, json_body=None, **kwargs):
        if "fapi.binance.com" in url:
            return _FakeResponse(451, text="restricted")
        if (json_body or {}).get("type") == "metaAndAssetCtxs":
            return _FakeResponse(200, hl_meta)
        if (json_body or {}).get("type") == "candleSnapshot":
            assert json_body["req"]["coin"] in ("BTC", "kPEPE")
            return _FakeResponse(200, hl_candles)
        raise AssertionError(f"Unexpected call: {url}")

    provider = MarketDataProvider(
        sample_dir=SAMPLE_DIR,
        session=_FakeSession(handler),  # type: ignore[arg-type]
        cache=TTLCache(default_ttl_seconds=0.0),
    )
    bundle = provider.get_market_data(["1000PEPEUSDT"], mode="auto", lookback_days=2)
    assert bundle.source == "hyperliquid"
    assert bundle.snapshots["1000PEPEUSDT"].mark_price == pytest.approx(0.012)
    assert len(bundle.candles["1000PEPEUSDT"]) == 48


def test_missing_symbol_sets_used_entry_proxy_flag() -> None:
    a = analyze_position(
        PositionInput(symbol="MYSTERYUSDT", side="long", entry_price=10.0, quantity=50.0, leverage=3.0, stop_price=9.0),
        AccountState(equity=10_000.0),
        market=None,
    )
    assert a.used_entry_proxy is True
    assert a.margin_tier_source == "_DEFAULT"


# ---------------------------------------------------------------------------
# 4. AppTest: Demo-mode non-sample coin message + Live-mode extra coin submit
#    + Positions page proxy & default-tier UI visibility
# ---------------------------------------------------------------------------
def test_apptest_pre_trade_demo_mode_non_sample_coin_message_and_proxy_run(monkeypatch: pytest.MonkeyPatch) -> None:
    from riskguard.ui.state import fetch_market_bundle_cached, get_market_provider

    monkeypatch.setenv("RG_DATA_MODE", "demo")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    def forbid(*args, **kwargs):
        raise AssertionError("Demo mode AppTest must never call network!")

    monkeypatch.setattr(requests.Session, "request", forbid)
    fetch_market_bundle_cached.clear()
    get_market_provider().cache.clear()

    at = AppTest.from_file(str(ROOT / "pages" / "4_Pre_Trade_and_Sizing.py"), default_timeout=40).run()
    assert not at.exception

    # Type a non-sample coin in Demo mode -> clear warning that Live auto mode is needed for its price
    at.text_input(key="pt_other_symbol").set_value("LINK").run()
    assert not at.exception
    warnings = [w.value for w in at.warning]
    assert any("LINKUSDT" in w and "Live auto" in w for w in warnings)

    # User can still enter an entry price manually and evaluate with explicit proxy warning & default margin tier
    at.number_input(key="pt_entry_LINKUSDT").set_value(15.0)
    at.number_input(key="pt_stop_LINKUSDT_long").set_value(14.0)
    at.number_input(key="pt_notional").set_value(1500.0)
    at.number_input(key="pt_leverage").set_value(3.0)
    next(b for b in at.button if b.label == "Evaluate Pre-Trade Risk").click().run()

    assert not at.exception
    assert any("Verdict:" in m.value for m in at.markdown)
    assert any("LINKUSDT @ $15" in c.value for c in at.caption)
    assert any("Proxy Warning:" in w.value and "LINKUSDT" in w.value for w in at.warning)
    pm_df = next(d.value for d in at.dataframe if "Field" in d.value.columns).set_index("Field")
    assert "_DEFAULT (default)" in pm_df.loc["Margin Tier", "Value"]

    fetch_market_bundle_cached.clear()
    get_market_provider().cache.clear()


def test_apptest_pre_trade_live_mode_type_extra_coin_and_submit(monkeypatch: pytest.MonkeyPatch) -> None:
    from riskguard.ui.state import fetch_market_bundle_cached, get_market_provider

    paths = _synthetic_price_paths(hours=240, pepe_beta=2.2)
    handler = _mock_binance_handler(paths)

    def fake_request(self, method, url, params=None, json=None, timeout=None):
        return handler(method, url, params=params, json_body=json)

    monkeypatch.setenv("RG_DATA_MODE", "auto")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(requests.Session, "request", fake_request)
    fetch_market_bundle_cached.clear()
    get_market_provider().cache.clear()

    at = AppTest.from_file(str(ROOT / "pages" / "4_Pre_Trade_and_Sizing.py"), default_timeout=40).run()
    assert not at.exception

    # Invalid coin rejected cleanly against mocked exchangeInfo
    at.text_input(key="pt_other_symbol").set_value("NOTACOIN").run()
    assert not at.exception
    assert any("not found among active Binance USDT-M" in e.value for e in at.error)

    # Type 'pepe' -> resolves to 1000PEPEUSDT via mocked exchangeInfo, fetches mark & candles, and submits
    at.text_input(key="pt_other_symbol").set_value("pepe").run()
    assert not at.exception
    assert any("1000PEPEUSDT" in i.value for i in at.info)

    expected_mark = paths["1000PEPEUSDT"][-1]
    assert at.number_input(key="pt_entry_1000PEPEUSDT").value == pytest.approx(expected_mark, rel=1e-4)

    at.number_input(key="pt_notional").set_value(1500.0)
    at.number_input(key="pt_leverage").set_value(3.0)
    next(b for b in at.button if b.label == "Evaluate Pre-Trade Risk").click().run()

    assert not at.exception
    assert any("Verdict:" in m.value for m in at.markdown)
    assert any("1000PEPEUSDT" in c.value for c in at.caption)
    pm_df = next(d.value for d in at.dataframe if "Field" in d.value.columns).set_index("Field")
    assert "_DEFAULT (default)" in pm_df.loc["Margin Tier", "Value"]
    ba_df = next(d.value for d in at.dataframe if "Before" in d.value.columns).set_index("Metric")
    assert ba_df.loc["Gross Exposure ($)", "Delta"] == pytest.approx(1500.0)
    assert ba_df.loc["BTC-Beta Net ($)", "Delta"] == pytest.approx(1500.0 * 2.2, rel=0.05)

    fetch_market_bundle_cached.clear()
    get_market_provider().cache.clear()


def test_apptest_positions_live_mode_non_sample_coin_and_exposure(monkeypatch: pytest.MonkeyPatch) -> None:
    from riskguard.ui.state import fetch_market_bundle_cached, get_market_provider

    paths = _synthetic_price_paths(hours=240, link_beta=1.6)
    handler = _mock_binance_handler(paths)

    def fake_request(self, method, url, params=None, json=None, timeout=None):
        return handler(method, url, params=params, json_body=json)

    monkeypatch.setenv("RG_DATA_MODE", "auto")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(requests.Session, "request", fake_request)
    fetch_market_bundle_cached.clear()
    get_market_provider().cache.clear()

    at = AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=40).run()
    assert not at.exception
    at.switch_page("pages/2_Positions.py").run()
    assert not at.exception

    link_mark = paths["LINKUSDT"][-1]
    at.session_state["positions_editor"] = {
        "edited_rows": {},
        "added_rows": [
            {
                "symbol": "LINKUSDT",
                "side": "long",
                "entry_price": link_mark * 0.98,
                "quantity": 100.0,
                "leverage": 5.0,
                "margin_mode": "isolated",
                "stop_price": link_mark * 0.90,
            }
        ],
        "deleted_rows": [],
    }
    at.run()
    assert not at.exception
    assert not any("LINKUSDT" in w.value for w in at.warning)

    pos_df = next(d.value for d in at.dataframe if "Margin Tier" in d.value.columns).set_index("Symbol")
    assert pos_df.loc["LINKUSDT", "Mark Source"] == "binance"
    assert pos_df.loc["LINKUSDT", "Margin Tier"] == "_DEFAULT (default)"
    assert pos_df.loc["LINKUSDT", "Dist to LP (ATR)"] is not None

    at.switch_page("pages/3_Exposure_and_Stress.py").run()
    assert not at.exception
    beta_df = next(d.value for d in at.dataframe if "beta" in d.value.columns).set_index("Symbol")
    assert beta_df.loc["LINKUSDT", "beta"] == pytest.approx(1.6, abs=0.05)

    fetch_market_bundle_cached.clear()
    get_market_provider().cache.clear()


def test_apptest_positions_shows_entry_proxy_and_default_margin_tier(monkeypatch: pytest.MonkeyPatch) -> None:
    from riskguard.ui.state import fetch_market_bundle_cached, get_market_provider

    monkeypatch.setenv("RG_DATA_MODE", "demo")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(requests.Session, "request", lambda *a, **k: (_ for _ in ()).throw(AssertionError("No network")))
    fetch_market_bundle_cached.clear()
    get_market_provider().cache.clear()

    at = AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=40).run()
    assert not at.exception
    at.switch_page("pages/2_Positions.py").run()
    assert not at.exception

    # Add a non-sample coin row in Demo mode -> falls back to entry-price proxy and uses _DEFAULT margin tier
    at.session_state["positions_editor"] = {
        "edited_rows": {},
        "added_rows": [
            {
                "symbol": "AVAXUSDT",
                "side": "long",
                "entry_price": 30.0,
                "quantity": 40.0,
                "leverage": 5.0,
                "margin_mode": "isolated",
                "stop_price": 27.0,
            }
        ],
        "deleted_rows": [],
    }
    at.run()
    assert not at.exception
    assert any("Entry-price proxy used for:" in w.value and "AVAXUSDT" in w.value for w in at.warning)

    pos_df = next(d.value for d in at.dataframe if "Margin Tier" in d.value.columns).set_index("Symbol")
    assert pos_df.loc["AVAXUSDT", "Mark Source"] == "Entry proxy"
    assert pos_df.loc["AVAXUSDT", "Margin Tier"] == "_DEFAULT (default)"
    assert pos_df.loc["DOGEUSDT", "Margin Tier"] == "DOGEUSDT"
    assert pos_df.loc["DOGEUSDT", "Mark Source"] == "sample"

    fetch_market_bundle_cached.clear()
    get_market_provider().cache.clear()
