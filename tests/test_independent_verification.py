"""Independent verification suite: hand-calculated expectations for every risk formula and feature.

Every expected number below is derived by hand in the comments (not copied from the implementation).
No network access: HTTP is mocked, the web app runs in offline demo mode, and no API key is used.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import anthropic
import httpx2 as httpx  # the anthropic 2.x SDK ships httpx2 (plain httpx is not installed with it)
import numpy as np
import pandas as pd
import pytest
import requests
from scipy import integrate, stats

from riskguard.ai.coach import DEFAULT_MODEL, build_offline_coach_report, run_coach_turn
from riskguard.ai.prompts import COACH_PROMPT
from riskguard.ai.tools import CoachContext, _bounded_json, build_coach_tools
from riskguard.analytics import portfolio as portfolio_mod
from riskguard.analytics.drawdown import analytical_risk_of_ruin, compute_drawdown_plan, estimate_risk_of_ruin
from riskguard.analytics.journal import analyze_journal, journal_reference_time
from riskguard.analytics.margin import (
    DEFAULT_MARGIN_TIERS,
    compute_liquidation_price,
    compute_maintenance_margin,
    lookup_margin_tier,
)
from riskguard.analytics.portfolio import (
    analyze_portfolio,
    compute_cornish_fisher_var_cvar,
    compute_historical_var_cvar,
)
from riskguard.analytics.positions import analyze_position, analyze_positions, liquidation_label
from riskguard.analytics.rules import evaluate_circuit_breaker, evaluate_rules
from riskguard.analytics.sizing import max_safe_leverage_for_stop, pre_trade_check, recommend_sizes
from riskguard.cli import format_cli_report
from riskguard.data.cache import TTLCache
from riskguard.data.providers import MarketDataBundle, MarketDataProvider, SymbolMarketData
from riskguard.models import AccountState, JournalTrade, MarginTier, PositionInput, RuleConfig

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DIR = ROOT / "data" / "sample"
T0 = datetime(2025, 10, 1, 0, 0, tzinfo=timezone.utc)


# --------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------
def _snap(symbol: str, mark: float, atr: float, funding_8h: float = 0.0001) -> SymbolMarketData:
    return SymbolMarketData(
        symbol=symbol,
        mark_price=mark,
        atr_1d=atr,
        funding_rate_8h=funding_8h,
        funding_rate_daily=funding_8h * 3,
        as_of=T0,
    )


def _candles(prices: list[float]) -> pd.DataFrame:
    times = [T0 + timedelta(hours=i) for i in range(len(prices))]
    return pd.DataFrame(
        {
            "open_time": times,
            "open": prices,
            "high": [p * 1.001 for p in prices],
            "low": [p * 0.999 for p in prices],
            "close": prices,
            "volume": [1.0] * len(prices),
        }
    )


def _beta_bundle(last_prices: dict[str, float], betas: dict[str, float], hours: int = 240) -> MarketDataBundle:
    """Hourly candles where each symbol's hourly return is EXACTLY beta x BTC's hourly return.

    BTC hourly sigma is 0.1%, so every daily sigma is < 4% and the correlation->1 move is the 10% floor.
    """
    rng = np.random.default_rng(11)
    btc_r = rng.normal(0.0, 0.001, hours - 1)
    candles: dict[str, pd.DataFrame] = {}
    snaps: dict[str, SymbolMarketData] = {}
    for sym, last in last_prices.items():
        b = betas[sym]
        path = [1.0]
        for r in btc_r:
            path.append(path[-1] * (1.0 + b * r))
        scale = last / path[-1]
        prices = [p * scale for p in path]
        candles[sym] = _candles(prices)
        snaps[sym] = _snap(sym, prices[-1], atr=prices[-1] * 0.04)
    return MarketDataBundle(
        source="sample",
        badge_label="TEST",
        reason="unit test",
        is_live=False,
        fetched_at=T0,
        snapshots=snaps,
        candles=candles,
    )


def _trade(
    open_dt: datetime,
    close_dt: datetime,
    pnl: float,
    *,
    sym: str = "BTCUSDT",
    side: str = "long",
    qty: float = 1.0,
    entry: float = 100.0,
    stop: float | None = 98.0,
    source: str | None = None,
) -> JournalTrade:
    sign = 1.0 if side == "long" else -1.0
    exit_px = entry + sign * pnl / qty  # zero fees: pnl = sign * qty * (exit - entry)
    return JournalTrade(
        opened_at=open_dt,
        closed_at=close_dt,
        symbol=sym,
        side=side,  # type: ignore[arg-type]
        quantity=qty,
        entry_price=entry,
        exit_price=exit_px,
        planned_stop=stop,
        fees=0.0,
        pnl=pnl,
        source=source,
    )


# --------------------------------------------------------------------------------------------
# 1. Liquidation price, tiers, cumB continuity
# --------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "wb,q,ep,side,mmr,cumb,expected",
    [
        # BTC 10x long: Q=0.1, EP=60,000 -> notional 6,000, WB = 600, MMR 0.4%
        # LP = (600 + 0 - 0.1*60000) / (0.1*0.004 - 0.1) = -5400 / -0.0996 = 54,216.8675
        (600.0, 0.1, 60_000.0, "long", 0.004, 0.0, 5400.0 / 0.0996),
        # ETH 20x short: Q=2, EP=3,000 -> 6,000, WB = 300
        # LP = (300 + 0 + 2*3000) / (2*0.004 + 2) = 6300 / 2.008 = 3,137.4502
        (300.0, 2.0, 3_000.0, "short", 0.004, 0.0, 6300.0 / 2.008),
        # SOL 50x long: Q=10, EP=100 -> 1,000, WB = 20, MMR 0.5%
        # LP = (20 - 1000) / (10*0.005 - 10) = -980 / -9.95 = 98.4925
        (20.0, 10.0, 100.0, "long", 0.005, 0.0, 980.0 / 9.95),
        # BTC 5x long in tier 3 (notional 300k): MMR 1.0%, cumB = 50000*0.001 + 250000*0.005 = 1,300
        # LP = (60000 + 1300 - 300000) / (3*0.01 - 3) = -238700 / -2.97 = 80,370.3704
        (60_000.0, 3.0, 100_000.0, "long", 0.010, 1300.0, 238_700.0 / 2.97),
        # Same, short: LP = (60000 + 1300 + 300000) / (3*0.01 + 3) = 361300 / 3.03 = 119,240.9241
        (60_000.0, 3.0, 100_000.0, "short", 0.010, 1300.0, 361_300.0 / 3.03),
    ],
)
def test_liquidation_price_hand_calculated(wb, q, ep, side, mmr, cumb, expected) -> None:
    assert compute_liquidation_price(wb, q, ep, side, mmr, cumb) == pytest.approx(expected, rel=1e-12)


def test_tier_lookup_and_cumb_continuity() -> None:
    btc = DEFAULT_MARGIN_TIERS["BTCUSDT"]
    # cumB_k = cumB_{k-1} + cap_{k-1} * (MMR_k - MMR_{k-1})
    # tier2: 0 + 50,000*(0.005-0.004) = 50; tier3: 50 + 250,000*0.005 = 1,300
    # tier4: 1,300 + 1,000,000*0.015 = 16,300; tier5: 16,300 + 5,000,000*0.025 = 141,300
    assert [t.cum_b for t in btc] == pytest.approx([0.0, 50.0, 1300.0, 16300.0, 141300.0])
    tier = lookup_margin_tier("BTCUSDT", 300_000.0)
    assert (tier.mmr, tier.cum_b) == (0.010, 1300.0)
    # MM is continuous at every cap: MM = N*MMR - cumB evaluated with both adjacent tiers
    for lower, upper in zip(btc[:-1], btc[1:]):
        n = lower.max_notional
        assert compute_maintenance_margin(n, lower) == pytest.approx(compute_maintenance_margin(n, upper), abs=1e-6)
    # e.g. at 50,000: 50,000*0.004 = 200 = 50,000*0.005 - 50
    assert compute_maintenance_margin(50_000.0, btc[0]) == pytest.approx(200.0)
    # Symbols without a table use the _DEFAULT brackets, and normalisation works ('pepe' -> PEPEUSDT)
    assert lookup_margin_tier("pepe", 1_000.0).mmr == 0.005


def test_isolated_long_1x_has_no_liquidation_price_and_says_why() -> None:
    # 1x long: WB = Q*EP -> numerator = cumB = 0 -> LP = 0 -> reported as none
    acct = AccountState(equity=10_000.0)
    pos = PositionInput(symbol="SOLUSDT", side="long", entry_price=100.0, quantity=10.0, leverage=1.0, stop_price=90.0)
    a = analyze_position(pos, acct, market={"SOLUSDT": _snap("SOLUSDT", 100.0, 5.0)})
    assert a.liquidation_price is None
    assert "isolated margin covers" in (a.liquidation_note or "")
    assert liquidation_label(a) == a.liquidation_note


def test_cross_liquidation_with_small_equity_and_no_lp_with_large_equity() -> None:
    # Equity 1,000, single cross BTC long Q=0.1 @ 60,000 (mark = entry -> uPnL 0, no other positions)
    # WB = 1,000 -> LP = (1000 - 6000) / (0.1*0.004 - 0.1) = -5000 / -0.0996 = 50,200.8032
    pos = PositionInput(symbol="BTCUSDT", side="long", entry_price=60_000.0, quantity=0.1, leverage=10.0,
                        margin_mode="cross", stop_price=57_000.0)
    m = {"BTCUSDT": _snap("BTCUSDT", 60_000.0, 2_000.0)}
    small = analyze_positions([pos], AccountState(equity=1_000.0), market=m)[0]
    assert small.liquidation_price == pytest.approx(5000.0 / 0.0996, rel=1e-12)
    # distance = (60000 - 50200.80) / 60000 = 16.332%; in ATRs = 9799.197 / 2000 = 4.8996
    assert small.distance_to_liq_pct == pytest.approx((60_000 - 5000 / 0.0996) / 600.0, rel=1e-9)
    assert small.distance_to_liq_atr == pytest.approx((60_000 - 5000 / 0.0996) / 2000.0, rel=1e-9)

    # Equity 10,000 > notional 6,000: numerator 10000 - 6000 > 0 -> LP < 0 -> no liquidation price
    big = analyze_positions([pos], AccountState(equity=10_000.0), market=m)[0]
    assert big.liquidation_price is None
    assert big.liquidation_note == "none at current equity (cross)"
    assert any("no liquidation price at current equity" in w for w in big.warnings)


def test_cross_wallet_excludes_ring_fenced_isolated_equity() -> None:
    # Equity 10,000. Isolated SOL long Q=10 @ 100, 10x -> margin 100; mark 95 -> uPnL -50 -> isolated equity 50.
    # Cross BTC long Q=0.5 @ 60,000 (mark = entry): notional 30,000 -> tier1 MMR 0.4%.
    # WB_cross = 10,000 - uPnL_btc(0) - 50 = 9,950
    # LP = (9950 - 30000) / (0.5*0.004 - 0.5) = -20050 / -0.498 = 40,261.0442
    acct = AccountState(equity=10_000.0)
    iso = PositionInput(symbol="SOLUSDT", side="long", entry_price=100.0, quantity=10.0, leverage=10.0, stop_price=90.0)
    crs = PositionInput(symbol="BTCUSDT", side="long", entry_price=60_000.0, quantity=0.5, leverage=5.0,
                        margin_mode="cross", stop_price=55_000.0)
    m = {"SOLUSDT": _snap("SOLUSDT", 95.0, 4.0), "BTCUSDT": _snap("BTCUSDT", 60_000.0, 2_000.0)}
    res = analyze_positions([iso, crs], acct, market=m)
    assert res[1].liquidation_price == pytest.approx(20050.0 / 0.498, rel=1e-12)
    # Isolated SOL LP is independent of the account: (100 - 1000) / (10*0.005 - 10) = 900/9.95 = 90.4523
    assert res[0].liquidation_price == pytest.approx(900.0 / 9.95, rel=1e-12)


# --------------------------------------------------------------------------------------------
# 2. Risk to stop, crossed stops, funding sign
# --------------------------------------------------------------------------------------------
def test_risk_to_stop_long_short_and_reward_to_risk() -> None:
    acct = AccountState(equity=1_000.0)
    # Long Q=2, mark 105, stop 100 -> risk = 2 * (105 - 100) = $10 = 1.0% of 1,000; TP 120 -> reward 30 -> 3.0R
    long_p = PositionInput(symbol="SOLUSDT", side="long", entry_price=100.0, quantity=2.0, leverage=2.0,
                           stop_price=100.0, take_profit=120.0)
    a = analyze_position(long_p, acct, market={"SOLUSDT": _snap("SOLUSDT", 105.0, 5.0)})
    assert a.risk_to_stop_usd == pytest.approx(10.0)
    assert a.risk_to_stop_pct == pytest.approx(1.0)
    assert a.reward_to_risk == pytest.approx(3.0)
    assert a.unrealized_pnl == pytest.approx(10.0)  # 2 * (105 - 100)
    # Short Q=3, mark 50, stop 55 -> risk = 3 * (55 - 50) = $15 = 1.5%
    short_p = PositionInput(symbol="XRPUSDT", side="short", entry_price=52.0, quantity=3.0, leverage=2.0, stop_price=55.0)
    b = analyze_position(short_p, acct, market={"XRPUSDT": _snap("XRPUSDT", 50.0, 2.0)})
    assert b.risk_to_stop_usd == pytest.approx(15.0)
    assert b.unrealized_pnl == pytest.approx(6.0)  # -1 * 3 * (50 - 52)
    # Trailing stop in profit: long entry 100, stop 102, mark 110 -> still gives back 2*(110-102) = 16
    trail = PositionInput(symbol="SOLUSDT", side="long", entry_price=100.0, quantity=2.0, leverage=2.0, stop_price=102.0)
    c = analyze_position(trail, acct, market={"SOLUSDT": _snap("SOLUSDT", 110.0, 5.0)})
    assert c.risk_to_stop_usd == pytest.approx(16.0)
    assert c.invalid_stop_direction is False and c.danger_level == "OK"


@pytest.mark.parametrize(
    "side,entry,stop,mark",
    [
        ("long", 100.0, 95.0, 93.0),   # long: mark already below the stop
        ("long", 100.0, 95.0, 95.0),   # long: mark exactly at the stop
        ("short", 100.0, 105.0, 106.0),  # short: mark already above the stop
        ("long", 100.0, 110.0, 105.0),  # stop above entry AND above mark (wrong side)
    ],
)
def test_stop_already_crossed_is_flagged_and_unprotected(side, entry, stop, mark) -> None:
    acct = AccountState(equity=10_000.0)
    pos = PositionInput(symbol="SOLUSDT", side=side, entry_price=entry, quantity=10.0, leverage=3.0, stop_price=stop)
    a = analyze_position(pos, acct, market={"SOLUSDT": _snap("SOLUSDT", mark, 5.0)})
    assert a.invalid_stop_direction is True
    assert a.danger_level == "DANGER"
    assert any("stop already crossed" in w for w in a.warnings)
    # Honest risk: the stop does not bound the loss -> unbounded (None), never $0
    assert a.risk_to_stop_usd is None and a.risk_to_stop_pct is None
    port = analyze_portfolio([a], acct)
    assert port.unstopped_positions == 1
    evals, _ = evaluate_rules([a], port, acct, RuleConfig())
    by_id = {e.rule_id: e for e in evals}
    assert by_id["mandatory_stop"].status == "BREACH"
    assert "crossed" in by_id["mandatory_stop"].explanation


@pytest.mark.parametrize(
    "side,rate,expected",
    [
        # daily cost = side * notional * 3 * rate_8h ; notional = 10 * 1000 = 10,000
        ("long", 0.0001, 3.0),     # longs PAY positive funding: +10,000 * 0.0003
        ("short", 0.0001, -3.0),   # shorts RECEIVE positive funding
        ("long", -0.0002, -6.0),   # negative funding: longs receive 10,000 * 0.0006
        ("short", -0.0002, 6.0),   # shorts pay
    ],
)
def test_funding_sign_convention(side, rate, expected) -> None:
    pos = PositionInput(symbol="SOLUSDT", side=side, entry_price=1_000.0, quantity=10.0, leverage=2.0, stop_price=900.0 if side == "long" else 1100.0)
    a = analyze_position(pos, AccountState(equity=50_000.0), market={"SOLUSDT": _snap("SOLUSDT", 1_000.0, 30.0, funding_8h=rate)})
    assert a.funding_cost_daily_usd == pytest.approx(expected)


def test_negative_quantity_is_treated_as_size_and_missing_market_uses_entry_proxy() -> None:
    p = PositionInput.model_validate({"symbol": "eth-usdt", "side": "SELL", "entry_price": 3000, "quantity": -0.5, "stop_price": 3100})
    assert p.symbol == "ETHUSDT" and p.side == "short" and p.quantity == 0.5
    a = analyze_position(p, AccountState(equity=10_000.0), market=None)
    assert a.mark_price == 3000.0
    assert any("using entry price as mark proxy" in w for w in a.warnings)


# --------------------------------------------------------------------------------------------
# 3. VaR / CVaR
# --------------------------------------------------------------------------------------------
def test_historical_var_cvar_exact_quantile() -> None:
    # Losses are exactly 1..100. numpy linear quantile at 0.95: position 99*0.95 = 94.05
    # -> 95 + 0.05*(96 - 95) = 95.05; CVaR = mean(losses >= 95.05) = mean(96..100) = 98
    # at 0.99: position 98.01 -> 99.01; CVaR = mean(100) = 100
    pnl = -np.arange(1.0, 101.0)
    assert compute_historical_var_cvar(pnl, 0.95) == pytest.approx((95.05, 98.0))
    assert compute_historical_var_cvar(pnl, 0.99) == pytest.approx((99.01, 100.0))
    assert compute_historical_var_cvar(np.array([5.0]), 0.95) == (0.0, 0.0)  # < 2 observations


def _normal_tail_reference(mu: float, sigma: float, alpha: float) -> tuple[float, float]:
    var = -(mu + sigma * stats.norm.ppf(1 - alpha))
    # ES as the tail average of the loss quantile, integrated numerically (independent of the closed form)
    es, _ = integrate.quad(lambda u: -(mu + sigma * stats.norm.ppf(1 - u)), alpha, 1, limit=200)
    return var, es / (1 - alpha)


@pytest.mark.parametrize("alpha", [0.95, 0.99])
def test_cornish_fisher_reduces_to_normal_when_skew_and_kurt_are_zero(monkeypatch, alpha) -> None:
    rng = np.random.default_rng(3)
    pnl = rng.normal(25.0, 400.0, 2_000)
    monkeypatch.setattr(portfolio_mod.stats, "skew", lambda *a, **k: 0.0)
    monkeypatch.setattr(portfolio_mod.stats, "kurtosis", lambda *a, **k: 0.0)
    mu, sigma = float(np.mean(pnl)), float(np.std(pnl, ddof=1))
    var_ref, es_ref = _normal_tail_reference(mu, sigma, alpha)
    var_cf, es_cf = compute_cornish_fisher_var_cvar(pnl, alpha)
    assert var_cf == pytest.approx(var_ref, rel=1e-9)
    assert es_cf == pytest.approx(es_ref, rel=1e-6)


def test_cornish_fisher_es_matches_numerical_tail_integral_with_skew_and_kurtosis() -> None:
    rng = np.random.default_rng(5)
    pnl = np.concatenate([rng.normal(0.0, 100.0, 3_000), rng.normal(-350.0, 150.0, 90)])  # left-skewed, fat tail
    mu, sigma = float(np.mean(pnl)), float(np.std(pnl, ddof=1))
    s, k = float(stats.skew(pnl, bias=False)), float(stats.kurtosis(pnl, bias=False))
    assert s < 0 and k > 0

    def cf_loss(u: float) -> float:
        z = stats.norm.ppf(1 - u)
        zc = z + (z**2 - 1) * s / 6 + (z**3 - 3 * z) * k / 24 - (2 * z**3 - 5 * z) * s**2 / 36
        return -(mu + zc * sigma)

    for alpha in (0.95, 0.99):
        es_ref = integrate.quad(cf_loss, alpha, 1, limit=200)[0] / (1 - alpha)
        var_cf, es_cf = compute_cornish_fisher_var_cvar(pnl, alpha)
        assert var_cf == pytest.approx(cf_loss(alpha), rel=1e-9)
        assert es_cf == pytest.approx(es_ref, rel=1e-6)
        assert var_cf > -(mu + sigma * stats.norm.ppf(1 - alpha))  # fat left tail -> CF VaR above normal


# --------------------------------------------------------------------------------------------
# 4. Stress tests
# --------------------------------------------------------------------------------------------
def test_stress_pnl_is_beta_times_shock_times_signed_notional_with_liquidations() -> None:
    bundle = _beta_bundle({"BTCUSDT": 60_000.0, "SOLUSDT": 150.0, "ETHUSDT": 3_000.0}, {"BTCUSDT": 1.0, "SOLUSDT": 2.0, "ETHUSDT": 0.5})
    sol_px = bundle.snapshots["SOLUSDT"].mark_price
    eth_px = bundle.snapshots["ETHUSDT"].mark_price
    acct = AccountState(equity=10_000.0)
    positions = [
        # long SOL $4,000 isolated 5x: LP/EP = (1 - 1/5)/(1 - 0.005) = 0.80402 -> liquidated by a -19.6% move
        PositionInput(symbol="SOLUSDT", side="long", entry_price=sol_px, notional=4_000.0, leverage=5.0, stop_price=sol_px * 0.95),
        # short ETH $3,000 isolated 2x: LP/EP = 1.5/1.004 -> +49% move, never hit here
        PositionInput(symbol="ETHUSDT", side="short", entry_price=eth_px, notional=3_000.0, leverage=2.0, stop_price=eth_px * 1.05),
    ]
    summary = analyze_portfolio(positions, acct, market=bundle)
    assert summary.betas_vs_btc["SOLUSDT"] == pytest.approx(2.0, abs=1e-6)
    assert summary.betas_vs_btc["ETHUSDT"] == pytest.approx(0.5, abs=1e-6)
    # beta-adjusted net = 2.0*4000 - 0.5*3000 = 6,500 ; gross 7,000 ; net 1,000 ; HHI = (4/7)^2 + (3/7)^2 = 25/49
    assert summary.btc_beta_adj_net_usd == pytest.approx(6_500.0, rel=1e-6)
    assert (summary.gross_exposure_usd, summary.net_exposure_usd) == pytest.approx((7_000.0, 1_000.0))
    assert summary.hhi == pytest.approx(25.0 / 49.0)
    assert summary.effective_num_positions == pytest.approx(49.0 / 25.0)

    by_name = {s.scenario: s for s in summary.stress_scenarios}
    # PnL = sum(beta * shock * signed notional) = 2*s*4000 + 0.5*s*(-3000), but an isolated position loses at
    # most its own margin + uPnL (SOL: 4000 / 5x = 800, uPnL 0 -> loss capped at 800):
    #   BTC -5%:  -400 + 75 = -325 ; BTC -10%: -800 + 150 = -650 ; BTC -20%: max(-1600, -800) + 300 = -500
    for name, expected, liq in (("BTC -5%", -325.0, []), ("BTC -10%", -650.0, ["SOLUSDT"]), ("BTC -20%", -500.0, ["SOLUSDT"])):
        assert by_name[name].portfolio_pnl_usd == pytest.approx(expected, rel=1e-6)
        assert by_name[name].liquidated_symbols == liq
        assert by_name[name].shocked_equity_usd == pytest.approx(10_000.0 + expected, rel=1e-9)
    # Alt crash: SOL -15%, ETH -15% -> -600 + 450 = -150 (SOL -15% is above its -19.6% LP)
    assert by_name["Alt Crash (1.5x BTC)"].portfolio_pnl_usd == pytest.approx(-150.0)
    assert by_name["Alt Crash (1.5x BTC)"].liquidated_symbols == []
    # Correlation -> 1: daily sigmas < 4% so every move is the 10% floor. Down: -400 + 300 = -100; up: +100.
    # Net long book -> the crash is the adverse direction
    assert by_name["Correlation -> 1 Crash"].portfolio_pnl_usd == pytest.approx(-100.0)
    # Funding spike 0.30%/day for 7 days: long SOL pays 4000*0.021 = 84, short ETH receives 3000*0.021 = 63 -> -21
    assert by_name["Funding Spike (7d @ 0.10%/8h)"].portfolio_pnl_usd == pytest.approx(-21.0)


def test_correlation_scenario_picks_the_adverse_direction_for_a_net_short_book() -> None:
    bundle = _beta_bundle({"BTCUSDT": 60_000.0, "SOLUSDT": 150.0, "ETHUSDT": 3_000.0}, {"BTCUSDT": 1.0, "SOLUSDT": 2.0, "ETHUSDT": 0.5})
    sol_px, eth_px = bundle.snapshots["SOLUSDT"].mark_price, bundle.snapshots["ETHUSDT"].mark_price
    positions = [
        PositionInput(symbol="SOLUSDT", side="short", entry_price=sol_px, notional=4_000.0, leverage=2.0, stop_price=sol_px * 1.05),
        PositionInput(symbol="ETHUSDT", side="long", entry_price=eth_px, notional=3_000.0, leverage=2.0, stop_price=eth_px * 0.95),
    ]
    summary = analyze_portfolio(positions, AccountState(equity=10_000.0), market=bundle)
    names = [s.scenario for s in summary.stress_scenarios]
    # Up 10%: -400 + 300 = -100 (adverse); down 10%: +100 -> squeeze is reported
    assert "Correlation -> 1 Squeeze" in names and "Correlation -> 1 Crash" not in names
    squeeze = next(s for s in summary.stress_scenarios if s.scenario == "Correlation -> 1 Squeeze")
    assert squeeze.portfolio_pnl_usd == pytest.approx(-100.0)
    assert len(summary.stress_scenarios) == 6


def test_stress_catches_account_level_cross_liquidation() -> None:
    bundle = _beta_bundle({"BTCUSDT": 60_000.0}, {"BTCUSDT": 1.0})
    btc_px = bundle.snapshots["BTCUSDT"].mark_price
    pos = PositionInput(symbol="BTCUSDT", side="long", entry_price=btc_px, quantity=0.1, leverage=10.0,
                        margin_mode="cross", stop_price=btc_px * 0.97)
    summary = analyze_portfolio([pos], AccountState(equity=1_000.0), market=bundle)
    by_name = {s.scenario: s for s in summary.stress_scenarios}
    # notional ~6,000: BTC -10% -> -600 -> cross pool 400 > MM (5,400*0.4% = 21.6) -> survives
    assert by_name["BTC -10%"].liquidated_symbols == []
    # BTC -20% -> -1,200 -> pool -200 <= MM -> whole cross account liquidated
    assert by_name["BTC -20%"].liquidated_symbols == ["BTCUSDT"]


def test_cross_hedge_is_netted_at_account_level() -> None:
    # ETH moves exactly like BTC (beta 1). Long BTC $6,000 cross + short ETH $6,000 cross on $1,000 equity:
    # every BTC-driven scenario nets to ~0, so the cross account is never liquidated, even though a
    # per-position LP for BTC alone (WB = 1000 - MM_eth 24 = 976 -> LP 15.9% below) would be crossed at -20%.
    bundle = _beta_bundle({"BTCUSDT": 60_000.0, "ETHUSDT": 3_000.0}, {"BTCUSDT": 1.0, "ETHUSDT": 1.0})
    btc_px, eth_px = bundle.snapshots["BTCUSDT"].mark_price, bundle.snapshots["ETHUSDT"].mark_price
    positions = [
        PositionInput(symbol="BTCUSDT", side="long", entry_price=btc_px, notional=6_000.0, leverage=10.0, margin_mode="cross", stop_price=btc_px * 0.9),
        PositionInput(symbol="ETHUSDT", side="short", entry_price=eth_px, notional=6_000.0, leverage=10.0, margin_mode="cross", stop_price=eth_px * 1.1),
    ]
    summary = analyze_portfolio(positions, AccountState(equity=1_000.0), market=bundle)
    for s in summary.stress_scenarios:
        if s.scenario.startswith("BTC"):
            assert s.portfolio_pnl_usd == pytest.approx(0.0, abs=1e-6)
            assert s.liquidated_symbols == []


def test_isolated_losses_do_not_drain_the_cross_pool() -> None:
    # Equity 2,000; isolated SOL long $4,000 at 5x (margin 800, beta 2); cross BTC long $3,000.
    # BTC -20%: SOL -40% -> isolated SOL liquidated (loses at most its 800 margin).
    # Cross pool = 2,000 - 800 = 1,200; cross PnL = -600 -> 600 > MM (2,400*0.4% = 9.6) -> BTC survives.
    bundle = _beta_bundle({"BTCUSDT": 60_000.0, "SOLUSDT": 150.0}, {"BTCUSDT": 1.0, "SOLUSDT": 2.0})
    btc_px, sol_px = bundle.snapshots["BTCUSDT"].mark_price, bundle.snapshots["SOLUSDT"].mark_price
    sol = PositionInput(symbol="SOLUSDT", side="long", entry_price=sol_px, notional=4_000.0, leverage=5.0, stop_price=sol_px * 0.95)
    btc_small = PositionInput(symbol="BTCUSDT", side="long", entry_price=btc_px, notional=3_000.0, leverage=5.0, margin_mode="cross", stop_price=btc_px * 0.95)
    s1 = {s.scenario: s for s in analyze_portfolio([sol, btc_small], AccountState(equity=2_000.0), market=bundle).stress_scenarios}
    assert s1["BTC -20%"].liquidated_symbols == ["SOLUSDT"]
    # With $7,000 cross BTC instead: cross PnL -1,400 -> pool -200 -> cross liquidated too
    btc_big = btc_small.model_copy(update={"quantity": 7_000.0 / btc_px})
    s2 = {s.scenario: s for s in analyze_portfolio([sol, btc_big], AccountState(equity=2_000.0), market=bundle).stress_scenarios}
    assert sorted(s2["BTC -20%"].liquidated_symbols) == ["BTCUSDT", "SOLUSDT"]


# --------------------------------------------------------------------------------------------
# 5. Rules: every OK / WARN / BREACH boundary (warn at 80% of limit)
# --------------------------------------------------------------------------------------------
def _risk_position(risk_usd: float, leverage: float = 1.0, sym: str = "SOLUSDT") -> PositionInput:
    # long Q=1 @ 1,000 with mark = entry (no market) -> risk to stop = 1000 - stop
    return PositionInput(symbol=sym, side="long", entry_price=1_000.0, quantity=1.0, leverage=leverage, stop_price=1_000.0 - risk_usd)


def _rule(rule_id: str, positions: list[PositionInput], rules: RuleConfig, journal=None, equity: float = 10_000.0):
    acct = AccountState(equity=equity)
    analyses = analyze_positions(positions, acct)
    port = analyze_portfolio(analyses, acct)
    evals, cb = evaluate_rules(analyses, port, acct, rules, journal=journal)
    return {e.rule_id: e for e in evals}[rule_id], cb


@pytest.mark.parametrize(
    "risk_usd,status",
    [(159.0, "OK"), (160.0, "WARN"), (200.0, "WARN"), (201.0, "BREACH")],  # 2% limit on 10k = $200; warn from $160
)
def test_max_risk_per_trade_boundaries(risk_usd, status) -> None:
    ev, _ = _rule("max_risk_per_trade_pct", [_risk_position(risk_usd)], RuleConfig())
    assert ev.status == status


@pytest.mark.parametrize(
    "risks,status",
    # limit 6% = $600, warn from $480 (sum of two positions' stop risk)
    [((200.0, 279.0), "OK"), ((200.0, 280.0), "WARN"), ((200.0, 200.0), "OK"), ((300.0, 300.0), "WARN"), ((300.0, 301.0), "BREACH")],
)
def test_max_total_open_risk_boundaries(risks, status) -> None:
    rules = RuleConfig(max_risk_per_trade_pct=5.0)
    ev, _ = _rule("max_total_open_risk_pct", [_risk_position(risks[0], sym="SOLUSDT"), _risk_position(risks[1], sym="ETHUSDT")], rules)
    assert ev.status == status


@pytest.mark.parametrize("lev,status", [(7.9, "OK"), (8.0, "WARN"), (10.0, "WARN"), (10.5, "BREACH")])
def test_max_leverage_boundaries(lev, status) -> None:
    ev, _ = _rule("max_leverage", [_risk_position(50.0, leverage=lev)], RuleConfig())
    assert ev.status == status


@pytest.mark.parametrize("n,status", [(3, "OK"), (4, "WARN"), (5, "WARN"), (6, "BREACH")])
def test_max_positions_boundaries(n, status) -> None:
    syms = ["SOLUSDT", "ETHUSDT", "XRPUSDT", "DOGEUSDT", "BNBUSDT", "ADAUSDT"][:n]
    ev, _ = _rule("max_positions", [_risk_position(10.0, sym=s) for s in syms], RuleConfig())
    assert ev.status == status


@pytest.mark.parametrize(
    "corr,notional_b,status",
    # Two longs, $8,000 + notional_b, limit 150% of $10k = $15,000 (warn from $12,000) when r >= 0.70
    [(0.69, 9_000.0, "OK"), (0.70, 3_999.0, "OK"), (0.70, 4_000.0, "WARN"), (0.70, 7_000.0, "WARN"), (0.70, 7_001.0, "BREACH")],
)
def test_correlated_exposure_boundaries(corr, notional_b, status) -> None:
    acct = AccountState(equity=10_000.0)
    pos = [
        PositionInput(symbol="SOLUSDT", side="long", entry_price=100.0, notional=8_000.0, leverage=5.0, stop_price=99.0),
        PositionInput(symbol="ETHUSDT", side="long", entry_price=100.0, notional=notional_b, leverage=5.0, stop_price=99.0),
    ]
    analyses = analyze_positions(pos, acct)
    port = analyze_portfolio(analyses, acct)
    corr_map = {"SOLUSDT": {"SOLUSDT": 1.0, "ETHUSDT": corr}, "ETHUSDT": {"SOLUSDT": corr, "ETHUSDT": 1.0}}
    port = port.model_copy(update={"corr_30d": corr_map})
    evals, _ = evaluate_rules(analyses, port, acct, RuleConfig(max_risk_per_trade_pct=50, max_total_open_risk_pct=50))
    assert {e.rule_id: e for e in evals}["max_correlated_exposure_pct"].status == status


def _journal_with_loss(loss: float, when: datetime, ref: datetime, start_equity: float = 10_000.0):
    trades = [_trade(when - timedelta(minutes=30), when, -loss, qty=1.0, entry=10_000.0, stop=None)]
    return analyze_journal(trades, starting_equity=start_equity, reference_time=ref)


@pytest.mark.parametrize(
    "loss,status,cb_status",
    # daily limit 3% of the DAY-START equity: current equity = 10,000 - loss, so the day started at $10k
    # -> limit $300 (inclusive), warn from $240
    [(239.0, "OK", "OPEN"), (240.0, "WARN", "CAUTION"), (299.0, "WARN", "CAUTION"), (300.0, "BREACH", "TRIPPED")],
)
def test_daily_loss_boundaries_match_circuit_breaker(loss, status, cb_status) -> None:
    ref = datetime(2025, 10, 8, 18, 0, tzinfo=timezone.utc)  # a Wednesday
    j = _journal_with_loss(loss, ref - timedelta(hours=2), ref)
    ev, cb = _rule("daily_loss_limit_pct", [], RuleConfig(weekly_loss_limit_pct=50, max_drawdown_pct=50), journal=j,
                   equity=10_000.0 - loss)
    assert ev.status == status
    assert cb.status == cb_status
    assert cb.trading_allowed is (cb_status != "TRIPPED")
    assert cb.reference_date == "2025-10-08"


@pytest.mark.parametrize("loss,status", [(559.0, "OK"), (560.0, "WARN"), (700.0, "BREACH")])
def test_weekly_loss_boundaries(loss, status) -> None:
    # loss closed on Monday, reference Wednesday same ISO week -> counts for the week, not for today
    ref = datetime(2025, 10, 8, 18, 0, tzinfo=timezone.utc)
    j = _journal_with_loss(loss, datetime(2025, 10, 6, 12, 0, tzinfo=timezone.utc), ref)
    assert j.today_pnl_usd == 0.0 and j.week_pnl_usd == -loss
    # week-start equity = current (10,000 - loss) + loss = 10,000 -> 7% limit = $700, warn from $560
    ev, _ = _rule("weekly_loss_limit_pct", [], RuleConfig(daily_loss_limit_pct=50, max_drawdown_pct=50), journal=j,
                  equity=10_000.0 - loss)
    assert ev.status == status


@pytest.mark.parametrize("loss,status", [(1_199.0, "OK"), (1_200.0, "WARN"), (1_500.0, "BREACH")])
def test_drawdown_boundaries(loss, status) -> None:
    # Peak = starting equity 10,000 ; drawdown % = loss / 10,000 ; limit 15% (warn from 12%)
    ref = datetime(2025, 10, 30, tzinfo=timezone.utc)
    j = _journal_with_loss(loss, datetime(2025, 10, 1, 12, 0, tzinfo=timezone.utc), ref)
    ev, _ = _rule("max_drawdown_pct", [], RuleConfig(daily_loss_limit_pct=50, weekly_loss_limit_pct=50), journal=j)
    assert ev.status == status


def test_mandatory_stop_and_liquidation_safety_statuses() -> None:
    m = {"SOLUSDT": _snap("SOLUSDT", 100.0, 5.0), "XRPUSDT": _snap("XRPUSDT", 2.0, 0.1)}
    acct = AccountState(equity=10_000.0)
    no_stop = PositionInput(symbol="XRPUSDT", side="long", entry_price=2.0, quantity=100.0, leverage=2.0)
    analyses = analyze_positions([no_stop], acct, market=m)
    port = analyze_portfolio(analyses, acct)
    on = {e.rule_id: e.status for e in evaluate_rules(analyses, port, acct, RuleConfig())[0]}
    off = {e.rule_id: e.status for e in evaluate_rules(analyses, port, acct, RuleConfig(mandatory_stop=False))[0]}
    assert on["mandatory_stop"] == "BREACH" and off["mandatory_stop"] == "WARN"
    # 25x long @100: LP = (4*... ) -> stop 93 is beyond LP (~96.5) -> BREACH
    bad = PositionInput(symbol="SOLUSDT", side="long", entry_price=100.0, quantity=10.0, leverage=25.0, stop_price=93.0)
    # 12x long: LP = (1 - 1/12)/0.995 * 100 = 92.13 -> 7.87 below mark = 1.57 ATR -> OK ; stop 95 inside
    ok = PositionInput(symbol="SOLUSDT", side="long", entry_price=100.0, quantity=10.0, leverage=12.0, stop_price=95.0)
    # 15x long: LP = (1 - 1/15)/0.995*100 = 93.80 -> 1.24 ATR from mark -> WARN (stop 95 still inside LP)
    tight = PositionInput(symbol="SOLUSDT", side="long", entry_price=100.0, quantity=10.0, leverage=15.0, stop_price=95.0)
    for pos, expected in ((bad, "BREACH"), (ok, "OK"), (tight, "WARN")):
        an = analyze_positions([pos], acct, market=m)
        ev = {e.rule_id: e.status for e in evaluate_rules(an, analyze_portfolio(an, acct), acct, RuleConfig(max_leverage=50))[0]}
        assert ev["liquidation_safety"] == expected


# --------------------------------------------------------------------------------------------
# 6. Sizing and pre-trade check
# --------------------------------------------------------------------------------------------
def test_each_sizing_method_hand_calculated() -> None:
    acct = AccountState(equity=20_000.0)
    one_tier = {"SOLUSDT": [MarginTier(max_notional=math.inf, mmr=0.005, cum_b=0.0)]}
    recs = {r.method: r for r in recommend_sizes(
        symbol="SOLUSDT", side="long", entry_price=50.0, stop_price=48.0, account=acct, atr_1d=2.5,
        risk_pct=1.0, win_rate=0.5, payoff_ratio=1.2, margin_tiers=one_tier,
    )}
    # Fixed fractional: 1% of 20,000 = $200 / stop distance $2 = 100 units ($5,000 notional)
    assert recs["Fixed Fractional"].quantity == pytest.approx(100.0)
    assert recs["Fixed Fractional"].notional_usd == pytest.approx(5_000.0)
    # ATR vol-target: $200 / ATR 2.5 = 80 units; its stop risk 80 * 2 = $160 = 0.8%
    assert recs["ATR Volatility-Targeted"].quantity == pytest.approx(80.0)
    assert recs["ATR Volatility-Targeted"].risk_pct == pytest.approx(0.8)
    # Kelly: f* = 0.5 - 0.5/1.2 = 0.083333 ; x0.25 = 2.0833% (< 2.5% cap) -> $416.67 / $2 = 208.33 units
    assert recs["Capped Fractional Kelly"].risk_pct == pytest.approx(25.0 / 12.0)
    assert recs["Capped Fractional Kelly"].quantity == pytest.approx(20_000 * (1 / 12 * 0.25) / 2.0)
    # Max-leverage-safe (isolated): target LP = 48 - 1.5*2.5 = 44.25 ; L = 1 / (1 - 0.885*0.995) = 8.2068
    lev = 1.0 / (1.0 - (44.25 / 50.0) * 0.995)
    assert recs["Max-Leverage-Safe"].max_safe_leverage == pytest.approx(round(lev, 2))
    # safe qty = min(FF qty 100, equity*L/entry = 20000*8.2068/50 = 3282.7) = 100
    assert recs["Max-Leverage-Safe"].quantity == pytest.approx(100.0)

    # Kelly cap and negative edge
    capped = {r.method: r for r in recommend_sizes(symbol="SOLUSDT", side="long", entry_price=50.0, stop_price=48.0,
                                                   account=acct, atr_1d=2.5, win_rate=0.6, payoff_ratio=2.0)}
    # f* = 0.6 - 0.4/2 = 0.4 -> 0.25x = 10% -> capped at 2.5% -> $500 / $2 = 250 units
    assert capped["Capped Fractional Kelly"].quantity == pytest.approx(250.0)
    neg = {r.method: r for r in recommend_sizes(symbol="SOLUSDT", side="long", entry_price=50.0, stop_price=48.0,
                                                account=acct, atr_1d=2.5, win_rate=0.4, payoff_ratio=1.0)}
    assert neg["Capped Fractional Kelly"].quantity == 0.0
    with pytest.raises(ValueError):
        recommend_sizes(symbol="SOLUSDT", side="long", entry_price=50.0, stop_price=51.0, account=acct, atr_1d=2.5)


def test_max_safe_leverage_formula_and_edge_cases() -> None:
    # Long 100 / stop 95 / ATR 4 / k 1.5 -> target 89 -> L = 1/(1 - 0.89*0.995) = 8.73744
    lev = max_safe_leverage_for_stop(100.0, 95.0, 4.0, "long", mmr=0.005, k_atr=1.5)
    assert lev == pytest.approx(1.0 / (1.0 - 0.89 * 0.995))
    # Round trip: isolated LP at that leverage sits exactly on the target
    q = 1.0
    assert compute_liquidation_price(100.0 / lev, q, 100.0, "long", 0.005) == pytest.approx(89.0)
    # Short 100 / stop 105 / ATR 4 -> target 111 -> L = 1/(1.11*1.005 - 1) = 8.65426
    lev_s = max_safe_leverage_for_stop(100.0, 105.0, 4.0, "short", mmr=0.005, k_atr=1.5)
    assert lev_s == pytest.approx(1.0 / (1.11 * 1.005 - 1.0))
    assert compute_liquidation_price(100.0 / lev_s, 1.0, 100.0, "short", 0.005) == pytest.approx(111.0)
    # Target LP at or below $0 (stop 5, 1.5 ATR of 4 below it = -1): only 1x avoids a positive LP
    assert max_safe_leverage_for_stop(10.0, 5.0, 4.0, "long", mmr=0.005, k_atr=1.5) == 1.0


def test_cross_margin_safe_size_cap() -> None:
    acct = AccountState(equity=10_000.0)
    one_tier = {"SOLUSDT": [MarginTier(max_notional=math.inf, mmr=0.005, cum_b=0.0)]}
    recs = {r.method: r for r in recommend_sizes(
        symbol="SOLUSDT", side="long", entry_price=100.0, stop_price=95.0, account=acct, atr_1d=4.0,
        risk_pct=50.0, margin_mode="cross", margin_tiers=one_tier,
    )}
    # FF qty = 5,000 / 5 = 1,000. Cross cap: Q <= WB / (EP - T*(1-MMR)) = 10000 / (100 - 89*0.995) = 873.744
    cap = 10_000.0 / (100.0 - 89.0 * 0.995)
    assert recs["Max-Leverage-Safe"].quantity == pytest.approx(cap)
    # At that size the cross LP is exactly the 89 target
    assert compute_liquidation_price(10_000.0, cap, 100.0, "long", 0.005) == pytest.approx(89.0)


def test_pre_trade_verdicts() -> None:
    acct = AccountState(equity=10_000.0)
    rules = RuleConfig(max_risk_per_trade_pct=2.0, max_total_open_risk_pct=5.0, max_leverage=10.0, max_positions=3)
    m = {"SOLUSDT": _snap("SOLUSDT", 100.0, 4.0), "ETHUSDT": _snap("ETHUSDT", 3_000.0, 100.0)}

    # GO: risk 10*(100-95) = $50 = 0.5%, 3x
    go = pre_trade_check(PositionInput(symbol="SOLUSDT", side="long", entry_price=100.0, quantity=10.0, leverage=3.0, stop_price=95.0), [], acct, m, rules)
    assert go.verdict == "GO" and go.after_open_risk_pct == pytest.approx(0.5)
    assert go.before_gross_exposure_usd == 0.0 and go.after_gross_exposure_usd == pytest.approx(1_000.0)

    # REDUCE SIZE: risk 60*(100-95) = $300 = 3% > 2% -> recommended qty = $200 / $5 = 40
    red = pre_trade_check(PositionInput(symbol="SOLUSDT", side="long", entry_price=100.0, quantity=60.0, leverage=3.0, stop_price=95.0), [], acct, m, rules)
    assert red.verdict == "REDUCE SIZE"
    assert red.recommended_quantity == pytest.approx(40.0)

    # NO-GO: stop beyond liquidation (25x long, LP ~96.5, stop 90)
    ng = pre_trade_check(PositionInput(symbol="SOLUSDT", side="long", entry_price=100.0, quantity=10.0, leverage=25.0, stop_price=90.0), [], acct, m, rules)
    assert ng.verdict == "NO-GO" and ng.proposed_analysis.stop_beyond_liquidation

    # NO-GO: stop on the wrong side of the entry
    wrong = pre_trade_check(PositionInput(symbol="SOLUSDT", side="long", entry_price=100.0, quantity=10.0, leverage=3.0, stop_price=101.0), [], acct, m, rules)
    assert wrong.verdict == "NO-GO" and any("wrong side" in r for r in wrong.reasons)

    # NO-GO: missing stop under the mandatory-stop rule; with the rule off it is flagged, not silently passed
    nostop = PositionInput(symbol="SOLUSDT", side="long", entry_price=100.0, quantity=10.0, leverage=3.0)
    assert pre_trade_check(nostop, [], acct, m, rules).verdict == "NO-GO"
    relaxed = pre_trade_check(nostop, [], acct, m, rules.model_copy(update={"mandatory_stop": False}))
    assert any("unbounded" in r for r in relaxed.reasons)

    # NO-GO: would exceed max positions (3 existing + 1)
    existing = [PositionInput(symbol=s, side="long", entry_price=100.0, quantity=1.0, leverage=2.0, stop_price=99.0)
                for s in ("XRPUSDT", "DOGEUSDT", "BNBUSDT")]
    assert pre_trade_check(PositionInput(symbol="SOLUSDT", side="long", entry_price=100.0, quantity=1.0, leverage=2.0, stop_price=99.0),
                           existing, acct, m, rules).verdict == "NO-GO"

    # NO-GO: circuit breaker tripped by today's -$400 (4% > 3%)
    ref = datetime(2025, 10, 8, 18, tzinfo=timezone.utc)
    j = _journal_with_loss(400.0, ref - timedelta(hours=1), ref)
    tripped = pre_trade_check(PositionInput(symbol="SOLUSDT", side="long", entry_price=100.0, quantity=10.0, leverage=3.0, stop_price=95.0), [], acct, m, rules, journal=j)
    assert tripped.verdict == "NO-GO" and any("Daily loss limit" in r for r in tripped.reasons)

    # REDUCE SIZE for leverage over the cap, with recommended leverage = cap
    hi = pre_trade_check(PositionInput(symbol="ETHUSDT", side="long", entry_price=3_000.0, quantity=0.1, leverage=12.0, stop_price=2_950.0), [], acct, m, rules)
    assert hi.verdict == "REDUCE SIZE" and hi.recommended_leverage == 10.0


def test_pre_trade_limit_entry_is_evaluated_at_its_entry_price() -> None:
    # Mark 100, proposed long LIMIT at 90 with stop 88, qty 50: risk if filled = 50 * (90 - 88) = $100 = 1% -> GO.
    # (Measured from the current mark it would wrongly be 50 * (100 - 88) = $600 = 6% -> REDUCE SIZE.)
    acct = AccountState(equity=10_000.0)
    m = {"SOLUSDT": _snap("SOLUSDT", 100.0, 4.0)}
    res = pre_trade_check(PositionInput(symbol="SOLUSDT", side="long", entry_price=90.0, quantity=50.0, leverage=2.0, stop_price=88.0),
                          [], acct, m, RuleConfig())
    assert res.proposed_analysis.mark_price == 90.0
    assert res.proposed_analysis.risk_to_stop_usd == pytest.approx(100.0)
    assert res.proposed_analysis.unrealized_pnl == 0.0
    assert res.verdict == "GO"


# --------------------------------------------------------------------------------------------
# 7. Journal analytics and behaviour detectors
# --------------------------------------------------------------------------------------------
def test_journal_core_metrics() -> None:
    d = T0
    trades = [
        _trade(d, d + timedelta(hours=1), 30.0, sym="BTCUSDT"),                         # +3R (risk 2*... see below)
        _trade(d + timedelta(hours=3), d + timedelta(hours=4), -10.0, sym="ETHUSDT"),
        _trade(d + timedelta(days=1), d + timedelta(days=1, hours=1), 20.0, sym="BTCUSDT"),
        _trade(d + timedelta(days=1, hours=3), d + timedelta(days=1, hours=4), -10.0, sym="ETHUSDT"),
    ]
    # Each trade: qty 1, entry 100, stop 98 -> risk $2 -> R = pnl/2 -> [15, -5, 10, -5]
    j = analyze_journal(trades, starting_equity=1_000.0)
    assert j.total_trades == 4 and j.win_rate == 0.5
    assert j.total_pnl_usd == pytest.approx(30.0)
    assert j.expectancy_usd == pytest.approx(7.5)            # 30 / 4
    assert j.profit_factor == pytest.approx(2.5)             # 50 / 20
    assert j.payoff_ratio == pytest.approx(2.5)              # avg win 25 / avg loss 10
    assert j.r_multiples == pytest.approx([15.0, -5.0, 10.0, -5.0])
    assert j.avg_r == pytest.approx(3.75)
    by_sym = {r["symbol"]: r for r in j.by_symbol}
    assert by_sym["BTCUSDT"]["trades"] == 2 and by_sym["BTCUSDT"]["total_pnl"] == pytest.approx(50.0)
    assert by_sym["ETHUSDT"]["win_rate"] == 0.0
    # 2025-10-01 is a Wednesday (closes on Wed and Thu)
    assert [(r["weekday"], r["trades"]) for r in j.by_weekday] == [("Wed", 2), ("Thu", 2)]
    assert {r["hour"]: r["trades"] for r in j.by_hour} == {1: 2, 4: 2}


def test_drawdown_depth_and_duration_including_recovery() -> None:
    d = T0
    trades = [
        _trade(d, d + timedelta(hours=1), 100.0, entry=1_000.0, stop=None),                      # 1,100 peak @ 01:00
        _trade(d + timedelta(hours=4), d + timedelta(hours=5), -220.0, entry=1_000.0, stop=None),  # 880: DD 220/1100 = 20%
        _trade(d + timedelta(hours=8), d + timedelta(hours=9), 110.0, entry=1_000.0, stop=None),   # 990 (still in DD)
        _trade(d + timedelta(hours=12), d + timedelta(hours=13), 200.0, entry=1_000.0, stop=None), # 1,190 recovered @ 13:00 -> 12h DD
        _trade(d + timedelta(hours=14), d + timedelta(hours=15), -119.0, entry=1_000.0, stop=None),# 1,071: DD 119/1190 = 10%
    ]
    j = analyze_journal(trades, starting_equity=1_000.0, reference_time=d + timedelta(hours=21))
    assert j.max_drawdown_pct == pytest.approx(20.0)
    assert j.max_drawdown_usd == pytest.approx(220.0)
    assert j.current_drawdown_pct == pytest.approx(10.0)
    assert j.current_drawdown_duration_hours == pytest.approx(8.0)   # 13:00 -> 21:00 (ref)
    assert j.max_drawdown_duration_hours == pytest.approx(12.0)      # peak 01:00 -> recovery 13:00


def _ep(j, kind: str) -> int:
    return j.episode_counts.get(kind, 0)


def test_revenge_trading_detector() -> None:
    d = T0 + timedelta(hours=10)
    loss = _trade(d, d + timedelta(minutes=30), -2.0, qty=1.0)                       # closes 10:30
    pos = analyze_journal([loss, _trade(d + timedelta(minutes=40), d + timedelta(minutes=70), 1.0, qty=1.5)])
    assert _ep(pos, "revenge_trading") == 1                                            # 10 min later, 1.5x size
    assert pos.episodes[0].severity == "HIGH"                                          # gap <= 10 min
    late = analyze_journal([loss, _trade(d + timedelta(minutes=65), d + timedelta(minutes=90), 1.0, qty=1.5)])
    same = analyze_journal([loss, _trade(d + timedelta(minutes=40), d + timedelta(minutes=70), 1.0, qty=1.1)])
    assert _ep(late, "revenge_trading") == 0 and _ep(same, "revenge_trading") == 0     # 35 min later / 1.1x


def test_overtrading_detector() -> None:
    def day_trades(day: int, n: int) -> list[JournalTrade]:
        base = T0 + timedelta(days=day, hours=1)
        return [_trade(base + timedelta(hours=2 * k), base + timedelta(hours=2 * k, minutes=20), 1.0) for k in range(n)]

    baseline = [t for dd in range(6) for t in day_trades(dd, 2)]
    # baseline median 2/day -> threshold max(6, 2 x 2) = 6 trades
    assert _ep(analyze_journal(baseline + day_trades(6, 6)), "overtrading") == 1
    assert _ep(analyze_journal(baseline + day_trades(6, 5)), "overtrading") == 0


def test_size_escalation_detector_without_lookahead() -> None:
    d = T0 + timedelta(hours=10)
    l1 = _trade(d, d + timedelta(minutes=30), -2.0)
    l2 = _trade(d + timedelta(hours=1), d + timedelta(hours=1, minutes=30), -2.0)
    big = _trade(d + timedelta(hours=2, minutes=31), d + timedelta(hours=3), 1.0, qty=1.5)  # 1.5x streak median
    assert _ep(analyze_journal([l1, l2, big]), "size_escalation") == 1
    # A win between the losses and the big trade resets the streak
    win = _trade(d + timedelta(hours=1, minutes=40), d + timedelta(hours=2), 2.0)
    assert _ep(analyze_journal([l1, l2, win, big]), "size_escalation") == 0
    # Losses that only CLOSE after the big trade is opened were not known at entry -> no escalation
    o1 = _trade(d, d + timedelta(hours=5), -2.0)
    o2 = _trade(d + timedelta(minutes=5), d + timedelta(hours=5, minutes=5), -2.0)
    early_big = _trade(d + timedelta(hours=2), d + timedelta(hours=6), 1.0, qty=1.5)
    assert _ep(analyze_journal([o1, o2, early_big]), "size_escalation") == 0


def test_disposition_effect_detector() -> None:
    d = T0
    def t(i, hold, pnl):
        return _trade(d + timedelta(hours=4 * i), d + timedelta(hours=4 * i, minutes=hold), pnl)
    pos = analyze_journal([t(0, 30, 2.0), t(1, 30, 2.0), t(2, 120, -1.0), t(3, 120, -1.0)])
    assert _ep(pos, "disposition_effect") == 1 and pos.disposition_ratio == pytest.approx(4.0)  # 120 / 30
    neg = analyze_journal([t(0, 60, 2.0), t(1, 60, 2.0), t(2, 40, -1.0), t(3, 40, -1.0)])
    assert _ep(neg, "disposition_effect") == 0


def test_stop_violation_detector() -> None:
    d = T0
    # long: entry 100, stop 98 (risk $2), exit 96 -> loss $4 = 2R > 1.15R -> HIGH violation
    v = analyze_journal([_trade(d, d + timedelta(hours=1), -4.0)])
    assert _ep(v, "stop_violation") == 1 and v.episodes[0].severity == "HIGH"
    # short: entry 100, stop 102, exit 104 -> 2R
    vs = analyze_journal([_trade(d, d + timedelta(hours=1), -4.0, side="short", stop=102.0)])
    assert _ep(vs, "stop_violation") == 1
    # slippage within tolerance: exit 97.8 -> 1.1R <= 1.15R -> not a violation
    ok = analyze_journal([_trade(d, d + timedelta(hours=1), -2.2)])
    assert _ep(ok, "stop_violation") == 0


def test_tilt_session_detector_and_discipline_score() -> None:
    d = T0 + timedelta(hours=9)
    # 3 equal-size losses of $15 inside 4h on $1,000 equity = 4.5% >= 3% -> tilt (no other flags needed)
    big_losses = [_trade(d + timedelta(hours=k), d + timedelta(hours=k, minutes=20), -15.0, stop=80.0) for k in range(3)]
    tilt = analyze_journal(big_losses, starting_equity=1_000.0)
    assert _ep(tilt, "tilt_session") == 1
    small = [_trade(d + timedelta(hours=k), d + timedelta(hours=k, minutes=20), -5.0, stop=80.0) for k in range(3)]
    assert _ep(analyze_journal(small, starting_equity=1_000.0), "tilt_session") == 0   # 1.5% and no flags

    # Clean journal: all stops respected, equal sizes, winners held longer -> perfect 100
    clean = []
    for i in range(10):
        win = i % 2 == 0
        clean.append(_trade(T0 + timedelta(hours=6 * i), T0 + timedelta(hours=6 * i, minutes=90 if win else 30), 3.0 if win else -2.0))
    cj = analyze_journal(clean, starting_equity=1_000.0)
    assert cj.episodes == [] and cj.discipline_score == 100.0 and cj.discipline_grade == "A"


def test_journal_reference_time_today_vs_synthetic() -> None:
    real = [_trade(T0, T0 + timedelta(hours=1), -500.0, entry=1_000.0, source=None)]
    now = datetime(2026, 11, 25, 12, tzinfo=timezone.utc)
    assert journal_reference_time(real, now=now) == now
    # A losing day weeks ago must not halt trading today
    j = analyze_journal(real, starting_equity=10_000.0, reference_time=journal_reference_time(real, now=now))
    assert j.today_pnl_usd == 0.0
    assert evaluate_circuit_breaker(AccountState(equity=10_000.0), RuleConfig(), j).trading_allowed is True
    synth = [_trade(T0, T0 + timedelta(hours=1), -500.0, entry=1_000.0, source="SYNTHETIC")]
    assert journal_reference_time(synth, now=now) is None  # evaluated as of its last trade


# --------------------------------------------------------------------------------------------
# 8. Drawdown schedule and risk of ruin
# --------------------------------------------------------------------------------------------
def test_drawdown_schedule_step_down_and_cppi() -> None:
    # Peak 10,000, equity 9,500 -> DD 5% (3-6% tier: 0.75x). CPPI floor 8,500, cushion 1,000 of 1,500
    # -> ratio 0.6667 x 1.25 = 0.8333 -> recommended = min(0.75, 0.8333) = 0.75%
    p = compute_drawdown_plan(current_equity=9_500.0, peak_equity=10_000.0, base_risk_pct=1.0, max_dd_limit_pct=15.0)
    assert p.current_dd_pct == pytest.approx(5.0)
    assert p.step_down_multiplier == 0.75
    assert p.cppi_cushion_usd == pytest.approx(1_000.0)
    assert p.cppi_multiplier == pytest.approx(1.25 * 1000 / 1500)
    assert p.recommended_risk_pct == pytest.approx(0.75)
    # 12% DD: quarter risk (10-15%) ; CPPI cushion 300/1500*1.25 = 0.25 -> 0.25%
    p12 = compute_drawdown_plan(current_equity=8_800.0, peak_equity=10_000.0, base_risk_pct=1.0)
    assert p12.step_down_multiplier == 0.25 and p12.recommended_risk_pct == pytest.approx(0.25)
    assert compute_drawdown_plan(current_equity=8_500.0, peak_equity=10_000.0).trading_halted is True


def test_risk_of_ruin_analytical_hand_values_and_mc_sanity() -> None:
    # b = 2: p u^2 + (1-p)/u = 1 -> (u-1)(p u^2 + p u - (1-p)) = 0 -> u = (-p + sqrt(p^2 + 4p(1-p))) / (2p)
    # p = 0.5: u = (-0.5 + sqrt(0.25 + 1.0)) / 1 = (sqrt(5) - 1)/2 = 0.618034
    # risk 5%, ruin 25%: N = ln(0.75)/ln(0.95) = 5.60857 -> u^N = exp(5.60857 * -0.481212) = 0.067279
    u = (math.sqrt(5.0) - 1.0) / 2.0
    n_units = math.log(0.75) / math.log(0.95)
    assert analytical_risk_of_ruin(0.5, 2.0, 5.0, 25.0) == pytest.approx(u**n_units, rel=1e-6)
    assert u**n_units == pytest.approx(0.067279, abs=1e-6)
    # b = 1: u = (1-p)/p = 2/3 ; risk 2%, ruin 20%: N = ln(0.8)/ln(0.98) = 11.0453 -> 0.011351
    assert analytical_risk_of_ruin(0.6, 1.0, 2.0, 20.0) == pytest.approx((2 / 3) ** (math.log(0.8) / math.log(0.98)), rel=1e-9)
    # No edge -> certain ruin
    assert analytical_risk_of_ruin(0.4, 1.0, 1.0, 25.0) == 1.0
    assert analytical_risk_of_ruin(0.5, 1.0, 1.0, 25.0) == 1.0
    # Vanishing edge does not crash the root finder
    assert 0.9 < analytical_risk_of_ruin(0.5, 1.0 + 1e-12, 1.0, 25.0) <= 1.0

    good = estimate_risk_of_ruin(win_rate=0.55, payoff_ratio=1.5, risk_per_trade_pct=1.0, simulations=2_000, horizon_trades=250)
    assert good.analytical_ruin_prob < 0.01 and good.mc_ruin_prob_flat < 0.02
    bad = estimate_risk_of_ruin(win_rate=0.35, payoff_ratio=1.0, risk_per_trade_pct=2.0, simulations=2_000, horizon_trades=1_000)
    assert bad.analytical_ruin_prob == 1.0 and bad.mc_ruin_prob_flat > 0.99
    # The step-down schedule halts at a 15% drawdown, so it can never reach a 25% loss from start
    assert bad.mc_ruin_prob_scheduled <= bad.mc_ruin_prob_flat
    # Bootstrapped R-multiples: empirical expectancy reported
    boot = estimate_risk_of_ruin(win_rate=0.5, payoff_ratio=1.5, r_multiples=[2.0, -1.0] * 10, simulations=200, horizon_trades=50)
    assert boot.expectancy_r == pytest.approx(0.5)
    # Non-finite R values are dropped before the 10-sample check (no crash on an all-NaN pool)
    nan_pool = estimate_risk_of_ruin(win_rate=0.5, payoff_ratio=1.5, r_multiples=[float("nan")] * 12, simulations=100, horizon_trades=20)
    assert nan_pool.expectancy_r == pytest.approx(0.25)


# --------------------------------------------------------------------------------------------
# 9. Market data providers: fallback chain with mocked failures (no network)
# --------------------------------------------------------------------------------------------
class _Resp:
    def __init__(self, status: int, payload: Any = None, text: str = "") -> None:
        self.status_code, self._payload, self.headers, self.text = status, payload, {}, text

    def json(self) -> Any:
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


class _Session:
    def __init__(self, handler) -> None:
        self.handler, self.calls = handler, []

    def request(self, method, url, params=None, json=None, timeout=None):
        self.calls.append(url)
        return self.handler(method, url, json)


def _provider(handler) -> MarketDataProvider:
    return MarketDataProvider(sample_dir=SAMPLE_DIR, session=_Session(handler), cache=TTLCache(default_ttl_seconds=0.0), sleep_fn=lambda s: None)


def test_provider_falls_back_binance_451_then_hyperliquid_500_to_sample() -> None:
    def handler(method, url, body):
        return _Resp(451, text="restricted") if "binance" in url else _Resp(500, text="boom")
    b = _provider(handler).get_market_data(["SOLUSDT"], mode="auto", lookback_days=5)
    assert b.source == "sample" and b.is_live is False
    assert "451" in b.reason and "500" in b.reason
    # Sample mark = last bundled close
    assert b.snapshots["SOLUSDT"].mark_price == pytest.approx(187.13)


def test_provider_survives_malformed_payloads() -> None:
    hl_meta = json.loads((ROOT / "tests" / "fixtures" / "hyperliquid_meta.json").read_text(encoding="utf-8"))
    hl_candles = json.loads((ROOT / "tests" / "fixtures" / "hyperliquid_candles.json").read_text(encoding="utf-8"))

    def handler(method, url, body):
        if "binance" in url:
            return _Resp(200, [{"symbol": "BTCUSDT", "lastFundingRate": "0.0001"}])  # no markPrice -> malformed
        if body and body.get("type") == "metaAndAssetCtxs":
            return _Resp(200, hl_meta)
        return _Resp(200, hl_candles)

    b = _provider(handler).get_market_data(["BTCUSDT"], mode="auto", lookback_days=2)
    assert b.source == "hyperliquid" and "Malformed Binance response" in b.reason

    def handler_bad_hl(method, url, body):
        if "binance" in url:
            return _Resp(200, [{"symbol": "BTCUSDT"}])
        return _Resp(200, [{"universe": [{"name": "BTC"}]}, [{"markPx": None, "midPx": None, "oraclePx": None}]])

    b2 = _provider(handler_bad_hl).get_market_data(["BTCUSDT"], mode="auto", lookback_days=2)
    assert b2.source == "sample" and "Malformed Hyperliquid response" in b2.reason


def test_demo_mode_never_touches_the_network_and_live_bundle_is_not_mixed_with_stale_sample() -> None:
    def forbid(method, url, body):
        raise AssertionError("network call in demo mode")
    p = _provider(forbid)
    assert p.get_market_data(["BTCUSDT", "ETHUSDT"], mode="demo").source == "sample"
    assert p.session.calls == []

    prem = json.loads((ROOT / "tests" / "fixtures" / "binance_premium_index.json").read_text(encoding="utf-8"))
    tick = json.loads((ROOT / "tests" / "fixtures" / "binance_ticker_24hr.json").read_text(encoding="utf-8"))
    kl = json.loads((ROOT / "tests" / "fixtures" / "binance_klines.json").read_text(encoding="utf-8"))

    def handler(method, url, body):
        if "premiumIndex" in url:
            return _Resp(200, [x for x in prem if x["symbol"] != "ETHUSDT"])  # ETH missing on the live feed
        if "ticker" in url:
            return _Resp(200, tick)
        return _Resp(200, kl)

    b = _provider(handler).get_market_data(["BTCUSDT", "ETHUSDT"], mode="auto", lookback_days=2)
    assert b.source == "binance" and b.missing_symbols == ["ETHUSDT"]
    assert "ETHUSDT" not in b.snapshots and "not on live feed: ETHUSDT" in b.reason


# --------------------------------------------------------------------------------------------
# 10. CLI
# --------------------------------------------------------------------------------------------
def test_cli_report_is_ascii_and_shows_cross_meaning() -> None:
    report, code = format_cli_report(account=AccountState(equity=10_000.0), positions_path=ROOT / "examples" / "positions_demo.csv", mode="demo")
    assert code == 0
    report.encode("ascii")  # no typographic characters left
    assert "none at current equity (cross)" in report
    assert "N/A" not in report
    # DOGE: Q 26,810 @ 0.1865, 25x -> WB 200.0065; LP = (200.0065 - 5000.1650)/(26810*0.005 - 26810) = 0.17994
    assert "LP=$0.1799" in report
    # Funding spike: gross 19,575.63 * 0.003 * 7 = 411.09
    assert "PnL=$  -411.09" in report


def test_cli_subprocess_on_cp1252_console() -> None:
    env = {**__import__("os").environ, "PYTHONIOENCODING": "cp1252", "RG_DATA_MODE": "demo"}
    out = subprocess.run(
        [sys.executable, "-m", "riskguard", "check", "--positions", "examples/positions_demo.csv", "--equity", "10000", "--demo"],
        cwd=ROOT, capture_output=True, env=env, timeout=60,
    )
    assert out.returncode == 0, out.stderr.decode("utf-8", "replace")
    out.stdout.decode("ascii")  # pure ASCII bytes
    assert b"RISKGUARD - PORTFOLIO RISK" in out.stdout


# --------------------------------------------------------------------------------------------
# 11. AI coach: tools, offline report, SDK contract, error paths (no API key, no network)
# --------------------------------------------------------------------------------------------
def _ctx() -> CoachContext:
    acct = AccountState(equity=10_000.0)
    rules = RuleConfig()
    m = {
        "DOGEUSDT": _snap("DOGEUSDT", 0.20, 0.012),
        "SOLUSDT": _snap("SOLUSDT", 93.0, 6.0),
        "ETHUSDT": _snap("ETHUSDT", 3_000.0, 120.0),
    }
    positions = [
        PositionInput(symbol="DOGEUSDT", side="long", entry_price=0.20, quantity=25_000, leverage=25, stop_price=0.18),  # stop beyond LP
        PositionInput(symbol="SOLUSDT", side="long", entry_price=100.0, quantity=20, leverage=5, stop_price=95.0),       # stop crossed
        PositionInput(symbol="ETHUSDT", side="long", entry_price=3_000.0, quantity=1.0, leverage=5, margin_mode="cross", stop_price=2_900.0),
    ]
    analyses = analyze_positions(positions, acct, market=m)
    port = analyze_portfolio(analyses, acct)
    ref = datetime(2025, 10, 8, 18, tzinfo=timezone.utc)
    journal = analyze_journal([_trade(ref - timedelta(hours=k + 1), ref - timedelta(hours=k), -20.0) for k in range(12)],
                              starting_equity=10_000.0, reference_time=ref)
    evals, cb = evaluate_rules(analyses, port, acct, rules, journal=journal)
    plan = compute_drawdown_plan(current_equity=10_000.0, current_dd_pct=journal.current_drawdown_pct)
    ror = estimate_risk_of_ruin(win_rate=0.5, payoff_ratio=1.5, simulations=300, horizon_trades=50)
    return CoachContext(account=acct, positions=positions, analyses=analyses, portfolio=port, rules=rules,
                        rule_evals=evals, circuit_breaker=cb, journal=journal, market=None, drawdown_plan=plan, risk_of_ruin=ror)


def test_every_tool_returns_compact_valid_json_with_correct_numbers() -> None:
    ctx = _ctx()
    tools = {t.name: t for t in build_coach_tools(ctx)}
    outs = {
        "get_portfolio_risk": tools["get_portfolio_risk"].call({}),
        "get_position_details": tools["get_position_details"].call({"symbol": "all"}),
        "one": tools["get_position_details"].call({"symbol": "doge"}),
        "missing": tools["get_position_details"].call({"symbol": "PEPE"}),
        "run_stress": tools["run_stress"].call({"scenario": "BTC -10%"}),
        "check_rules": tools["check_rules"].call({}),
        "evaluate_new_trade": tools["evaluate_new_trade"].call({"trade_json": json.dumps({"symbol": "XRPUSDT", "side": "long", "entry_price": 2.0, "quantity": 100, "leverage": 2, "stop_price": 1.9})}),
        "bad_trade": tools["evaluate_new_trade"].call({"trade_json": "{not json"}),
        "journal_behaviour": tools["journal_behaviour"].call({}),
        "size_trade": tools["size_trade"].call({"symbol": "SOLUSDT", "side": "long", "entry": 100.0, "stop": 95.0, "method": "fixed"}),
        "drawdown_plan": tools["drawdown_plan"].call({}),
    }
    parsed = {}
    for name, out in outs.items():
        assert isinstance(out, str) and len(out) < 3000, name
        parsed[name] = json.loads(out)
    p = ctx.portfolio
    assert parsed["get_portfolio_risk"]["gross_usd"] == round(p.gross_exposure_usd, 2)
    assert any(d.startswith("DOGEUSDT:STOP_BEYOND_LIQ") for d in parsed["get_portfolio_risk"]["dangers"])
    assert any(d.startswith("SOLUSDT:STOP_ALREADY_CROSSED") for d in parsed["get_portfolio_risk"]["dangers"])
    rows = {r["sym"]: r for r in parsed["get_position_details"]["positions"]}
    assert rows["ETHUSDT"]["lp"] == "none at current equity (cross)"
    # DOGE 25x isolated: WB = 5000/25 = 200 -> LP = (200 - 5000)/(25000*0.005 - 25000) = 0.192965
    assert parsed["one"]["liquidation_price"] == pytest.approx(4800 / 24_875, abs=1e-6)
    assert "error" in parsed["missing"] and "error" in parsed["bad_trade"]
    assert [s["name"] for s in parsed["run_stress"]["scenarios"]] == ["BTC -10%"]
    assert parsed["check_rules"]["trading_allowed_today"] == ctx.circuit_breaker.trading_allowed
    direct = pre_trade_check(PositionInput(symbol="XRPUSDT", side="long", entry_price=2.0, quantity=100, leverage=2, stop_price=1.9),
                             ctx.positions, ctx.account, ctx.market, ctx.rules, journal=ctx.journal)
    assert parsed["evaluate_new_trade"]["verdict"] == direct.verdict
    assert parsed["journal_behaviour"]["total_trades"] == 12
    # size_trade fixed fractional: 1% of 10,000 = $100 / $5 = 20 units; ATR proxy (no market) = 3% of entry
    rec = parsed["size_trade"]["recommendations"]
    assert len(rec) == 1 and rec[0]["qty"] == pytest.approx(20.0)
    assert parsed["size_trade"]["atr_source"].startswith("proxy")
    assert parsed["drawdown_plan"]["recommended_risk_pct"] == round(ctx.drawdown_plan.recommended_risk_pct, 3)


def test_bounded_json_is_always_valid_and_short() -> None:
    huge = {"positions": [{"sym": f"S{i}", "warnings": ["x" * 400] * 5} for i in range(500)], "equity": 1.0}
    out = _bounded_json(huge)
    assert len(out) <= 2950
    data = json.loads(out)
    assert data["truncated"] is True and data["equity"] == 1.0


def test_offline_report_uses_real_numbers() -> None:
    ctx = _ctx()
    rep = build_offline_coach_report(ctx, user_question="What now?")
    p = ctx.portfolio
    assert f"${p.equity:,.2f}" in rep
    assert f"${p.var_95_hist_usd:,.0f}" in rep
    doge = next(a for a in ctx.analyses if a.symbol == "DOGEUSDT")
    assert f"${doge.liquidation_price:,.4f}" in rep and "beyond your liquidation price" in rep
    assert "already crossed your stop" in rep and "none at current equity (cross)" not in rep.split("Prioritized")[0]
    assert f"{ctx.journal.discipline_score:.1f}/100" in rep
    assert "Not financial advice" in rep
    assert "N/A" not in rep


def test_tool_runner_call_contract(monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    ctx = _ctx()
    captured: dict[str, Any] = {}

    class Block(SimpleNamespace):
        pass

    final = SimpleNamespace(stop_reason="end_turn", model=DEFAULT_MODEL,
                            content=[Block(type="text", text="Cut DOGEUSDT leverage.")])

    class Client:
        def __init__(self, api_key=None):
            captured["api_key"] = api_key
            self.beta = SimpleNamespace(messages=SimpleNamespace(tool_runner=self.runner))

        def runner(self, **kwargs):
            captured.update(kwargs)
            return iter([SimpleNamespace(stop_reason="tool_use", content=[Block(type="tool_use", name="check_rules", input={}, id="t1")]), final])

    history: list[dict] = [{"role": "user", "content": "earlier"}, {"role": "assistant", "content": [{"type": "text", "text": "ok"}]}]
    res = run_coach_turn(ctx, history, "Top danger?", api_key="sk-ant-test", client_factory=Client)
    assert captured["api_key"] == "sk-ant-test"
    assert set(captured) - {"api_key"} == {"model", "max_tokens", "max_iterations", "system", "output_config", "betas", "fallbacks", "tools", "messages"}
    assert captured["model"] == "claude-opus-5-5" and captured["max_tokens"] == 16000
    assert captured["max_iterations"] == 16 and captured["system"] == COACH_PROMPT
    assert captured["output_config"] == {"effort": "high"}
    assert captured["betas"] == ["server-side-fallback-2026-07-01"] and captured["fallbacks"] == "default"
    assert [t.name for t in captured["tools"]] == [t.name for t in build_coach_tools(ctx)]
    assert captured["messages"][-1] == {"role": "user", "content": "Top danger?"} and len(captured["messages"]) == 3
    assert res.used_fallback_offline is False and res.tool_calls[0]["tool"] == "check_rules"
    assert res.reply_text.endswith("*Not financial advice.*")
    assert history[-1]["role"] == "assistant" and len(history) == 4


@pytest.mark.parametrize("kind", ["auth", "timeout", "bad_request", "empty", "refusal"])
def test_coach_error_paths_fall_back_offline(kind) -> None:
    ctx = _ctx()
    req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")

    class Client:
        def __init__(self, api_key=None):
            self.beta = SimpleNamespace(messages=SimpleNamespace(tool_runner=self.runner))

        def runner(self, **kwargs):
            if kind == "auth":
                raise anthropic.AuthenticationError("bad key", response=httpx.Response(401, request=req), body=None)
            if kind == "timeout":
                raise anthropic.APITimeoutError(request=req)
            if kind == "bad_request":
                raise anthropic.BadRequestError("bad", response=httpx.Response(400, request=req), body=None)
            if kind == "empty":
                return iter([])
            return iter([SimpleNamespace(stop_reason="refusal", content=[])])

    history: list[dict] = []
    res = run_coach_turn(ctx, history, "Review", api_key="sk-ant-x", client_factory=Client)
    assert res.used_fallback_offline is True and res.notice
    assert "DOGEUSDT" in res.reply_text and "Not financial advice" in res.reply_text
    assert [h["role"] for h in history] == ["user", "assistant"]


def test_no_key_uses_offline_coach(monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    def boom(**kwargs):
        raise AssertionError("client must not be created without a key")

    res = run_coach_turn(_ctx(), [], "Audit", api_key=None, client_factory=boom)
    assert res.used_fallback_offline is True and res.stop_reason == "offline_no_key"


# --------------------------------------------------------------------------------------------
# 12. Web app: every page, every main interaction (offline demo, network blocked)
# --------------------------------------------------------------------------------------------
@pytest.fixture()
def offline_app(monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("RG_DATA_MODE", "demo")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    def forbid(*a, **k):
        raise AssertionError("network call from the web app during tests")

    monkeypatch.setattr(requests.Session, "request", forbid)
    return AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=60)


def _df_with(at, column: str) -> pd.DataFrame:
    return next(d.value for d in at.dataframe if column in d.value.columns)


def test_web_app_all_pages_and_interactions(offline_app) -> None:
    at = offline_app.run()
    assert not at.exception
    # Dashboard: real numbers from the demo positions + sample data
    labels = {m.label: m.value for m in at.metric}
    assert labels["Account Equity"] == "$10,000.00"
    assert labels["Gross Exposure"] == "$19,576"  # 4996.04 + 4116.86 + 4231.06 + 3511.34 + 2720.33
    assert "64/100 (D)" in labels["Discipline Score"]
    snap = _df_with(at, "Liq Price")
    assert snap.set_index("Symbol").loc["ETHUSDT", "Liq Price"] == "none at current equity (cross)"
    assert snap.set_index("Symbol").loc["DOGEUSDT", "Liq Price"] == "$0.1799"
    assert any("beyond liquidation" in e.value for e in at.error)
    assert any("evaluated for 2025-10-31" in c.value for c in at.caption)

    # Positions: edit DOGE to 3x -> stop no longer beyond LP; move SOL stop above mark -> crossed-stop DANGER;
    # an invalid row (leverage 0.5) is excluded with a visible error, never silently.
    at.switch_page("pages/2_Positions.py").run()
    assert not at.exception and len(at.data_editor) == 1
    at.session_state["positions_editor"] = {
        "edited_rows": {0: {"leverage": 3.0}, 1: {"stop_price": 200.0}, 3: {"leverage": 0.5}},
        "added_rows": [],
        "deleted_rows": [],
    }
    at.run()
    assert not at.exception
    errors = [e.value for e in at.error]
    assert not any("DOGEUSDT stop" in e and "beyond liquidation" in e for e in errors)
    assert any("SOLUSDT stop already crossed" in e for e in errors)
    assert any("Excluded from analysis" in e and "XRPUSDT" in e for e in errors)
    table = _df_with(at, "Risk to Stop").set_index("Symbol")
    assert "XRPUSDT" not in table.index
    assert table.loc["SOLUSDT", "Risk to Stop"] == "UNBOUNDED (stop crossed)"
    # DOGE at 3x: LP = (1666.72 - 5000.17)/(26810*0.005 - 26810) = 0.12497
    assert table.loc["DOGEUSDT", "Liq Price"] == "$0.1250"

    # Exposure & Stress
    at.switch_page("pages/3_Exposure_and_Stress.py").run()
    assert not at.exception
    stress = _df_with(at, "Scenario")
    assert len(stress) == 6 and stress["Scenario"].iloc[0] == "BTC -5%"
    at.radio(key="corr_window_radio").set_value("90d (2160h)").run()
    assert not at.exception

    # Pre-trade: switching symbol re-seeds the entry with that symbol's mark; submit evaluates ETH
    at.switch_page("pages/4_Pre_Trade_and_Sizing.py").run()
    assert not at.exception
    at.selectbox(key="pt_symbol").set_value("ETHUSDT").run()
    assert at.number_input(key="pt_entry_ETHUSDT").value == pytest.approx(3846.42)
    assert at.number_input(key="pt_stop_ETHUSDT_long").value == pytest.approx(3846.42 * 0.97)
    at.number_input(key="pt_notional").set_value(1500.0)
    at.number_input(key="pt_leverage").set_value(3.0)
    next(b for b in at.button if b.label == "Evaluate Pre-Trade Risk").click().run()
    assert not at.exception
    assert any("Verdict:" in m.value for m in at.markdown)
    assert any("ETHUSDT @ $3,846.42" in c.value for c in at.caption)
    ba = _df_with(at, "Before").set_index("Metric")
    assert ba.loc["Gross Exposure ($)", "Delta"] == pytest.approx(1500.0)

    # Journal & Discipline
    at.switch_page("pages/5_Journal_and_Discipline.py").run()
    assert not at.exception
    assert any("SYNTHETIC DEMO JOURNAL" in i.value and "400 simulated trades" in i.value for i in at.info)
    assert {m.label for m in at.metric} >= {"Total Trades", "Profit Factor", "Max Drawdown"}
    at.multiselect(key="ep_filter").set_value(["tilt_session"]).run()
    assert not at.exception
    episodes = _df_with(at, "Severity")
    assert set(episodes["Type"]) == {"tilt_session"}

    # Rules: raise max leverage to 30 -> DOGE (3x after edit) and others OK; YAML export reflects it
    at.switch_page("pages/6_Rules.py").run()
    assert not at.exception
    next(n for n in at.number_input if n.label == "Max Leverage (x)").set_value(30.0)
    next(b for b in at.button if b.label == "Apply Rules").click().run()
    assert not at.exception
    assert at.session_state["rules"].max_leverage == 30.0
    assert any("max_leverage: 30.0" in c.value for c in at.code)
    audit = _df_with(at, "Explanation").set_index("Rule")
    assert audit.loc["Max Position Leverage", "Status"] == "OK"  # highest is SOL 8x < 24x warn level

    # AI Coach: offline one-click review renders a full report with real numbers
    at.switch_page("pages/7_AI_Coach.py").run()
    assert not at.exception
    next(b for b in at.button if "One-Click" in b.label).click().run()
    assert not at.exception
    md = "\n".join(m.value for m in at.markdown)
    assert "Prioritized Actions" in md and "$10,000.00" in md and "Not financial advice" in md
    at.chat_input[0].set_value("What happens if BTC drops 15%?").run()
    assert not at.exception
    assert len([c for c in at.chat_message if c.name == "assistant"]) == 2

    # About
    at.switch_page("pages/8_About.py").run()
    assert not at.exception
    assert any("About RiskGuard" in t.value for t in at.title)

    # Sidebar equity takes effect on the SAME rerun (analytics run before the sidebar is drawn)
    at.switch_page("streamlit_app.py").run()
    at.number_input(key="sidebar_equity_input").set_value(20_000.0).run()
    assert not at.exception
    assert {m.label: m.value for m in at.metric}["Account Equity"] == "$20,000.00"

    # Load Demo Data button restores the demo portfolio, rules and equity (undoes the edits above)
    next(b for b in at.button if b.label == "Load Demo Data").click().run()
    assert not at.exception
    assert at.session_state["rules"].max_leverage == 10.0
    assert {m.label: m.value for m in at.metric}["Account Equity"] == "$10,000.00"
    assert at.number_input(key="sidebar_equity_input").value == 10_000.0


def test_upload_is_applied_once_not_on_every_rerun() -> None:
    from streamlit.testing.v1 import AppTest

    def script():
        from types import SimpleNamespace

        import streamlit as st

        from riskguard.ui.state import is_new_upload

        f = SimpleNamespace(file_id="abc", name="p.csv", size=10)
        st.session_state.setdefault("applied", 0)
        if is_new_upload("positions", f):
            st.session_state["applied"] += 1
        st.write(f"applied={st.session_state['applied']}")

    at = AppTest.from_function(script, default_timeout=60).run()
    at.run()
    at.run()
    assert at.session_state["applied"] == 1


def test_web_app_live_mode_falls_back_to_sample_when_every_feed_fails(monkeypatch) -> None:
    from streamlit.testing.v1 import AppTest

    from riskguard.ui.state import fetch_market_bundle_cached, get_market_provider

    monkeypatch.setenv("RG_DATA_MODE", "auto")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    calls: list[str] = []

    def offline(self, method, url, **kwargs):
        calls.append(url)
        raise requests.ConnectionError("offline for test")

    monkeypatch.setattr(requests.Session, "request", offline)
    fetch_market_bundle_cached.clear()
    get_market_provider().cache.clear()
    at = AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=60).run()
    fetch_market_bundle_cached.clear()
    assert not at.exception
    assert calls == ["https://fapi.binance.com/fapi/v1/premiumIndex", "https://api.hyperliquid.xyz/info"]
    assert any("fell back to bundled sample" in c.value for c in at.sidebar.caption)
    assert {m.label: m.value for m in at.metric}["Gross Exposure"] == "$19,576"


def test_loss_limits_use_day_start_equity_not_current_equity() -> None:
    # Current equity $9,700 after a -$300 day: the day started at $10,000 -> loss = 300 / 10,000 = 3.00 % (BREACH
    # at a 3 % limit). Dividing by current equity would give 3.09 %; dividing by a $10,300 entry would give 2.91 %.
    ref = datetime(2025, 10, 8, 18, 0, tzinfo=timezone.utc)
    j = _journal_with_loss(300.0, ref - timedelta(hours=2), ref)
    cb = evaluate_circuit_breaker(AccountState(equity=9_700.0), RuleConfig(weekly_loss_limit_pct=50, max_drawdown_pct=50), j)
    assert cb.daily_pnl_pct == pytest.approx(-3.0)
    assert cb.status == "TRIPPED" and cb.trading_allowed is False
