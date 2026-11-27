"""Tests for Binance USDT-M liquidation price formula, cross approximation, and stop-beyond-LP detection."""

from __future__ import annotations

import math
from datetime import datetime, timezone

import pytest

from riskguard.analytics.margin import compute_liquidation_price, lookup_margin_tier
from riskguard.analytics.portfolio import analyze_portfolio
from riskguard.analytics.positions import analyze_position, analyze_positions, effective_leverage_label
from riskguard.analytics.rules import evaluate_rules
from riskguard.data.providers import SymbolMarketData
from riskguard.models import AccountState, PositionInput, RuleConfig


def _snap(symbol: str, mark: float, atr_1d: float = 10.0, funding_8h: float = 0.0001) -> dict[str, SymbolMarketData]:
    return {
        symbol: SymbolMarketData(
            symbol=symbol,
            mark_price=mark,
            atr_1d=atr_1d,
            funding_rate_8h=funding_8h,
            funding_rate_daily=funding_8h * 3.0,
            source="sample",
            as_of=datetime(2025, 10, 31, tzinfo=timezone.utc),
        )
    }


def test_liquidation_price_hand_calculated_isolated_long() -> None:
    # Hand calculation:
    # EP = 100, Q = 10 (notional 1000), lev = 10x -> WB = 100, MMR = 0.005, cumB = 0
    # LP = (100 + 0 - 1*10*100) / (10*0.005 - 10) = -900 / -9.95 = 90.4522613065...
    lp = compute_liquidation_price(
        wallet_balance=100.0,
        quantity=10.0,
        entry_price=100.0,
        side="long",
        mmr=0.005,
        cum_b=0.0,
    )
    assert lp is not None
    assert lp == pytest.approx(900.0 / 9.95, rel=1e-9)


def test_liquidation_price_hand_calculated_isolated_short() -> None:
    # Hand calculation:
    # EP = 100, Q = 10, lev = 10x -> WB = 100, MMR = 0.005, cumB = 0
    # LP = (100 + 0 - (-1)*10*100) / (10*0.005 - (-1)*10) = 1100 / 10.05 = 109.452736318...
    lp = compute_liquidation_price(
        wallet_balance=100.0,
        quantity=10.0,
        entry_price=100.0,
        side="short",
        mmr=0.005,
        cum_b=0.0,
    )
    assert lp is not None
    assert lp == pytest.approx(1100.0 / 10.05, rel=1e-9)


def test_liquidation_price_higher_tier_with_cumb() -> None:
    # BTCUSDT notional = 100,000 -> Tier 2 (50k..250k): MMR=0.005, cumB=50
    tier = lookup_margin_tier("BTCUSDT", 100_000.0)
    assert tier.mmr == pytest.approx(0.005)
    assert tier.cum_b == pytest.approx(50.0)
    # Q = 1.0, EP = 100,000, lev = 20x -> WB = 5,000
    # LP = (5000 + 50 - 100000) / (0.005 - 1) = -94950 / -0.995 = 95427.135678...
    lp = compute_liquidation_price(
        wallet_balance=5000.0,
        quantity=1.0,
        entry_price=100_000.0,
        side="long",
        mmr=tier.mmr,
        cum_b=tier.cum_b,
    )
    assert lp == pytest.approx(94950.0 / 0.995, rel=1e-9)


def test_cross_margin_documented_approximation() -> None:
    # Account equity = 10,000; position 1: BTC long 0.1 @ 100,000 (notional 10,000, MM=40)
    # position 2: ETH long 2.0 @ 4,000 (notional 8,000, MM=32)
    # For BTC at mark == entry: WB_cross = 10,000 - 32 = 9,968 -> LP = (9968 - 10000) / (0.1*0.004 - 0.1)
    account = AccountState(equity=10_000.0)
    p1 = PositionInput(symbol="BTCUSDT", side="long", entry_price=100_000.0, quantity=0.1, leverage=10.0, margin_mode="cross", stop_price=95_000.0)
    p2 = PositionInput(symbol="ETHUSDT", side="long", entry_price=4_000.0, quantity=2.0, leverage=10.0, margin_mode="cross", stop_price=3_700.0)
    market = {**_snap("BTCUSDT", 100_000.0, atr_1d=2500.0), **_snap("ETHUSDT", 4_000.0, atr_1d=150.0)}

    analyses = analyze_positions([p1, p2], account, market=market)
    btc_a = analyses[0]
    expected_wb = 10_000.0 - (8_000.0 * 0.004)
    expected_lp = (expected_wb - 0.1 * 100_000.0) / (0.1 * 0.004 - 0.1)
    assert btc_a.liquidation_price == pytest.approx(expected_lp, rel=1e-6)


def test_stop_beyond_liquidation_and_missing_stop_detection() -> None:
    account = AccountState(equity=10_000.0)
    # 25x long @ 100 -> LP ~ 96.48; stop @ 93.0 is beyond LP!
    p_bad_long = PositionInput(symbol="SOLUSDT", side="long", entry_price=100.0, quantity=50.0, leverage=25.0, margin_mode="isolated", stop_price=93.0)
    # 25x short @ 100 -> LP ~ 103.48; stop @ 106.0 is beyond LP!
    p_bad_short = PositionInput(symbol="DOGEUSDT", side="short", entry_price=100.0, quantity=50.0, leverage=25.0, margin_mode="isolated", stop_price=106.0)
    # Missing stop
    p_no_stop = PositionInput(symbol="XRPUSDT", side="long", entry_price=2.5, quantity=1000.0, leverage=5.0, margin_mode="isolated")

    market = {
        **_snap("SOLUSDT", 100.0, atr_1d=5.0),
        **_snap("DOGEUSDT", 100.0, atr_1d=5.0),
        **_snap("XRPUSDT", 2.5, atr_1d=0.15),
    }

    a_long = analyze_position(p_bad_long, account, market=market)
    a_short = analyze_position(p_bad_short, account, market=market)
    a_no_stop = analyze_position(p_no_stop, account, market=market)

    assert a_long.stop_beyond_liquidation is True
    assert a_long.danger_level == "DANGER"
    assert any("beyond liquidation" in w for w in a_long.warnings)

    assert a_short.stop_beyond_liquidation is True
    assert a_short.danger_level == "DANGER"

    assert a_no_stop.missing_stop is True
    assert a_no_stop.danger_level == "DANGER"
    assert a_no_stop.risk_to_stop_usd is None


def test_liquidated_position_marks_flag_nan_eff_lev_and_flags_breach() -> None:
    account = AccountState(equity=10_000.0)
    # Demo DOGEUSDT long 25x @ 0.1865 (LP ~0.17994), mark crashes to 0.05 (well past LP, pos equity < 0)
    pos_liq_long = PositionInput(
        symbol="DOGEUSDT",
        side="long",
        entry_price=0.1865,
        quantity=26_810.0,
        leverage=25.0,
        margin_mode="isolated",
        stop_price=0.1720,
    )
    # Short position past its LP (20x @ 100 -> LP ~104.48, mark rises to 110)
    pos_liq_short = PositionInput(
        symbol="BNBUSDT",
        side="short",
        entry_price=100.0,
        quantity=10.0,
        leverage=20.0,
        margin_mode="isolated",
        stop_price=102.0,
    )
    # Healthy 5x isolated long on SOL (entry 100, mark 105 -> margin 200, uPnL +50, pos_eq 250, notional 1050)
    pos_healthy = PositionInput(
        symbol="SOLUSDT",
        side="long",
        entry_price=100.0,
        quantity=10.0,
        leverage=5.0,
        margin_mode="isolated",
        stop_price=90.0,
    )
    market = {
        **_snap("DOGEUSDT", 0.05, atr_1d=0.01),
        **_snap("BNBUSDT", 110.0, atr_1d=5.0),
        **_snap("SOLUSDT", 105.0, atr_1d=4.0),
    }

    a_liq_long = analyze_position(pos_liq_long, account, market=market)
    a_liq_short = analyze_position(pos_liq_short, account, market=market)
    a_healthy = analyze_position(pos_healthy, account, market=market)

    for a_liq in (a_liq_long, a_liq_short):
        assert a_liq.liquidated is True
        assert a_liq.at_or_past_liquidation is True
        assert math.isnan(a_liq.effective_leverage)
        assert not math.isinf(a_liq.effective_leverage)
        assert not (a_liq.effective_leverage > 1e6)
        assert a_liq.distance_to_liq_pct is None
        assert a_liq.danger_level == "DANGER"
        assert effective_leverage_label(a_liq) == "LIQUIDATED"

    # Healthy position effective leverage unchanged: notional / (margin_used + uPnL)
    expected_healthy_eff = (10.0 * 105.0) / ((10.0 * 100.0 / 5.0) + 10.0 * (105.0 - 100.0))
    assert a_healthy.liquidated is False
    assert a_healthy.effective_leverage == pytest.approx(expected_healthy_eff, rel=1e-12)
    assert effective_leverage_label(a_healthy) == f"{expected_healthy_eff:.2f}x"

    # Mixed portfolio: averages ignore NaN, while danger/breach logic still flags liquidated positions
    mixed_port = analyze_portfolio([pos_liq_long, pos_healthy], account, market=market)
    assert mixed_port.liquidated_positions == 1
    assert mixed_port.max_effective_leverage == pytest.approx(expected_healthy_eff, rel=1e-12)
    assert mixed_port.avg_effective_leverage == pytest.approx(expected_healthy_eff, rel=1e-12)

    # Even with generous max_leverage=50, liquidated position causes BREACH in rules
    evals, _ = evaluate_rules(
        [a_liq_long, a_healthy],
        mixed_port,
        account,
        RuleConfig(max_leverage=50.0),
    )
    by_id = {e.rule_id: e for e in evals}
    assert by_id["liquidation_safety"].status == "BREACH"
    assert "DOGEUSDT" in by_id["liquidation_safety"].affected_symbols
    assert by_id["max_leverage"].status == "BREACH"
    assert "DOGEUSDT" in by_id["max_leverage"].affected_symbols

    # All-liquidated portfolio does not produce huge numbers or look safe
    liq_only_port = analyze_portfolio([pos_liq_long], account, market=market)
    assert liq_only_port.liquidated_positions == 1
    assert math.isnan(liq_only_port.max_effective_leverage)
    assert math.isnan(liq_only_port.avg_effective_leverage)


def test_healthy_positions_effective_leverage_unchanged() -> None:
    account = AccountState(equity=10_000.0)
    # Healthy isolated position in slight drawdown (still well above LP ~90.45)
    iso_pos = PositionInput(
        symbol="SOLUSDT",
        side="long",
        entry_price=100.0,
        quantity=10.0,
        leverage=10.0,
        margin_mode="isolated",
        stop_price=92.0,
    )
    cross_pos = PositionInput(
        symbol="ETHUSDT",
        side="long",
        entry_price=4_000.0,
        quantity=1.0,
        leverage=5.0,
        margin_mode="cross",
        stop_price=3_600.0,
    )
    market = {**_snap("SOLUSDT", 96.0, atr_1d=3.0), **_snap("ETHUSDT", 4_200.0, atr_1d=150.0)}
    analyses = analyze_positions([iso_pos, cross_pos], account, market=market)
    a_iso, a_cross = analyses

    # Isolated: notional=960, margin=100, uPnL=-40 -> pos_equity=60 -> eff_lev = 960/60 = 16.0
    assert a_iso.liquidated is False
    assert a_iso.effective_leverage == pytest.approx(960.0 / 60.0, rel=1e-12)
    assert effective_leverage_label(a_iso) == "16.00x"

    # Cross: notional=4200, account equity=10000 -> eff_lev = 0.42
    assert a_cross.liquidated is False
    assert a_cross.effective_leverage == pytest.approx(4_200.0 / 10_000.0, rel=1e-12)
    assert effective_leverage_label(a_cross) == "0.42x"
