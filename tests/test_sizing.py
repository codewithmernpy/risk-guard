"""Tests for sizing formulas and pre-trade GO / REDUCE SIZE / NO-GO verdicts."""

from __future__ import annotations

import pytest

from riskguard.analytics.sizing import max_safe_leverage_for_stop, pre_trade_check, recommend_sizes
from riskguard.models import AccountState, PositionInput, RuleConfig


def test_sizing_formulas_exact() -> None:
    account = AccountState(equity=10_000.0)
    # Long @ 100, stop @ 95 (dist=5), ATR=4, risk=1% ($100)
    recs = recommend_sizes(
        symbol="SOLUSDT",
        side="long",
        entry_price=100.0,
        stop_price=95.0,
        account=account,
        atr_1d=4.0,
        risk_pct=1.0,
        leverage=10.0,
        margin_mode="isolated",
        k_atr_buffer=1.5,
        kelly_fraction=0.25,
        max_kelly_risk_pct=3.0,
        win_rate=0.50,
        payoff_ratio=2.0,  # full kelly = 0.5 - 0.5/2 = 0.25 -> 0.25x = 6.25% capped at 3%
    )
    by_method = {r.method: r for r in recs}

    # Fixed fractional: $100 / $5 = 20 units ($2,000 notional)
    assert by_method["Fixed Fractional"].quantity == pytest.approx(20.0)
    assert by_method["Fixed Fractional"].risk_usd == pytest.approx(100.0)

    # ATR Vol-Targeted: $100 / $4 = 25 units ($2,500 notional)
    assert by_method["ATR Volatility-Targeted"].quantity == pytest.approx(25.0)

    # Capped Fractional Kelly: capped at 3% = $300 / $5 = 60 units
    assert by_method["Capped Fractional Kelly"].quantity == pytest.approx(60.0)
    assert by_method["Capped Fractional Kelly"].risk_pct == pytest.approx(3.0)

    # Max-leverage-safe: target LP = 95 - 1.5*4 = 89.0 -> max safe lev ~ 8.73x
    safe_lev = max_safe_leverage_for_stop(100.0, 95.0, 4.0, "long", mmr=0.005, k_atr=1.5)
    assert safe_lev < 10.0
    assert by_method["Max-Leverage-Safe"].max_safe_leverage == pytest.approx(safe_lev, abs=0.02)


def test_pre_trade_verdicts_go_reduce_nogo() -> None:
    account = AccountState(equity=10_000.0)
    rules = RuleConfig(max_risk_per_trade_pct=2.0, max_total_open_risk_pct=5.0, max_leverage=10.0, mandatory_stop=True)

    # 1. GO: small conservative trade (risk 1% = $100, lev 3x, stop well inside LP)
    go_trade = PositionInput(symbol="BTCUSDT", side="long", entry_price=100_000.0, quantity=0.02, leverage=3.0, margin_mode="isolated", stop_price=95_000.0)
    res_go = pre_trade_check(go_trade, [], account, market=None, rules=rules)
    assert res_go.verdict == "GO"

    # 2. REDUCE SIZE: valid stop before LP, 3x leverage, but risk = 4% ($400 > 2% limit)
    big_trade = PositionInput(symbol="BTCUSDT", side="long", entry_price=100_000.0, quantity=0.08, leverage=3.0, margin_mode="isolated", stop_price=95_000.0)
    res_red = pre_trade_check(big_trade, [], account, market=None, rules=rules)
    assert res_red.verdict == "REDUCE SIZE"
    assert res_red.recommended_quantity is not None
    assert res_red.recommended_quantity <= 0.04 + 1e-9

    # 3. NO-GO: stop beyond liquidation (25x long @ 100 with stop @ 90)
    nogo_trade = PositionInput(symbol="SOLUSDT", side="long", entry_price=100.0, quantity=20.0, leverage=25.0, margin_mode="isolated", stop_price=90.0)
    res_nogo = pre_trade_check(nogo_trade, [], account, market=None, rules=rules)
    assert res_nogo.verdict == "NO-GO"
    assert res_nogo.proposed_analysis.stop_beyond_liquidation is True
