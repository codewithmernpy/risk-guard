"""Tests for drawdown step-down schedule, CPPI multiplier, and analytical/MC risk of ruin."""

from __future__ import annotations

import pytest

from riskguard.analytics.drawdown import (
    analytical_risk_of_ruin,
    compute_drawdown_plan,
    estimate_risk_of_ruin,
)


def test_drawdown_schedule_step_down_and_cppi() -> None:
    # At 0% DD: 1.0x multiplier, full base risk
    p0 = compute_drawdown_plan(current_equity=10_000.0, peak_equity=10_000.0, base_risk_pct=1.0, max_dd_limit_pct=15.0)
    assert p0.step_down_multiplier == pytest.approx(1.0)
    assert p0.cppi_multiplier == pytest.approx(1.0)
    assert p0.recommended_risk_pct == pytest.approx(1.0)
    assert p0.trading_halted is False

    # At 8% DD (6..10% tier): step-down 0.50x -> 0.50%
    p8 = compute_drawdown_plan(current_equity=9_200.0, peak_equity=10_000.0, base_risk_pct=1.0, max_dd_limit_pct=15.0)
    assert p8.current_dd_pct == pytest.approx(8.0)
    assert p8.step_down_multiplier == pytest.approx(0.50)
    assert p8.cppi_risk_pct < 1.0
    assert p8.recommended_risk_pct <= 0.50

    # At 15% DD (limit hit): trading halted -> 0% risk
    p15 = compute_drawdown_plan(current_equity=8_500.0, peak_equity=10_000.0, base_risk_pct=1.0, max_dd_limit_pct=15.0)
    assert p15.trading_halted is True
    assert p15.recommended_risk_pct == pytest.approx(0.0)


def test_risk_of_ruin_sanity() -> None:
    # Negative or zero expectancy -> analytical ruin = 1.0
    assert analytical_risk_of_ruin(win_rate=0.40, payoff_ratio=1.0, risk_per_trade_pct=2.0) == pytest.approx(1.0)

    # Higher risk per trade increases risk of ruin monotonically
    ror_low_risk = estimate_risk_of_ruin(
        win_rate=0.55,
        payoff_ratio=1.5,
        risk_per_trade_pct=1.0,
        ruin_drawdown_pct=25.0,
        simulations=1500,
        horizon_trades=200,
        seed=42,
    )
    ror_high_risk = estimate_risk_of_ruin(
        win_rate=0.55,
        payoff_ratio=1.5,
        risk_per_trade_pct=5.0,
        ruin_drawdown_pct=25.0,
        simulations=1500,
        horizon_trades=200,
        seed=42,
    )
    assert ror_low_risk.analytical_ruin_prob < ror_high_risk.analytical_ruin_prob
    assert ror_low_risk.mc_ruin_prob_flat < ror_high_risk.mc_ruin_prob_flat
    assert ror_high_risk.mc_ruin_prob_scheduled <= ror_high_risk.mc_ruin_prob_flat + 1e-6
