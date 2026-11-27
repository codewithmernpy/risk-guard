"""Tests for VaR/CVaR (historical & Cornish-Fisher), BTC beta, HHI, and stress beta propagation."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from riskguard.analytics.portfolio import (
    analyze_portfolio,
    compute_cornish_fisher_var_cvar,
    compute_historical_var_cvar,
)
from riskguard.data.providers import MarketDataBundle, SymbolMarketData
from riskguard.models import AccountState, PositionInput


def test_var_on_known_normal_and_historical_samples() -> None:
    # Known exact symmetric normal-like sample (mean=0, std=100, skew=0, excess_kurt=0 approx)
    rng = np.random.default_rng(123)
    pnl = rng.normal(loc=0.0, scale=100.0, size=20_000)

    var95_h, cvar95_h = compute_historical_var_cvar(pnl, alpha=0.95)
    var99_h, cvar99_h = compute_historical_var_cvar(pnl, alpha=0.99)
    var95_cf, cvar95_cf = compute_cornish_fisher_var_cvar(pnl, alpha=0.95)
    var99_cf, cvar99_cf = compute_cornish_fisher_var_cvar(pnl, alpha=0.99)

    assert var95_h == pytest.approx(164.485, rel=0.04)
    assert cvar95_h == pytest.approx(206.27, rel=0.05)
    assert var95_cf == pytest.approx(164.485, rel=0.04)
    assert cvar95_cf == pytest.approx(206.27, rel=0.05)
    assert var99_h > var95_h
    assert cvar99_cf >= var99_cf

    # Verify Cornish-Fisher increases tail VaR under left skew & fat tails
    skewed_pnl = np.concatenate([pnl, np.array([-800.0] * 120)])
    var99_skew_cf, _ = compute_cornish_fisher_var_cvar(skewed_pnl, alpha=0.99)
    normal_var99 = float(np.std(skewed_pnl, ddof=1) * stats.norm.ppf(0.99) - np.mean(skewed_pnl))
    assert var99_skew_cf > normal_var99


def test_stress_beta_propagation_and_hhi() -> None:
    # Construct synthetic hourly candles where SOL moves exactly 2.0x BTC hourly returns
    n_hours = 200
    t0 = datetime(2025, 10, 1, tzinfo=timezone.utc)
    times = [t0 + timedelta(hours=i) for i in range(n_hours)]
    rng = np.random.default_rng(7)
    btc_rets = rng.normal(0.0, 0.005, size=n_hours - 1)
    sol_rets = 2.0 * btc_rets

    btc_prices = [100_000.0]
    sol_prices = [200.0]
    for r_b, r_s in zip(btc_rets, sol_rets):
        btc_prices.append(btc_prices[-1] * (1.0 + r_b))
        sol_prices.append(sol_prices[-1] * (1.0 + r_s))

    def _df(pxs: list[float]) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "open_time": times,
                "open": pxs,
                "high": [p * 1.002 for p in pxs],
                "low": [p * 0.998 for p in pxs],
                "close": pxs,
                "volume": [100.0] * len(pxs),
                "quote_volume": [p * 100.0 for p in pxs],
                "trades": [50] * len(pxs),
                "taker_buy_base": [50.0] * len(pxs),
            }
        )

    btc_df = _df(btc_prices)
    sol_df = _df(sol_prices)
    now = times[-1]
    bundle = MarketDataBundle(
        source="sample",
        badge_label="TEST",
        reason="unit test",
        is_live=False,
        fetched_at=now,
        candles={"BTCUSDT": btc_df, "SOLUSDT": sol_df},
        snapshots={
            "BTCUSDT": SymbolMarketData(symbol="BTCUSDT", mark_price=btc_prices[-1], atr_1d=2500.0, as_of=now),
            "SOLUSDT": SymbolMarketData(symbol="SOLUSDT", mark_price=sol_prices[-1], atr_1d=10.0, as_of=now),
        },
    )

    account = AccountState(equity=10_000.0)
    # 100% notional in SOL ($5,000 long, 25x isolated -> LP ~3.6% below mark!)
    pos = PositionInput(
        symbol="SOLUSDT",
        side="long",
        entry_price=sol_prices[-1],
        notional=5_000.0,
        leverage=25.0,
        margin_mode="isolated",
        stop_price=sol_prices[-1] * 0.90,
    )
    summary = analyze_portfolio([pos], account, market=bundle)

    assert summary.betas_vs_btc["SOLUSDT"] == pytest.approx(2.0, abs=0.05)
    assert summary.hhi == pytest.approx(1.0)
    assert summary.btc_beta_adj_net_usd == pytest.approx(10_000.0, rel=0.05)

    # Under BTC -10%, SOL shock is ~ -20% -> linear PnL ~ -$1,000, but the 25x isolated position is liquidated
    # and can lose only its own margin: 5,000 / 25 = $200 (uPnL 0 at the mark).
    btc_10 = next(s for s in summary.stress_scenarios if s.scenario == "BTC -10%")
    assert btc_10.portfolio_pnl_usd == pytest.approx(-200.0, rel=0.05)
    assert "SOLUSDT" in btc_10.liquidated_symbols
