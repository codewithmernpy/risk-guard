"""Portfolio exposure, BTC-beta adjustment, correlation, VaR/CVaR, and stress testing."""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from riskguard.analytics.margin import compute_maintenance_margin, lookup_margin_tier
from riskguard.analytics.positions import analyze_positions, committed_margin
from riskguard.data.indicators import (
    compute_btc_betas,
    compute_correlation_matrix,
    compute_daily_returns,
    compute_hourly_returns,
)
from riskguard.data.providers import MarketDataBundle, SymbolMarketData
from riskguard.models import (
    AccountState,
    MarginTier,
    PortfolioSummary,
    PositionAnalysis,
    PositionInput,
    StressScenarioResult,
)


def compute_historical_var_cvar(
    pnl_series: np.ndarray | pd.Series,
    alpha: float = 0.95,
) -> tuple[float, float]:
    """Compute positive 1-day Historical VaR and CVaR (Expected Shortfall) at confidence alpha."""
    arr = np.asarray(pnl_series, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return 0.0, 0.0
    losses = -arr
    var_val = float(np.quantile(losses, alpha))
    tail = losses[losses >= var_val]
    cvar_val = float(np.mean(tail)) if tail.size > 0 else var_val
    return max(0.0, var_val), max(0.0, cvar_val)


def _cf_z(z: float, skew: float, excess_kurt: float) -> float:
    return (
        z
        + ((z**2 - 1.0) * skew) / 6.0
        + ((z**3 - 3.0 * z) * excess_kurt) / 24.0
        - ((2.0 * z**3 - 5.0 * z) * (skew**2)) / 36.0
    )


def compute_cornish_fisher_var_cvar(
    pnl_series: np.ndarray | pd.Series,
    alpha: float = 0.95,
) -> tuple[float, float]:
    """Compute positive 1-day Parametric Cornish-Fisher VaR and CVaR at confidence alpha.

    Uses sample mean, std (ddof=1), Fisher skewness S, and excess kurtosis K of the PnL series.
    CVaR is the exact tail average of the Cornish-Fisher quantile function,
        ES = -mu + sigma * phi(c) / (1 - alpha) * [1 + S*c/6 - K*(1 - c^2)/24 + S^2*(1 - 2c^2)/36],
    with c = Phi^-1(1 - alpha), from the closed-form truncated normal moments
    int_{-inf}^{c} z^k phi(z) dz (k = 0..3). When S == K == 0 both VaR and CVaR reduce
    identically to the Normal parametric values.
    """
    arr = np.asarray(pnl_series, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 3:
        return compute_historical_var_cvar(arr, alpha=alpha)

    mu = float(np.mean(arr))
    sigma = float(np.std(arr, ddof=1))
    if sigma <= 1e-12:
        return max(0.0, -mu), max(0.0, -mu)

    skew = float(stats.skew(arr, bias=False)) if arr.size >= 4 else 0.0
    kurt = float(stats.kurtosis(arr, fisher=True, bias=False)) if arr.size >= 5 else 0.0
    if not math.isfinite(skew):
        skew = 0.0
    if not math.isfinite(kurt):
        kurt = 0.0

    # Clamp extreme higher-order moments only to keep CF expansion monotonic in pathological tiny samples
    z = float(stats.norm.ppf(1.0 - alpha))
    z_cf = _cf_z(z, skew, kurt)
    # Guard against CF polynomial inversion on extreme skew/kurtosis
    if z_cf > 0 and z < 0:
        z_cf = z

    var_val = max(0.0, -(mu + z_cf * sigma))

    # Exact expectation of the CF quantile over the lower tail u in (0, 1 - alpha)
    c = z
    tail_factor = (
        1.0
        + skew * c / 6.0
        - kurt * (1.0 - c * c) / 24.0
        + (skew**2) * (1.0 - 2.0 * c * c) / 36.0
    )
    cvar_val = -mu + sigma * float(stats.norm.pdf(c)) / (1.0 - alpha) * tail_factor
    return float(var_val), max(float(var_val), float(cvar_val), 0.0)


def _estimate_daily_pnl_series(
    analyses: Sequence[PositionAnalysis],
    daily_returns: pd.DataFrame,
) -> np.ndarray:
    """Reconstruct historical daily portfolio PnL ($) using current signed notionals."""
    if not analyses or daily_returns.empty:
        return np.array([], dtype=float)

    signed_by_sym: dict[str, float] = {}
    for a in analyses:
        sign = 1.0 if a.side == "long" else -1.0
        signed_by_sym[a.symbol] = signed_by_sym.get(a.symbol, 0.0) + sign * a.notional

    common_syms = [s for s in signed_by_sym if s in daily_returns.columns]
    if not common_syms:
        return np.array([], dtype=float)

    sub = daily_returns[common_syms].dropna(how="all").fillna(0.0)
    if sub.empty:
        return np.array([], dtype=float)

    weights = np.array([signed_by_sym[s] for s in common_syms], dtype=float)
    pnl = sub.to_numpy(dtype=float) @ weights
    return pnl


def _evaluate_shock_scenario(
    name: str,
    description: str,
    analyses: Sequence[PositionAnalysis],
    account: AccountState,
    shocks_by_symbol: Mapping[str, float],
    *,
    funding_override_daily_rate: float | None = None,
    funding_days: float = 0.0,
    margin_tiers: Mapping[str, Sequence[MarginTier]] | None = None,
) -> StressScenarioResult:
    """Compute shocked portfolio PnL, equity, and liquidations for one stress scenario.

    Liquidation checks:
      * isolated positions: shocked mark at/through the position's own liquidation price;
      * cross positions: account-level check. The cross pool (account equity minus the ring-fenced
        isolated equity of isolated positions) plus the shocked PnL of all cross positions is compared
        with the total maintenance margin of the cross positions at shocked prices. If the pool is
        at or below that margin, every cross position is liquidated together, which is how a cross
        account is liquidated (offsetting cross hedges are netted, as on the exchange).
    """
    total_pnl = 0.0
    liq_symbols: list[str] = []
    cross_mm_total = 0.0
    cross_pnl = 0.0
    cross_pool = float(account.equity) - sum(committed_margin(a) for a in analyses if a.margin_mode == "isolated")
    has_cross = any(a.margin_mode == "cross" for a in analyses)

    for a in analyses:
        sign = 1.0 if a.side == "long" else -1.0
        shock = float(shocks_by_symbol.get(a.symbol, 0.0))
        shocked_px = max(1e-9, a.mark_price * (1.0 + shock))
        price_pnl = sign * a.quantity * (shocked_px - a.mark_price)

        funding_pnl = 0.0
        if funding_override_daily_rate is not None and funding_days > 0:
            # Positive funding: longs pay, shorts receive
            funding_pnl = -sign * a.notional * funding_override_daily_rate * funding_days

        pos_pnl = price_pnl + funding_pnl
        if a.margin_mode == "isolated":
            # An isolated position can lose at most what is left of its own margin (margin + current uPnL);
            # beyond that it is liquidated and the rest of the account is untouched.
            pos_pnl = max(pos_pnl, -max(0.0, a.margin_used + a.unrealized_pnl))
        total_pnl += pos_pnl

        if a.margin_mode == "isolated":
            if a.liquidation_price is not None and (
                (a.side == "long" and shocked_px <= a.liquidation_price)
                or (a.side == "short" and shocked_px >= a.liquidation_price)
            ):
                if a.symbol not in liq_symbols:
                    liq_symbols.append(a.symbol)
        else:
            cross_pnl += pos_pnl
            shocked_notional = a.quantity * shocked_px
            tier = lookup_margin_tier(
                a.symbol,
                shocked_notional,
                margin_tiers=margin_tiers,
                fallback_mmr=account.default_mmr,
            )
            cross_mm_total += compute_maintenance_margin(shocked_notional, tier)

    shocked_equity = float(account.equity) + total_pnl
    # Account-level cross margin call: shocked cross pool at/below total cross maintenance margin
    if has_cross and cross_pool + cross_pnl <= cross_mm_total:
        for a in analyses:
            if a.margin_mode == "cross" and a.symbol not in liq_symbols:
                liq_symbols.append(a.symbol)

    pnl_pct = (total_pnl / float(account.equity) * 100.0) if account.equity > 0 else 0.0
    return StressScenarioResult(
        scenario=name,
        description=description,
        portfolio_pnl_usd=float(total_pnl),
        portfolio_pnl_pct=float(pnl_pct),
        shocked_equity_usd=float(shocked_equity),
        liquidated_symbols=liq_symbols,
        liquidation_count=len(liq_symbols),
        symbol_shocks_pct={k: round(v * 100.0, 2) for k, v in shocks_by_symbol.items()},
    )


def analyze_portfolio(
    positions: Sequence[PositionInput] | Sequence[PositionAnalysis],
    account: AccountState,
    market: MarketDataBundle | Mapping[str, SymbolMarketData] | None = None,
    *,
    var_lookback_days: int = 180,
    margin_tiers: Mapping[str, Sequence[MarginTier]] | None = None,
) -> PortfolioSummary:
    """Compute portfolio exposures, BTC beta, correlations, HHI, VaR/CVaR, and stress tests."""
    if not positions:
        return PortfolioSummary(
            equity=float(account.equity),
            position_count=0,
            gross_exposure_usd=0.0,
            net_exposure_usd=0.0,
            long_exposure_usd=0.0,
            short_exposure_usd=0.0,
            gross_leverage=0.0,
            net_leverage=0.0,
            liquidated_positions=0,
            max_effective_leverage=0.0,
            avg_effective_leverage=0.0,
            btc_beta_adj_net_usd=0.0,
            btc_beta_adj_gross_usd=0.0,
            btc_beta_adj_net_pct=0.0,
            total_unrealized_pnl_usd=0.0,
            total_daily_funding_cost_usd=0.0,
            known_stop_risk_usd=0.0,
            known_stop_risk_pct=0.0,
            unstopped_positions=0,
            unstopped_notional_usd=0.0,
            worst_case_stops_usd=0.0,
            worst_case_stops_pct=0.0,
            hhi=0.0,
            hhi_10000=0.0,
            effective_num_positions=0.0,
            max_pairwise_corr_30d=0.0,
            max_pairwise_corr_90d=0.0,
            var_95_hist_usd=0.0,
            var_99_hist_usd=0.0,
            cvar_95_hist_usd=0.0,
            cvar_99_hist_usd=0.0,
            var_95_cf_usd=0.0,
            var_99_cf_usd=0.0,
            cvar_95_cf_usd=0.0,
            cvar_99_cf_usd=0.0,
            var_lookback_days=0,
        )

    if isinstance(positions[0], PositionAnalysis):
        analyses = list(positions)  # type: ignore[arg-type]
    else:
        analyses = analyze_positions(positions, account, market=market, margin_tiers=margin_tiers)  # type: ignore[arg-type]

    # Snapshot-only mappings (accepted by analyze_positions) carry no candle history
    candles = market.candles if isinstance(market, MarketDataBundle) else {}
    hourly_rets = compute_hourly_returns(candles, window_hours=2160) if candles else pd.DataFrame()
    daily_rets = compute_daily_returns(candles, lookback_days=var_lookback_days) if candles else pd.DataFrame()

    betas = compute_btc_betas(hourly_rets) if not hourly_rets.empty else {}
    for a in analyses:
        if a.symbol not in betas:
            betas[a.symbol] = 1.0

    # Restrict correlation matrix to held symbols (+ BTCUSDT if helpful, keeping clean portfolio matrix)
    held_syms = list(dict.fromkeys(a.symbol for a in analyses))
    held_hourly = hourly_rets[[s for s in held_syms if s in hourly_rets.columns]] if not hourly_rets.empty else pd.DataFrame()
    corr_30d_df = compute_correlation_matrix(held_hourly, window_hours=720)
    corr_90d_df = compute_correlation_matrix(held_hourly, window_hours=2160)

    def _max_off_diag(corr_df: pd.DataFrame) -> float:
        if corr_df.empty or len(corr_df) < 2:
            return 0.0
        vals = corr_df.to_numpy(dtype=float).copy()
        np.fill_diagonal(vals, -np.inf)
        m = float(np.max(vals))
        return m if math.isfinite(m) else 0.0

    long_usd = sum(a.notional for a in analyses if a.side == "long")
    short_usd = sum(a.notional for a in analyses if a.side == "short")
    gross_usd = long_usd + short_usd
    net_usd = long_usd - short_usd

    per_asset: dict[str, dict[str, float]] = {}
    for a in analyses:
        bucket = per_asset.setdefault(
            a.symbol,
            {"long_usd": 0.0, "short_usd": 0.0, "gross_usd": 0.0, "net_usd": 0.0, "pct_equity": 0.0, "beta": betas.get(a.symbol, 1.0)},
        )
        if a.side == "long":
            bucket["long_usd"] += a.notional
            bucket["net_usd"] += a.notional
        else:
            bucket["short_usd"] += a.notional
            bucket["net_usd"] -= a.notional
        bucket["gross_usd"] += a.notional
        eq_val = float(account.equity)
        bucket["pct_equity"] = (bucket["gross_usd"] / eq_val * 100.0) if eq_val > 0 else math.nan

    # HHI concentration over gross asset weights
    if gross_usd > 0:
        weights = [v["gross_usd"] / gross_usd for v in per_asset.values()]
        hhi = float(sum(w * w for w in weights))
    else:
        hhi = 0.0
    eff_n = (1.0 / hhi) if hhi > 1e-9 else 0.0

    btc_beta_net = sum((1.0 if a.side == "long" else -1.0) * a.notional * betas.get(a.symbol, 1.0) for a in analyses)
    btc_beta_gross = sum(a.notional * abs(betas.get(a.symbol, 1.0)) for a in analyses)

    total_upnl = sum(a.unrealized_pnl for a in analyses)
    total_funding = sum(a.funding_cost_daily_usd for a in analyses)

    known_stop_risk = sum(a.risk_to_stop_usd for a in analyses if a.risk_to_stop_usd is not None)
    # Unprotected = no stop, or a stop already crossed by the mark (it is not limiting the loss)
    unstopped = [a for a in analyses if a.missing_stop or a.invalid_stop_direction]
    unstopped_notional = sum(a.notional for a in unstopped)
    # Worst-case if stops hit: known stop losses + isolated margin (or equity-capped notional) at risk on unstopped positions
    unstopped_margin_at_risk = sum(
        min(a.margin_used if a.margin_mode == "isolated" else a.notional, float(account.equity))
        for a in unstopped
    )
    worst_case_stops = known_stop_risk + unstopped_margin_at_risk

    # VaR & CVaR
    pnl_series = _estimate_daily_pnl_series(analyses, daily_rets)
    var95_h, cvar95_h = compute_historical_var_cvar(pnl_series, alpha=0.95)
    var99_h, cvar99_h = compute_historical_var_cvar(pnl_series, alpha=0.99)
    var95_cf, cvar95_cf = compute_cornish_fisher_var_cvar(pnl_series, alpha=0.95)
    var99_cf, cvar99_cf = compute_cornish_fisher_var_cvar(pnl_series, alpha=0.99)

    # Stress scenarios
    daily_vols = {}
    for sym in held_syms:
        if sym in daily_rets.columns and len(daily_rets[sym].dropna()) >= 5:
            daily_vols[sym] = float(daily_rets[sym].dropna().std(ddof=1))
        else:
            daily_vols[sym] = 0.04

    stress_results: list[StressScenarioResult] = []
    for btc_drop in (-0.05, -0.10, -0.20):
        shocks = {sym: betas.get(sym, 1.0) * btc_drop for sym in held_syms}
        stress_results.append(
            _evaluate_shock_scenario(
                name=f"BTC {int(btc_drop * 100)}%",
                description=f"BTC moves {btc_drop * 100:.0f}% with 90d hourly beta propagation to alts.",
                analyses=analyses,
                account=account,
                shocks_by_symbol=shocks,
                margin_tiers=margin_tiers,
            )
        )

    # Alt crash: BTC -10%, non-BTC alts move 1.5x BTC (-15%)
    alt_shocks = {
        sym: (-0.10 if sym == "BTCUSDT" else -0.15)
        for sym in held_syms
    }
    stress_results.append(
        _evaluate_shock_scenario(
            name="Alt Crash (1.5x BTC)",
            description="BTC drops -10% while non-BTC altcoins drop -15% (1.5x BTC beta shock).",
            analyses=analyses,
            account=account,
            shocks_by_symbol=alt_shocks,
            margin_tiers=margin_tiers,
        )
    )

    # Correlation-to-1: every asset moves together by max(10%, 2.5 daily sigmas). Both directions are
    # evaluated and the one that hurts this portfolio more is reported (a crash for net-long books,
    # a squeeze for net-short books), so hedged or short books are stressed too.
    corr1_moves = {sym: max(0.10, 2.5 * daily_vols.get(sym, 0.04)) for sym in held_syms}
    corr1_down = _evaluate_shock_scenario(
        name="Correlation -> 1 Crash",
        description="All assets sell off together by max(10%, 2.5 daily sigmas) as correlation spikes to 1.0 "
        "(adverse direction for this portfolio).",
        analyses=analyses,
        account=account,
        shocks_by_symbol={sym: -m for sym, m in corr1_moves.items()},
        margin_tiers=margin_tiers,
    )
    corr1_up = _evaluate_shock_scenario(
        name="Correlation -> 1 Squeeze",
        description="All assets rally together by max(10%, 2.5 daily sigmas) as correlation spikes to 1.0 "
        "(adverse direction for this portfolio).",
        analyses=analyses,
        account=account,
        shocks_by_symbol=dict(corr1_moves),
        margin_tiers=margin_tiers,
    )
    worse_corr1 = corr1_up if (corr1_up.portfolio_pnl_usd, -corr1_up.liquidation_count) < (
        corr1_down.portfolio_pnl_usd,
        -corr1_down.liquidation_count,
    ) else corr1_down
    stress_results.append(worse_corr1)

    # Funding spike: 0.10% per 8h (0.30%/day) sustained for 7 days
    stress_results.append(
        _evaluate_shock_scenario(
            name="Funding Spike (7d @ 0.10%/8h)",
            description="Funding rate spikes to +0.10% per 8h (0.30%/day, ~109.5% APR) sustained for 7 days.",
            analyses=analyses,
            account=account,
            shocks_by_symbol={sym: 0.0 for sym in held_syms},
            funding_override_daily_rate=0.003,
            funding_days=7.0,
            margin_tiers=margin_tiers,
        )
    )

    eq = float(account.equity)
    liquidated_positions = sum(1 for a in analyses if a.liquidated)
    finite_eff_levs = [a.effective_leverage for a in analyses if math.isfinite(a.effective_leverage)]
    max_effective_leverage = float(max(finite_eff_levs)) if finite_eff_levs else math.nan
    avg_effective_leverage = float(sum(finite_eff_levs) / len(finite_eff_levs)) if finite_eff_levs else math.nan

    return PortfolioSummary(
        equity=eq,
        position_count=len(analyses),
        gross_exposure_usd=float(gross_usd),
        net_exposure_usd=float(net_usd),
        long_exposure_usd=float(long_usd),
        short_exposure_usd=float(short_usd),
        gross_leverage=float(gross_usd / eq) if eq > 0 else math.nan,
        net_leverage=float(net_usd / eq) if eq > 0 else math.nan,
        liquidated_positions=liquidated_positions,
        max_effective_leverage=max_effective_leverage,
        avg_effective_leverage=avg_effective_leverage,
        btc_beta_adj_net_usd=float(btc_beta_net),
        btc_beta_adj_gross_usd=float(btc_beta_gross),
        btc_beta_adj_net_pct=float(btc_beta_net / eq * 100.0) if eq > 0 else math.nan,
        total_unrealized_pnl_usd=float(total_upnl),
        total_daily_funding_cost_usd=float(total_funding),
        known_stop_risk_usd=float(known_stop_risk),
        known_stop_risk_pct=float(known_stop_risk / eq * 100.0) if eq > 0 else math.nan,
        unstopped_positions=len(unstopped),
        unstopped_notional_usd=float(unstopped_notional),
        worst_case_stops_usd=float(worst_case_stops),
        worst_case_stops_pct=float(worst_case_stops / eq * 100.0) if eq > 0 else math.nan,
        hhi=hhi,
        hhi_10000=hhi * 10000.0,
        effective_num_positions=eff_n,
        max_pairwise_corr_30d=_max_off_diag(corr_30d_df),
        max_pairwise_corr_90d=_max_off_diag(corr_90d_df),
        var_95_hist_usd=var95_h,
        var_99_hist_usd=var99_h,
        cvar_95_hist_usd=cvar95_h,
        cvar_99_hist_usd=cvar99_h,
        var_95_cf_usd=var95_cf,
        var_99_cf_usd=var99_cf,
        cvar_95_cf_usd=cvar95_cf,
        cvar_99_cf_usd=cvar99_cf,
        var_lookback_days=int(len(pnl_series)),
        per_asset_exposure=per_asset,
        betas_vs_btc={k: round(float(v), 4) for k, v in betas.items()},
        corr_30d=corr_30d_df.round(4).to_dict() if not corr_30d_df.empty else {},
        corr_90d=corr_90d_df.round(4).to_dict() if not corr_90d_df.empty else {},
        stress_scenarios=stress_results,
    )
