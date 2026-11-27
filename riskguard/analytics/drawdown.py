"""Drawdown-based risk schedules (step-down + CPPI) and Risk of Ruin estimation."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from scipy import optimize

from riskguard.models import DrawdownPlan, DrawdownScheduleTier, RiskOfRuinResult

DEFAULT_STEP_DOWN_TIERS: list[tuple[float, float, float, str]] = [
    (0.0, 3.0, 1.00, "Normal Risk"),
    (3.0, 6.0, 0.75, "Caution — Step Down 25%"),
    (6.0, 10.0, 0.50, "Defensive — Half Risk"),
    (10.0, 15.0, 0.25, "Preservation — Quarter Risk"),
    (15.0, 100.0, 0.00, "HALT — Circuit Breaker"),
]


def _step_down_multiplier(dd_pct: float, max_dd_limit_pct: float) -> float:
    if dd_pct >= max_dd_limit_pct:
        return 0.0
    scale = max_dd_limit_pct / 15.0
    for min_d, max_d, mult, _ in DEFAULT_STEP_DOWN_TIERS:
        if min_d * scale <= dd_pct < max_d * scale:
            return mult
    return 0.0


def compute_drawdown_plan(
    *,
    current_equity: float,
    peak_equity: float | None = None,
    current_dd_pct: float | None = None,
    base_risk_pct: float = 1.0,
    max_dd_limit_pct: float = 15.0,
    cppi_leverage: float = 1.25,
) -> DrawdownPlan:
    """Compute step-down risk schedule and CPPI-style drawdown risk multiplier."""
    eq = max(1e-9, float(current_equity))
    if current_dd_pct is not None:
        dd_pct = max(0.0, float(current_dd_pct))
        peak = eq / max(1e-9, 1.0 - dd_pct / 100.0)
    else:
        peak = max(eq, float(peak_equity or eq))
        dd_pct = max(0.0, (peak - eq) / peak * 100.0)

    floor = peak * max(0.0, 1.0 - max_dd_limit_pct / 100.0)
    cushion_usd = max(0.0, eq - floor)
    max_cushion_usd = max(1e-9, peak - floor)
    cushion_ratio = cushion_usd / max_cushion_usd

    step_mult = _step_down_multiplier(dd_pct, max_dd_limit_pct)
    step_risk = base_risk_pct * step_mult

    cppi_mult = min(1.0, max(0.0, cppi_leverage * cushion_ratio)) if dd_pct < max_dd_limit_pct else 0.0
    cppi_risk = base_risk_pct * cppi_mult

    halted = dd_pct >= max_dd_limit_pct or step_mult == 0.0
    rec_risk = 0.0 if halted else min(step_risk, cppi_risk)

    scale = max_dd_limit_pct / 15.0
    schedule_table: list[DrawdownScheduleTier] = []
    for min_d, max_d, mult, label in DEFAULT_STEP_DOWN_TIERS:
        s_min = round(min_d * scale, 2)
        s_max = round(min(100.0, max_d * scale), 2) if max_d < 100.0 else 100.0
        schedule_table.append(
            DrawdownScheduleTier(
                min_dd_pct=s_min,
                max_dd_pct=s_max,
                risk_multiplier=mult,
                effective_risk_pct=round(base_risk_pct * mult, 4),
                status=label,
            )
        )

    return DrawdownPlan(
        base_risk_pct=float(base_risk_pct),
        current_dd_pct=float(dd_pct),
        max_dd_limit_pct=float(max_dd_limit_pct),
        step_down_multiplier=float(step_mult),
        step_down_risk_pct=float(step_risk),
        cppi_multiplier=float(cppi_mult),
        cppi_risk_pct=float(cppi_risk),
        cppi_cushion_usd=float(cushion_usd),
        recommended_risk_pct=float(rec_risk),
        trading_halted=halted,
        schedule_table=schedule_table,
    )


def analytical_risk_of_ruin(
    win_rate: float,
    payoff_ratio: float,
    risk_per_trade_pct: float,
    ruin_drawdown_pct: float = 25.0,
) -> float:
    """Probability of ever losing ruin_drawdown_pct of STARTING equity (infinite horizon).

    Models each trade as +b R (prob p) or -1 R. Solves p * u^b + (1 - p) * u^(-1) = 1 for the root
    u in (0, 1) when expectancy > 0, and returns u^N with N = ln(1 - ruin) / ln(1 - f) compounding
    loss units (f = risk per trade as a fraction). Returns 1.0 when expectancy <= 0.
    """
    p = max(0.0, min(1.0, float(win_rate)))
    b = max(1e-6, float(payoff_ratio))
    f = max(1e-6, float(risk_per_trade_pct) / 100.0)
    ruin_frac = max(1e-6, min(0.999, float(ruin_drawdown_pct) / 100.0))

    expectancy_r = p * b - (1.0 - p)
    if expectancy_r <= 0.0 or p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 0.0

    # Compounding risk units to hit ruin drawdown
    n_units = math.log(1.0 - ruin_frac) / math.log(1.0 - f) if f < 0.99 else 1.0
    n_units = max(1.0, n_units)

    if abs(b - 1.0) < 1e-9:
        u = (1.0 - p) / p
    else:
        def _mgf(u: float) -> float:
            return p * (u ** b) + (1.0 - p) / u - 1.0

        # As u -> 0+, _mgf(u) -> +inf; at u -> 1-, derivative is p*b - (1-p) > 0 so _mgf(1-eps) < 0.
        # For a vanishingly small edge the root sits numerically at 1 (ruin ~ certain).
        hi = 1.0 - 1e-9
        if _mgf(hi) >= 0.0:
            return 1.0
        u = float(optimize.brentq(_mgf, 1e-12, hi))

    prob = float(u ** n_units)
    return min(1.0, max(0.0, prob))


def estimate_risk_of_ruin(
    *,
    win_rate: float,
    payoff_ratio: float,
    risk_per_trade_pct: float = 1.0,
    ruin_drawdown_pct: float = 25.0,
    max_dd_limit_pct: float = 15.0,
    r_multiples: Sequence[float] | None = None,
    starting_equity: float = 10_000.0,
    simulations: int = 3000,
    horizon_trades: int = 250,
    seed: int = 42,
) -> RiskOfRuinResult:
    """Estimate analytical and Monte Carlo Risk of Ruin (flat vs drawdown-scheduled risk).

    Ruin has the same definition in every estimate: equity ever falls to or below
    starting_equity * (1 - ruin_drawdown_pct / 100). The analytical value is infinite-horizon;
    the Monte Carlo values cover `horizon_trades` trades. The worst peak-to-trough drawdown of the
    flat-risk paths is reported separately (mc_p95_max_dd_pct).
    """
    p = max(0.0, min(1.0, float(win_rate)))
    b = max(1e-6, float(payoff_ratio))
    expectancy_r = p * b - (1.0 - p)
    anal_prob = analytical_risk_of_ruin(
        win_rate=p,
        payoff_ratio=b,
        risk_per_trade_pct=risk_per_trade_pct,
        ruin_drawdown_pct=ruin_drawdown_pct,
    )

    rng = np.random.default_rng(seed)
    r_pool = np.asarray(list(r_multiples or []), dtype=float)
    r_pool = r_pool[np.isfinite(r_pool)]
    if r_pool.size >= 10:
        # Bootstrap the trader's own realised R-multiples
        draws = rng.choice(r_pool, size=(simulations, horizon_trades), replace=True)
        emp_expectancy_r = float(np.mean(r_pool))
    else:
        bern = rng.random((simulations, horizon_trades)) < p
        draws = np.where(bern, b, -1.0)
        emp_expectancy_r = expectancy_r

    base_f = max(1e-6, float(risk_per_trade_pct) / 100.0)
    ruin_thresh = 1.0 - float(ruin_drawdown_pct) / 100.0

    # Vectorized flat-risk simulation
    flat_rets = np.clip(1.0 + base_f * draws, 0.0, None)
    flat_paths = starting_equity * np.cumprod(flat_rets, axis=1)
    flat_peaks = np.maximum.accumulate(np.hstack([np.full((simulations, 1), starting_equity), flat_paths]), axis=1)[:, 1:]
    flat_dd_frac = 1.0 - flat_paths / np.maximum(1e-9, flat_peaks)
    flat_max_dd = np.max(flat_dd_frac, axis=1)
    # Ruin: equity touches starting_equity * (1 - ruin%) at any point (same definition as analytical)
    flat_ruined = np.any(flat_paths <= starting_equity * ruin_thresh, axis=1)

    # Vectorized drawdown-scheduled simulation
    sched_eq = np.full(simulations, float(starting_equity), dtype=float)
    sched_peak = sched_eq.copy()
    sched_ruined = np.zeros(simulations, dtype=bool)
    max_lim = max(1e-6, float(max_dd_limit_pct))

    for step in range(horizon_trades):
        dd_pct = np.maximum(0.0, (sched_peak - sched_eq) / np.maximum(1e-9, sched_peak) * 100.0)
        # Step-down schedule vectorized
        mult = np.where(
            dd_pct >= max_lim,
            0.0,
            np.where(
                dd_pct >= max_lim * (10.0 / 15.0),
                0.25,
                np.where(
                    dd_pct >= max_lim * (6.0 / 15.0),
                    0.50,
                    np.where(dd_pct >= max_lim * (3.0 / 15.0), 0.75, 1.0),
                ),
            ),
        )
        step_f = base_f * mult
        sched_eq = np.maximum(0.0, sched_eq * (1.0 + step_f * draws[:, step]))
        sched_peak = np.maximum(sched_peak, sched_eq)
        sched_ruined |= sched_eq <= starting_equity * ruin_thresh

    return RiskOfRuinResult(
        ruin_drawdown_pct=float(ruin_drawdown_pct),
        risk_per_trade_pct=float(risk_per_trade_pct),
        win_rate=float(p),
        payoff_ratio=float(b),
        expectancy_r=float(emp_expectancy_r),
        analytical_ruin_prob=float(anal_prob),
        mc_ruin_prob_flat=float(np.mean(flat_ruined)),
        mc_ruin_prob_scheduled=float(np.mean(sched_ruined)),
        mc_median_terminal_equity=float(np.median(flat_paths[:, -1])),
        mc_p05_terminal_equity=float(np.quantile(flat_paths[:, -1], 0.05)),
        mc_p95_max_dd_pct=float(np.quantile(flat_max_dd * 100.0, 0.95)),
        simulations=int(simulations),
        horizon_trades=int(horizon_trades),
    )
