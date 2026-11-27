"""Configurable risk rules engine and daily/weekly/drawdown circuit breaker."""

from __future__ import annotations

import math
from typing import Sequence

from riskguard.models import (
    AccountState,
    CircuitBreakerStatus,
    JournalAnalytics,
    PortfolioSummary,
    PositionAnalysis,
    RuleConfig,
    RuleEvaluation,
    RuleStatus,
)


def _tol(limit: float) -> float:
    """Comparison tolerance so boundaries are not decided by float noise (e.g. 3.0 * 0.8 = 2.4000000000000004)."""
    return 1e-9 * max(1.0, abs(float(limit)))


def reaches(value: float, threshold: float) -> bool:
    return value >= threshold - _tol(threshold)


def exceeds(value: float, threshold: float) -> bool:
    return value > threshold + _tol(threshold)


def _status_from_ratio(value: float, limit: float, warn_fraction: float, *, inclusive: bool = False) -> RuleStatus:
    """OK / WARN / BREACH for a 'max' style limit.

    Caps (risk per trade, leverage, positions, exposure) may be used up to the limit: BREACH only when
    value > limit. Loss limits (daily, weekly, drawdown) are hit when the loss reaches the limit:
    `inclusive=True` gives BREACH at value >= limit, matching the circuit breaker.
    WARN from limit * warn_fraction.
    """
    if not math.isfinite(value):
        return "BREACH"
    if exceeds(value, limit) or (inclusive and reaches(value, limit)):
        return "BREACH"
    if reaches(value, limit * warn_fraction):
        return "WARN"
    return "OK"


def evaluate_circuit_breaker(
    account: AccountState,
    rules: RuleConfig,
    journal: JournalAnalytics | None = None,
) -> CircuitBreakerStatus:
    """Determine whether trading is allowed today from daily/weekly PnL and drawdown."""
    eq = float(account.equity)
    daily_pnl = float(journal.today_pnl_usd) if journal is not None else 0.0
    weekly_pnl = float(journal.week_pnl_usd) if journal is not None else 0.0
    current_dd = float(journal.current_drawdown_pct) if journal is not None else 0.0

    # Loss limits are measured against the equity at the START of the day / week (as prop firms and most
    # risk desks do): start equity = current equity minus the PnL realised since then.
    day_start_eq = eq - daily_pnl
    week_start_eq = eq - weekly_pnl
    daily_pnl_pct = (daily_pnl / day_start_eq * 100.0) if day_start_eq > 0 else 0.0
    weekly_pnl_pct = (weekly_pnl / week_start_eq * 100.0) if week_start_eq > 0 else 0.0
    daily_loss_pct = max(0.0, -daily_pnl_pct)
    weekly_loss_pct = max(0.0, -weekly_pnl_pct)

    reasons: list[str] = []
    tripped = False
    caution = False

    if reaches(daily_loss_pct, rules.daily_loss_limit_pct):
        tripped = True
        reasons.append(
            f"Daily loss limit breached ({daily_loss_pct:.2f}% >= {rules.daily_loss_limit_pct:.2f}%)."
        )
    elif reaches(daily_loss_pct, rules.daily_loss_limit_pct * rules.warn_fraction):
        caution = True
        reasons.append(
            f"Approaching daily loss limit ({daily_loss_pct:.2f}% vs {rules.daily_loss_limit_pct:.2f}%)."
        )

    if reaches(weekly_loss_pct, rules.weekly_loss_limit_pct):
        tripped = True
        reasons.append(
            f"Weekly loss limit breached ({weekly_loss_pct:.2f}% >= {rules.weekly_loss_limit_pct:.2f}%)."
        )
    elif reaches(weekly_loss_pct, rules.weekly_loss_limit_pct * rules.warn_fraction):
        caution = True
        reasons.append(
            f"Approaching weekly loss limit ({weekly_loss_pct:.2f}% vs {rules.weekly_loss_limit_pct:.2f}%)."
        )

    if reaches(current_dd, rules.max_drawdown_pct):
        tripped = True
        reasons.append(
            f"Max drawdown limit breached ({current_dd:.2f}% >= {rules.max_drawdown_pct:.2f}%)."
        )
    elif reaches(current_dd, rules.max_drawdown_pct * rules.warn_fraction):
        caution = True
        reasons.append(
            f"Drawdown near circuit breaker ({current_dd:.2f}% vs {rules.max_drawdown_pct:.2f}%)."
        )

    if tripped:
        status = "TRIPPED"
        allowed = False
    elif caution:
        status = "CAUTION"
        allowed = True
    else:
        status = "OPEN"
        allowed = True
        reasons.append("Daily loss, weekly loss, and drawdown are all within limits.")

    return CircuitBreakerStatus(
        trading_allowed=allowed,
        status=status,
        daily_pnl_usd=daily_pnl,
        daily_pnl_pct=daily_pnl_pct,
        weekly_pnl_usd=weekly_pnl,
        weekly_pnl_pct=weekly_pnl_pct,
        current_drawdown_pct=current_dd,
        reference_date=journal.reference_date if journal is not None else None,
        reasons=reasons,
    )


def _max_correlated_cluster_exposure_pct(
    analyses: Sequence[PositionAnalysis],
    portfolio: PortfolioSummary,
    rules: RuleConfig,
) -> tuple[float, list[str]]:
    """Compute highest same-direction correlated cluster exposure as % of equity."""
    if not analyses or portfolio.equity <= 0:
        return 0.0, []

    corr_map = portfolio.corr_30d or portfolio.corr_90d
    # Signed net exposure per symbol
    net_by_sym: dict[str, float] = {}
    for a in analyses:
        sign = 1.0 if a.side == "long" else -1.0
        net_by_sym[a.symbol] = net_by_sym.get(a.symbol, 0.0) + sign * a.notional

    syms = list(net_by_sym.keys())
    if len(syms) == 1:
        s0 = syms[0]
        return abs(net_by_sym[s0]) / portfolio.equity * 100.0, [s0]

    best_cluster_usd = 0.0
    best_syms: list[str] = []
    for anchor in syms:
        anchor_net = net_by_sym[anchor]
        if abs(anchor_net) <= 1e-9:
            continue
        anchor_sign = 1.0 if anchor_net > 0 else -1.0
        cluster_usd = abs(anchor_net)
        cluster_syms = [anchor]
        for other in syms:
            if other == anchor:
                continue
            other_net = net_by_sym[other]
            if abs(other_net) <= 1e-9:
                continue
            other_sign = 1.0 if other_net > 0 else -1.0
            pair_corr = float(corr_map.get(anchor, {}).get(other, 0.0))
            # Same directional risk if corr >= thresh and same sign, OR corr <= -thresh and opposite sign
            directional_corr = pair_corr * anchor_sign * other_sign
            if directional_corr >= rules.correlation_threshold:
                cluster_usd += abs(other_net)
                cluster_syms.append(other)
        if cluster_usd > best_cluster_usd:
            best_cluster_usd = cluster_usd
            best_syms = cluster_syms

    return best_cluster_usd / portfolio.equity * 100.0, best_syms


def evaluate_rules(
    analyses: Sequence[PositionAnalysis],
    portfolio: PortfolioSummary,
    account: AccountState,
    rules: RuleConfig,
    journal: JournalAnalytics | None = None,
) -> tuple[list[RuleEvaluation], CircuitBreakerStatus]:
    """Evaluate all configured portfolio, position, and account-level risk rules."""
    cb = evaluate_circuit_breaker(account, rules, journal=journal)
    evals: list[RuleEvaluation] = []
    wf = rules.warn_fraction

    # 1. Mandatory stop-loss (a stop already crossed by the mark is not protective either)
    missing_syms = [a.symbol for a in analyses if a.missing_stop]
    crossed_syms = [a.symbol for a in analyses if a.invalid_stop_direction]
    no_stop_syms = missing_syms + crossed_syms
    unprotected_detail = "; ".join(
        part
        for part in (
            f"no stop: {', '.join(missing_syms)}" if missing_syms else "",
            f"stop already crossed by mark: {', '.join(crossed_syms)}" if crossed_syms else "",
        )
        if part
    )
    if rules.mandatory_stop:
        if no_stop_syms:
            evals.append(
                RuleEvaluation(
                    rule_id="mandatory_stop",
                    name="Mandatory Stop-Loss",
                    status="BREACH",
                    current_value=f"{len(no_stop_syms)} unprotected",
                    limit_value="0 unprotected",
                    explanation=f"Positions without a protective stop ({unprotected_detail}).",
                    affected_symbols=no_stop_syms,
                )
            )
        else:
            evals.append(
                RuleEvaluation(
                    rule_id="mandatory_stop",
                    name="Mandatory Stop-Loss",
                    status="OK",
                    current_value="All protected",
                    limit_value="Required",
                    explanation="Every open position has a stop-loss defined.",
                )
            )
    else:
        evals.append(
            RuleEvaluation(
                rule_id="mandatory_stop",
                name="Mandatory Stop-Loss",
                status="WARN" if no_stop_syms else "OK",
                current_value=f"{len(no_stop_syms)} unprotected",
                limit_value="Optional",
                explanation=(
                    "Mandatory stop rule is disabled, though unstopped exposure carries tail risk"
                    + (f" ({unprotected_detail})." if unprotected_detail else ".")
                ),
                affected_symbols=no_stop_syms,
            )
        )

    # 2. Liquidation vs Stop safety
    bad_liq_syms = [
        a.symbol for a in analyses if a.stop_beyond_liquidation or a.at_or_past_liquidation or a.liquidated
    ]
    near_liq_syms = [
        a.symbol
        for a in analyses
        if a.distance_to_liq_atr is not None and 0 < a.distance_to_liq_atr < 1.5 and a.symbol not in bad_liq_syms
    ]
    if bad_liq_syms:
        evals.append(
            RuleEvaluation(
                rule_id="liquidation_safety",
                name="Stop vs Liquidation Safety",
                status="BREACH",
                current_value=f"{len(bad_liq_syms)} critical",
                limit_value="Stop before LP",
                explanation=(
                    f"{', '.join(bad_liq_syms)} will liquidate before the stop-loss triggers "
                    "(or is already past liquidation)."
                ),
                affected_symbols=bad_liq_syms,
            )
        )
    elif near_liq_syms:
        evals.append(
            RuleEvaluation(
                rule_id="liquidation_safety",
                name="Stop vs Liquidation Safety",
                status="WARN",
                current_value=f"{len(near_liq_syms)} tight (<1.5 ATR)",
                limit_value=">= 1.5 ATR buffer",
                explanation=f"{', '.join(near_liq_syms)} liquidation is within 1.5 daily ATRs of mark.",
                affected_symbols=near_liq_syms,
            )
        )
    else:
        evals.append(
            RuleEvaluation(
                rule_id="liquidation_safety",
                name="Stop vs Liquidation Safety",
                status="OK",
                current_value="Safe buffer",
                limit_value="Stop before LP",
                explanation="All stopped positions have liquidation prices safely beyond their stops.",
            )
        )

    # 3. Max risk per trade %
    risk_vals = [
        (a.symbol, a.risk_to_stop_pct)
        for a in analyses
        if a.risk_to_stop_pct is not None
    ]
    max_trade_risk_sym, max_trade_risk = (
        max(risk_vals, key=lambda x: x[1]) if risk_vals else ("", 0.0)
    )
    breach_risk_syms = [s for s, r in risk_vals if _status_from_ratio(r, rules.max_risk_per_trade_pct, wf) == "BREACH"]
    warn_risk_syms = [s for s, r in risk_vals if _status_from_ratio(r, rules.max_risk_per_trade_pct, wf) == "WARN"]
    trade_risk_status = _status_from_ratio(max_trade_risk, rules.max_risk_per_trade_pct, wf)
    if no_stop_syms and trade_risk_status == "OK":
        trade_risk_status = "WARN"
    evals.append(
        RuleEvaluation(
            rule_id="max_risk_per_trade_pct",
            name="Max Risk Per Trade",
            status=trade_risk_status,
            current_value=f"{max_trade_risk:.2f}% ({max_trade_risk_sym or 'none'})",
            limit_value=f"{rules.max_risk_per_trade_pct:.2f}%",
            explanation=(
                f"Largest defined stop risk is {max_trade_risk:.2f}% of equity"
                + (f" on {max_trade_risk_sym}." if max_trade_risk_sym else ".")
            ),
            affected_symbols=breach_risk_syms or warn_risk_syms,
        )
    )

    # 4. Max total open risk %
    total_risk_pct = portfolio.known_stop_risk_pct
    total_risk_status = _status_from_ratio(total_risk_pct, rules.max_total_open_risk_pct, wf)
    if portfolio.unstopped_positions > 0 and total_risk_status != "BREACH":
        # Unstopped positions make true open risk strictly higher than known stop risk
        if exceeds(portfolio.worst_case_stops_pct, rules.max_total_open_risk_pct):
            total_risk_status = "BREACH"
        else:
            total_risk_status = "WARN"
    evals.append(
        RuleEvaluation(
            rule_id="max_total_open_risk_pct",
            name="Max Total Open Risk",
            status=total_risk_status,
            current_value=f"{total_risk_pct:.2f}% (worst-case {portfolio.worst_case_stops_pct:.2f}%)",
            limit_value=f"{rules.max_total_open_risk_pct:.2f}%",
            explanation=(
                f"Known stop risk is {total_risk_pct:.2f}% across stopped positions; "
                f"{portfolio.unstopped_positions} unstopped position(s) raise worst-case risk to {portfolio.worst_case_stops_pct:.2f}%."
            ),
            affected_symbols=no_stop_syms,
        )
    )

    # 5. Max leverage
    max_pos_lev = max((a.leverage for a in analyses), default=0.0)
    liquidated_syms = [a.symbol for a in analyses if a.liquidated]
    lev_breach_syms = list(
        dict.fromkeys(
            liquidated_syms
            + [a.symbol for a in analyses if _status_from_ratio(a.leverage, rules.max_leverage, wf) == "BREACH"]
        )
    )
    lev_warn_syms = [
        a.symbol
        for a in analyses
        if a.symbol not in lev_breach_syms and _status_from_ratio(a.leverage, rules.max_leverage, wf) == "WARN"
    ]
    lev_status: RuleStatus = "BREACH" if liquidated_syms else _status_from_ratio(max_pos_lev, rules.max_leverage, wf)
    if liquidated_syms:
        lev_cur = f"LIQUIDATED ({len(liquidated_syms)}) / {max_pos_lev:.1f}x (gross {portfolio.gross_leverage:.2f}x)"
        lev_exp = (
            f"Liquidated position(s): {', '.join(liquidated_syms)} (effective leverage NaN). "
            f"Highest position leverage is {max_pos_lev:.1f}x vs {rules.max_leverage:.1f}x limit."
        )
    else:
        lev_cur = f"{max_pos_lev:.1f}x (gross {portfolio.gross_leverage:.2f}x)"
        lev_exp = f"Highest position leverage is {max_pos_lev:.1f}x vs {rules.max_leverage:.1f}x limit."
    evals.append(
        RuleEvaluation(
            rule_id="max_leverage",
            name="Max Position Leverage",
            status=lev_status,
            current_value=lev_cur,
            limit_value=f"{rules.max_leverage:.1f}x",
            explanation=lev_exp,
            affected_symbols=lev_breach_syms or lev_warn_syms,
        )
    )

    # 6. Max positions
    pos_count = len(analyses)
    pos_status = _status_from_ratio(float(pos_count), float(rules.max_positions), wf)
    evals.append(
        RuleEvaluation(
            rule_id="max_positions",
            name="Max Open Positions",
            status=pos_status,
            current_value=str(pos_count),
            limit_value=str(rules.max_positions),
            explanation=f"{pos_count} open position(s) vs max {rules.max_positions}.",
        )
    )

    # 7. Max correlated exposure %
    corr_exp_pct, corr_syms = _max_correlated_cluster_exposure_pct(analyses, portfolio, rules)
    corr_status = _status_from_ratio(corr_exp_pct, rules.max_correlated_exposure_pct, wf)
    evals.append(
        RuleEvaluation(
            rule_id="max_correlated_exposure_pct",
            name="Max Correlated Cluster Exposure",
            status=corr_status,
            current_value=f"{corr_exp_pct:.1f}%",
            limit_value=f"{rules.max_correlated_exposure_pct:.1f}% (r>={rules.correlation_threshold:.2f})",
            explanation=(
                f"Largest same-direction correlated cluster ({', '.join(corr_syms) or 'none'}) "
                f"is {corr_exp_pct:.1f}% of equity."
            ),
            affected_symbols=corr_syms if corr_status != "OK" else [],
        )
    )

    # 8. Daily loss limit %
    daily_loss_pct = max(0.0, -cb.daily_pnl_pct)
    evals.append(
        RuleEvaluation(
            rule_id="daily_loss_limit_pct",
            name="Daily Loss Limit",
            status=_status_from_ratio(daily_loss_pct, rules.daily_loss_limit_pct, wf, inclusive=True),
            current_value=f"{cb.daily_pnl_pct:+.2f}%",
            limit_value=f"-{rules.daily_loss_limit_pct:.2f}%",
            explanation=f"Session PnL is ${cb.daily_pnl_usd:+,.2f} ({cb.daily_pnl_pct:+.2f}% of equity).",
        )
    )

    # 9. Weekly loss limit %
    weekly_loss_pct = max(0.0, -cb.weekly_pnl_pct)
    evals.append(
        RuleEvaluation(
            rule_id="weekly_loss_limit_pct",
            name="Weekly Loss Limit",
            status=_status_from_ratio(weekly_loss_pct, rules.weekly_loss_limit_pct, wf, inclusive=True),
            current_value=f"{cb.weekly_pnl_pct:+.2f}%",
            limit_value=f"-{rules.weekly_loss_limit_pct:.2f}%",
            explanation=f"7-day / weekly PnL is ${cb.weekly_pnl_usd:+,.2f} ({cb.weekly_pnl_pct:+.2f}% of equity).",
        )
    )

    # 10. Max drawdown %
    evals.append(
        RuleEvaluation(
            rule_id="max_drawdown_pct",
            name="Max Account Drawdown",
            status=_status_from_ratio(cb.current_drawdown_pct, rules.max_drawdown_pct, wf, inclusive=True),
            current_value=f"{cb.current_drawdown_pct:.2f}%",
            limit_value=f"{rules.max_drawdown_pct:.2f}%",
            explanation=f"Current journal drawdown is {cb.current_drawdown_pct:.2f}% vs {rules.max_drawdown_pct:.2f}% limit.",
        )
    )

    return evals, cb
