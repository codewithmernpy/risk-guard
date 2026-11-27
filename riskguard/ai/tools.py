"""@beta_tool closures over current RiskGuard session state (strictly < 3000 chars each)."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

from anthropic import beta_tool

from riskguard.analytics.drawdown import compute_drawdown_plan, estimate_risk_of_ruin
from riskguard.analytics.positions import effective_leverage_label, liquidation_label
from riskguard.analytics.sizing import pre_trade_check, recommend_sizes
from riskguard.data.providers import MarketDataBundle
from riskguard.models import (
    AccountState,
    CircuitBreakerStatus,
    DrawdownPlan,
    JournalAnalytics,
    PortfolioSummary,
    PositionAnalysis,
    PositionInput,
    RiskOfRuinResult,
    RuleConfig,
    RuleEvaluation,
    normalize_side,
    normalize_symbol,
)


@dataclass
class CoachContext:
    """Immutable snapshot of current session analytics passed into AI tool closures."""

    account: AccountState
    positions: list[PositionInput]
    analyses: list[PositionAnalysis]
    portfolio: PortfolioSummary
    rules: RuleConfig
    rule_evals: list[RuleEvaluation]
    circuit_breaker: CircuitBreakerStatus
    journal: JournalAnalytics | None = None
    market: MarketDataBundle | None = None
    drawdown_plan: DrawdownPlan | None = None
    risk_of_ruin: RiskOfRuinResult | None = None
    k_atr_buffer: float = 1.5


def _dumps(obj: Any) -> str:
    return json.dumps(obj, separators=(",", ":"), default=str)


def _shrink(obj: Any, max_list: int, max_str: int) -> Any:
    """Recursively cap list lengths and string lengths (marks dropped list items)."""
    if isinstance(obj, dict):
        return {k: _shrink(v, max_list, max_str) for k, v in obj.items()}
    if isinstance(obj, list):
        items = [_shrink(v, max_list, max_str) for v in obj[:max_list]]
        if len(obj) > max_list:
            items.append(f"... {len(obj) - max_list} more omitted")
        return items
    if isinstance(obj, str) and len(obj) > max_str:
        return obj[: max_str - 3] + "..."
    return obj


def _bounded_json(payload: dict[str, Any], max_chars: int = 2950) -> str:
    """Serialize payload to compact JSON that is ALWAYS valid JSON and at most max_chars long.

    Oversized payloads are shrunk structurally (shorter lists / strings, flagged with "truncated": true),
    never by cutting the JSON text, so the model always receives parseable output.
    """
    text = _dumps(payload)
    if len(text) <= max_chars:
        return text
    for max_list, max_str in ((12, 240), (8, 160), (5, 120), (3, 80), (2, 60), (1, 40)):
        shrunk = _shrink(payload, max_list, max_str)
        shrunk["truncated"] = True
        text = _dumps(shrunk)
        if len(text) <= max_chars:
            return text
    # Last resort: keep only scalar top-level fields
    scalars = {k: v for k, v in payload.items() if isinstance(v, (int, float, bool, type(None)))}
    scalars["truncated"] = True
    text = _dumps(scalars)
    if len(text) <= max_chars:
        return text
    return _dumps({"truncated": True, "error": "tool output too large to summarise"})


def build_coach_tools(ctx: CoachContext) -> list[Any]:
    """Create the 8 @beta_tool functions bound to the provided CoachContext."""

    @beta_tool
    def get_portfolio_risk() -> str:
        """Return compact portfolio exposure, VaR/CVaR, concentration, beta, and top dangers."""
        p = ctx.portfolio
        dangers = []
        for a in ctx.analyses:
            if a.at_or_past_liquidation or a.liquidated:
                dangers.append(f"{a.symbol}:AT_OR_PAST_LIQ(mark={a.mark_price},lp={a.liquidation_price})")
            elif a.stop_beyond_liquidation:
                dangers.append(f"{a.symbol}:STOP_BEYOND_LIQ(stop={a.stop_price},lp={a.liquidation_price})")
            elif a.invalid_stop_direction:
                dangers.append(f"{a.symbol}:STOP_ALREADY_CROSSED(stop={a.stop_price},mark={a.mark_price})")
            elif a.missing_stop:
                dangers.append(f"{a.symbol}:MISSING_STOP")
        compact = {
            "equity": round(p.equity, 2),
            "positions": p.position_count,
            "gross_usd": round(p.gross_exposure_usd, 2),
            "net_usd": round(p.net_exposure_usd, 2),
            "gross_lev": round(p.gross_leverage, 2),
            "btc_beta_net_usd": round(p.btc_beta_adj_net_usd, 2),
            "btc_beta_net_pct": round(p.btc_beta_adj_net_pct, 1),
            "upnl_usd": round(p.total_unrealized_pnl_usd, 2),
            "daily_funding_usd": round(p.total_daily_funding_cost_usd, 2),
            "known_stop_risk_pct": round(p.known_stop_risk_pct, 2),
            "unstopped_count": p.unstopped_positions,
            "worst_case_stops_usd": round(p.worst_case_stops_usd, 2),
            "hhi": round(p.hhi, 3),
            "max_corr_30d": round(p.max_pairwise_corr_30d, 3),
            "var95_hist": round(p.var_95_hist_usd, 2),
            "var99_hist": round(p.var_99_hist_usd, 2),
            "cvar95_hist": round(p.cvar_95_hist_usd, 2),
            "var95_cf": round(p.var_95_cf_usd, 2),
            "var99_cf": round(p.var_99_cf_usd, 2),
            "betas": p.betas_vs_btc,
            "dangers": dangers,
        }
        return _bounded_json(compact)

    @beta_tool
    def get_position_details(symbol: str) -> str:
        """Return detailed risk, stop, funding, and liquidation metrics for a symbol (or 'ALL')."""
        target = symbol.strip().upper()
        if target in ("ALL", "*"):
            items = [
                {
                    "sym": a.symbol,
                    "side": a.side,
                    "mode": a.margin_mode,
                    "lev": a.leverage,
                    "notional": round(a.notional, 2),
                    "entry": a.entry_price,
                    "mark": a.mark_price,
                    "stop": a.stop_price,
                    "lp": round(a.liquidation_price, 6) if a.liquidation_price is not None else liquidation_label(a),
                    "dist_lp_pct": round(a.distance_to_liq_pct, 2) if (a.distance_to_liq_pct is not None and math.isfinite(a.distance_to_liq_pct)) else None,
                    "dist_lp_atr": round(a.distance_to_liq_atr, 2) if (a.distance_to_liq_atr is not None and math.isfinite(a.distance_to_liq_atr)) else None,
                    "risk_pct": round(a.risk_to_stop_pct, 2) if a.risk_to_stop_pct is not None else None,
                    "eff_lev": effective_leverage_label(a),
                    "liquidated": a.liquidated,
                    "margin_tier": "_DEFAULT (default)" if a.margin_tier_source == "_DEFAULT" else (a.margin_tier_source or "_DEFAULT (default)"),
                    "entry_proxy": a.used_entry_proxy,
                    "danger": a.danger_level,
                    "flags": a.warnings[:2],
                }
                for a in ctx.analyses
            ]
            return _bounded_json({"positions": items})

        norm = normalize_symbol(target)
        matches = [a for a in ctx.analyses if a.symbol == norm]
        if not matches:
            return _bounded_json({"error": f"No open position found for {norm}", "open_symbols": [a.symbol for a in ctx.analyses]})
        a = matches[0]
        compact = {
            "symbol": a.symbol,
            "side": a.side,
            "margin_mode": a.margin_mode,
            "quantity": a.quantity,
            "entry_price": a.entry_price,
            "mark_price": a.mark_price,
            "notional_usd": round(a.notional, 2),
            "margin_used_usd": round(a.margin_used, 2),
            "leverage": a.leverage,
            "effective_leverage": "LIQUIDATED" if (a.liquidated or not math.isfinite(a.effective_leverage)) else round(a.effective_leverage, 2),
            "liquidated": a.liquidated,
            "unrealized_pnl_usd": round(a.unrealized_pnl, 2),
            "stop_price": a.stop_price,
            "take_profit": a.take_profit,
            "risk_to_stop_usd": round(a.risk_to_stop_usd, 2) if a.risk_to_stop_usd is not None else None,
            "risk_to_stop_pct": round(a.risk_to_stop_pct, 2) if a.risk_to_stop_pct is not None else None,
            "reward_to_risk": round(a.reward_to_risk, 2) if a.reward_to_risk is not None else None,
            "funding_daily_usd": round(a.funding_cost_daily_usd, 2),
            "mmr_pct": round(a.mmr * 100.0, 2),
            "margin_tier": "_DEFAULT (default)" if a.margin_tier_source == "_DEFAULT" else (a.margin_tier_source or "_DEFAULT (default)"),
            "used_entry_proxy": a.used_entry_proxy,
            "liquidation_price": round(a.liquidation_price, 6) if a.liquidation_price is not None else None,
            "liquidation_note": a.liquidation_note,
            "dist_to_liq_pct": round(a.distance_to_liq_pct, 2) if (a.distance_to_liq_pct is not None and math.isfinite(a.distance_to_liq_pct)) else None,
            "dist_to_liq_atr": round(a.distance_to_liq_atr, 2) if (a.distance_to_liq_atr is not None and math.isfinite(a.distance_to_liq_atr)) else None,
            "stop_beyond_liquidation": a.stop_beyond_liquidation,
            "stop_already_crossed": a.invalid_stop_direction,
            "missing_stop": a.missing_stop,
            "danger_level": a.danger_level,
            "warnings": a.warnings,
        }
        return _bounded_json(compact)

    @beta_tool
    def run_stress(scenario: str) -> str:
        """Return portfolio PnL and liquidation check for a stress scenario (e.g. 'all', 'BTC -10%', 'alt crash')."""
        q = scenario.strip().lower()
        scenarios = ctx.portfolio.stress_scenarios
        if q not in ("all", "*", ""):
            filtered = [s for s in scenarios if q in s.scenario.lower() or q in s.description.lower()]
            if filtered:
                scenarios = filtered
        compact = {
            "equity_usd": round(ctx.account.equity, 2),
            "scenarios": [
                {
                    "name": s.scenario,
                    "pnl_usd": round(s.portfolio_pnl_usd, 2),
                    "pnl_pct": round(s.portfolio_pnl_pct, 2),
                    "shocked_equity_usd": round(s.shocked_equity_usd, 2),
                    "liquidated": s.liquidated_symbols,
                }
                for s in scenarios
            ],
        }
        return _bounded_json(compact)

    @beta_tool
    def check_rules() -> str:
        """Return rule evaluation statuses (OK/WARN/BREACH) and daily/weekly/DD circuit-breaker state."""
        cb = ctx.circuit_breaker
        compact = {
            "trading_allowed_today": cb.trading_allowed,
            "circuit_breaker": cb.status,
            "daily_pnl_usd": round(cb.daily_pnl_usd, 2),
            "daily_pnl_pct": round(cb.daily_pnl_pct, 2),
            "weekly_pnl_pct": round(cb.weekly_pnl_pct, 2),
            "current_drawdown_pct": round(cb.current_drawdown_pct, 2),
            "cb_reasons": cb.reasons,
            "rules": [
                {
                    "id": r.rule_id,
                    "status": r.status,
                    "current": r.current_value,
                    "limit": r.limit_value,
                    "affected": r.affected_symbols,
                }
                for r in ctx.rule_evals
            ],
        }
        return _bounded_json(compact)

    @beta_tool
    def evaluate_new_trade(trade_json: str) -> str:
        """Evaluate a proposed new trade JSON string with keys symbol, side, entry_price, quantity (or notional), leverage, margin_mode, stop_price."""
        try:
            raw = json.loads(trade_json)
            proposed = PositionInput.model_validate(raw)
        except Exception as exc:
            return _bounded_json({"error": f"Invalid trade_json: {exc}"})

        res = pre_trade_check(
            proposed=proposed,
            existing_positions=ctx.positions,
            account=ctx.account,
            market=ctx.market,
            rules=ctx.rules,
            journal=ctx.journal,
        )
        compact = {
            "verdict": res.verdict,
            "reasons": res.reasons,
            "before_var95": round(res.before_var_95_usd, 2),
            "after_var95": round(res.after_var_95_usd, 2),
            "before_open_risk_pct": round(res.before_open_risk_pct, 2),
            "after_open_risk_pct": round(res.after_open_risk_pct, 2),
            "before_gross_usd": round(res.before_gross_exposure_usd, 2),
            "after_gross_usd": round(res.after_gross_exposure_usd, 2),
            "proposed_lp": (
                round(res.proposed_analysis.liquidation_price, 6)
                if res.proposed_analysis.liquidation_price is not None
                else liquidation_label(res.proposed_analysis)
            ),
            "stop_beyond_liq": res.proposed_analysis.stop_beyond_liquidation,
            "recommended_quantity": round(res.recommended_quantity, 6) if res.recommended_quantity is not None else None,
            "recommended_leverage": res.recommended_leverage,
            "new_breaches": [b for b in res.after_breaches if b not in res.before_breaches],
        }
        return _bounded_json(compact)

    @beta_tool
    def journal_behaviour() -> str:
        """Return journal performance stats, discipline score (0-100), and recent behavioural red flags."""
        if ctx.journal is None or ctx.journal.total_trades == 0:
            return _bounded_json({"status": "No journal trades loaded."})
        j = ctx.journal
        compact = {
            "total_trades": j.total_trades,
            "win_rate": round(j.win_rate * 100.0, 1),
            "profit_factor": round(j.profit_factor, 2),
            "expectancy_usd": round(j.expectancy_usd, 2),
            "avg_r": round(j.avg_r, 3) if j.avg_r is not None else None,
            "total_pnl_usd": round(j.total_pnl_usd, 2),
            "max_dd_pct": round(j.max_drawdown_pct, 2),
            "current_dd_pct": round(j.current_drawdown_pct, 2),
            "disposition_ratio": round(j.disposition_ratio, 2),
            "discipline_score": j.discipline_score,
            "discipline_grade": j.discipline_grade,
            "episode_counts": j.episode_counts,
            "recent_episodes": [
                {"type": e.episode_type, "sym": e.symbol, "sev": e.severity, "desc": e.description[:120]}
                for e in j.episodes[-6:]
            ],
        }
        return _bounded_json(compact)

    @beta_tool
    def size_trade(
        symbol: str,
        side: str,
        entry: float,
        stop: float,
        method: str = "all",
    ) -> str:
        """Recommend position size for symbol/side/entry/stop using fixed_fractional, atr, kelly, max_leverage_safe, or all."""
        norm_sym = normalize_symbol(symbol)
        norm_side = normalize_side(side)
        atr = entry * 0.03
        atr_source = "proxy: 3% of entry (no daily ATR on feed)"
        if ctx.market and norm_sym in ctx.market.snapshots and ctx.market.snapshots[norm_sym].atr_1d > 0:
            atr = ctx.market.snapshots[norm_sym].atr_1d
            atr_source = f"{ctx.market.source} daily ATR(14)"
        win_rate = ctx.journal.win_rate if (ctx.journal and ctx.journal.total_trades >= 10) else 0.50
        payoff = ctx.journal.payoff_ratio if (ctx.journal and ctx.journal.payoff_ratio > 0) else 1.5
        try:
            recs = recommend_sizes(
                symbol=norm_sym,
                side=norm_side,
                entry_price=entry,
                stop_price=stop,
                account=ctx.account,
                atr_1d=atr,
                risk_pct=min(1.0, ctx.rules.max_risk_per_trade_pct),
                win_rate=win_rate,
                payoff_ratio=payoff,
            )
        except ValueError as exc:
            return _bounded_json({"error": str(exc)})

        m = method.strip().lower()
        if m not in ("all", ""):
            filtered = [r for r in recs if m in r.method.lower().replace(" ", "_").replace("-", "_")]
            if filtered:
                recs = filtered
        compact = {
            "symbol": norm_sym,
            "side": norm_side,
            "entry": entry,
            "stop": stop,
            "atr_1d": round(atr, 4),
            "atr_source": atr_source,
            "recommendations": [
                {
                    "method": r.method,
                    "qty": round(r.quantity, 6),
                    "notional_usd": round(r.notional_usd, 2),
                    "risk_usd": round(r.risk_usd, 2),
                    "risk_pct": round(r.risk_pct, 2),
                    "max_safe_lev": r.max_safe_leverage,
                    "notes": r.notes,
                }
                for r in recs
            ],
        }
        return _bounded_json(compact)

    @beta_tool
    def drawdown_plan() -> str:
        """Return drawdown step-down schedule, CPPI risk multiplier, and risk-of-ruin estimates."""
        dd_pct = ctx.journal.current_drawdown_pct if ctx.journal else 0.0
        plan = ctx.drawdown_plan or compute_drawdown_plan(
            current_equity=ctx.account.equity,
            current_dd_pct=dd_pct,
            base_risk_pct=min(1.0, ctx.rules.max_risk_per_trade_pct),
            max_dd_limit_pct=ctx.rules.max_drawdown_pct,
        )
        ror = ctx.risk_of_ruin or estimate_risk_of_ruin(
            win_rate=ctx.journal.win_rate if (ctx.journal and ctx.journal.total_trades >= 5) else 0.50,
            payoff_ratio=ctx.journal.payoff_ratio if (ctx.journal and ctx.journal.payoff_ratio > 0) else 1.5,
            risk_per_trade_pct=plan.recommended_risk_pct or plan.base_risk_pct,
            max_dd_limit_pct=ctx.rules.max_drawdown_pct,
            r_multiples=ctx.journal.r_multiples if ctx.journal else None,
            starting_equity=ctx.account.equity,
        )
        compact = {
            "current_dd_pct": round(plan.current_dd_pct, 2),
            "max_dd_limit_pct": plan.max_dd_limit_pct,
            "step_down_mult": plan.step_down_multiplier,
            "step_down_risk_pct": round(plan.step_down_risk_pct, 3),
            "cppi_mult": round(plan.cppi_multiplier, 3),
            "cppi_risk_pct": round(plan.cppi_risk_pct, 3),
            "recommended_risk_pct": round(plan.recommended_risk_pct, 3),
            "trading_halted": plan.trading_halted,
            "analytical_ruin_prob_pct": round(ror.analytical_ruin_prob * 100.0, 2),
            "mc_ruin_flat_pct": round(ror.mc_ruin_prob_flat * 100.0, 2),
            "mc_ruin_scheduled_pct": round(ror.mc_ruin_prob_scheduled * 100.0, 2),
            "mc_p95_max_dd_pct": round(ror.mc_p95_max_dd_pct, 2),
        }
        return _bounded_json(compact)

    return [
        get_portfolio_risk,
        get_position_details,
        run_stress,
        check_rules,
        evaluate_new_trade,
        journal_behaviour,
        size_trade,
        drawdown_plan,
    ]
