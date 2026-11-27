"""Position sizing models (fixed fractional, ATR, Kelly, max-leverage-safe) and pre-trade check."""

from __future__ import annotations

import math
from typing import Mapping, Sequence

from riskguard.analytics.margin import lookup_margin_tier
from riskguard.analytics.portfolio import analyze_portfolio
from riskguard.analytics.positions import analyze_positions, committed_margin
from riskguard.analytics.rules import evaluate_rules, exceeds
from riskguard.data.providers import MarketDataBundle, SymbolMarketData
from riskguard.models import (
    AccountState,
    JournalAnalytics,
    MarginMode,
    MarginTier,
    PositionInput,
    PreTradeCheckResult,
    RuleConfig,
    Side,
    SizingRecommendation,
    Verdict,
)


def max_safe_leverage_for_stop(
    entry_price: float,
    stop_price: float,
    atr_1d: float,
    side: Side,
    mmr: float,
    k_atr: float = 1.5,
) -> float:
    """Compute maximum isolated leverage keeping LP at least k_atr ATRs beyond the stop.

    Isolated LP ignoring cumB (ignoring cumB is conservative: cumB moves LP further away):
        long:  LP/EP = (1 - 1/L) / (1 - MMR)   ->  L = 1 / (1 - (LP/EP)(1 - MMR))
        short: LP/EP = (1 + 1/L) / (1 + MMR)   ->  L = 1 / ((LP/EP)(1 + MMR) - 1)
    Result is clamped to [1x, 100x].
    """
    ep = float(entry_price)
    sp = float(stop_price)
    atr = max(0.0, float(atr_1d))
    if ep <= 0:
        return 1.0

    if side == "long":
        target_lp = sp - k_atr * atr
        if target_lp <= 0:
            # LP must sit at or below $0: only an unlevered (1x) long has no positive liquidation price
            return 1.0
        denom = 1.0 - (target_lp / ep) * (1.0 - mmr)
        if denom <= 1e-6:
            return 100.0
        return float(max(1.0, min(100.0, 1.0 / denom)))

    target_lp = sp + k_atr * atr
    denom = (target_lp / ep) * (1.0 + mmr) - 1.0
    if denom <= 1e-6:
        return 100.0
    return float(max(1.0, min(100.0, 1.0 / denom)))


def recommend_sizes(
    *,
    symbol: str,
    side: Side,
    entry_price: float,
    stop_price: float,
    account: AccountState,
    atr_1d: float,
    risk_pct: float = 1.0,
    leverage: float = 5.0,
    margin_mode: MarginMode = "isolated",
    k_atr_buffer: float = 1.5,
    kelly_fraction: float = 0.25,
    max_kelly_risk_pct: float = 2.5,
    win_rate: float = 0.50,
    payoff_ratio: float = 1.5,
    other_positions_mm: float = 0.0,
    margin_tiers: Mapping[str, Sequence[MarginTier]] | None = None,
) -> list[SizingRecommendation]:
    """Compute recommended trade quantities under four risk-management sizing methods."""
    ep = float(entry_price)
    sp = float(stop_price)
    eq = float(account.equity)
    atr = max(1e-9, float(atr_1d))

    if ep <= 0 or eq <= 0:
        raise ValueError("entry_price and account.equity must be positive.")
    if side == "long" and sp >= ep:
        raise ValueError("For a long trade, stop_price must be strictly below entry_price.")
    if side == "short" and sp <= ep:
        raise ValueError("For a short trade, stop_price must be strictly above entry_price.")

    stop_dist = abs(ep - sp)

    # 1. Fixed fractional
    ff_risk_usd = eq * (risk_pct / 100.0)
    ff_qty = ff_risk_usd / stop_dist
    ff_notional = ff_qty * ep

    tier = lookup_margin_tier(symbol, ff_notional, margin_tiers=margin_tiers, fallback_mmr=account.default_mmr)

    # 2. ATR volatility-targeted (1 ATR move = risk_pct% of equity)
    atr_risk_budget = eq * (risk_pct / 100.0)
    atr_qty = atr_risk_budget / atr
    atr_notional = atr_qty * ep
    atr_actual_stop_risk = atr_qty * stop_dist

    # 3. Capped fractional Kelly (f* = p - (1-p)/b in R units)
    p = max(0.0, min(1.0, float(win_rate)))
    b = max(1e-6, float(payoff_ratio))
    full_kelly = p - (1.0 - p) / b
    if full_kelly <= 0:
        kelly_risk_pct = 0.0
        kelly_qty = 0.0
        kelly_notes = f"Negative/zero Kelly edge (f*={full_kelly * 100:.2f}%); skip or improve R:R."
    else:
        kelly_risk_pct = min(max_kelly_risk_pct, full_kelly * 100.0 * kelly_fraction)
        kelly_qty = (eq * kelly_risk_pct / 100.0) / stop_dist
        kelly_notes = (
            f"Full Kelly={full_kelly * 100:.2f}%, {kelly_fraction:.2f}x fractional capped at {max_kelly_risk_pct:.1f}%."
        )
    kelly_notional = kelly_qty * ep
    kelly_risk_usd = kelly_qty * stop_dist

    # 4. Max-leverage-safe size (LP >= k_atr_buffer ATRs beyond stop)
    safe_lev = max_safe_leverage_for_stop(
        entry_price=ep,
        stop_price=sp,
        atr_1d=atr,
        side=side,
        mmr=tier.mmr,
        k_atr=k_atr_buffer,
    )
    if margin_mode == "isolated":
        max_safe_notional = eq * safe_lev
        quantity_cap = max_safe_notional / ep
        safe_qty = min(ff_qty, quantity_cap)
        safe_notes = (
            f"Max isolated leverage keeping LP {k_atr_buffer:.1f} ATR beyond stop is {safe_lev:.2f}x"
            + (f" (requested {leverage:.1f}x is unsafe!)." if leverage > safe_lev else ".")
        )
    else:
        wb_avail = max(0.0, eq - other_positions_mm)
        if side == "long":
            target_lp = max(1e-9, sp - k_atr_buffer * atr)
            denom = ep - target_lp * (1.0 - tier.mmr)
        else:
            target_lp = sp + k_atr_buffer * atr
            denom = target_lp * (1.0 + tier.mmr) - ep
        q_cross_cap = ((wb_avail + tier.cum_b) / denom) if denom > 1e-9 else ff_qty
        quantity_cap = max(0.0, q_cross_cap)
        safe_qty = min(ff_qty, quantity_cap)
        safe_notes = (
            f"Cross size capped so LP stays {k_atr_buffer:.1f} ATR beyond stop "
            f"(cross cap={q_cross_cap:.6g}, safe isolated equiv={safe_lev:.2f}x)."
        )
    safe_notional = safe_qty * ep
    safe_risk_usd = safe_qty * stop_dist

    return [
        SizingRecommendation(
            method="Fixed Fractional",
            quantity=float(ff_qty),
            notional_usd=float(ff_notional),
            risk_usd=float(ff_risk_usd),
            risk_pct=float(risk_pct),
            notes=f"Risks {risk_pct:.2f}% (${ff_risk_usd:,.2f}) if stop at ${sp:,.4f} hits.",
        ),
        SizingRecommendation(
            method="ATR Volatility-Targeted",
            quantity=float(atr_qty),
            notional_usd=float(atr_notional),
            risk_usd=float(atr_actual_stop_risk),
            risk_pct=float(atr_actual_stop_risk / eq * 100.0),
            notes=f"Sizes 1 daily ATR (${atr:,.4f}) equal to {risk_pct:.2f}% of equity.",
        ),
        SizingRecommendation(
            method="Capped Fractional Kelly",
            quantity=float(kelly_qty),
            notional_usd=float(kelly_notional),
            risk_usd=float(kelly_risk_usd),
            risk_pct=float(kelly_risk_usd / eq * 100.0),
            notes=kelly_notes,
        ),
        SizingRecommendation(
            method="Max-Leverage-Safe",
            quantity=float(safe_qty),
            notional_usd=float(safe_notional),
            risk_usd=float(safe_risk_usd),
            risk_pct=float(safe_risk_usd / eq * 100.0),
            max_safe_leverage=float(round(safe_lev, 2)),
            quantity_cap=float(quantity_cap),
            notes=safe_notes,
        ),
    ]


def pre_trade_check(
    proposed: PositionInput,
    existing_positions: Sequence[PositionInput],
    account: AccountState,
    market: MarketDataBundle | Mapping[str, SymbolMarketData] | None,
    rules: RuleConfig,
    journal: JournalAnalytics | None = None,
    *,
    k_atr_buffer: float = 1.5,
    margin_tiers: Mapping[str, Sequence[MarginTier]] | None = None,
) -> PreTradeCheckResult:
    """Evaluate portfolio impact and GO / REDUCE SIZE / NO-GO verdict for a proposed trade.

    The proposed position is evaluated as if filled at its own entry price (mark = entry), so a limit
    entry away from the current mark is judged on entry-to-stop risk and its own liquidation price.
    Existing positions keep the live mark.
    """
    before_analyses = analyze_positions(existing_positions, account, market=market, margin_tiers=margin_tiers)
    before_portfolio = analyze_portfolio(before_analyses, account, market=market, margin_tiers=margin_tiers)
    before_rules, cb = evaluate_rules(before_analyses, before_portfolio, account, rules, journal=journal)

    combined_inputs = [*existing_positions, proposed]
    after_analyses = analyze_positions(
        combined_inputs,
        account,
        market=market,
        margin_tiers=margin_tiers,
        mark_overrides=[None] * len(existing_positions) + [float(proposed.entry_price)],
    )
    after_portfolio = analyze_portfolio(after_analyses, account, market=market, margin_tiers=margin_tiers)
    after_rules, _ = evaluate_rules(after_analyses, after_portfolio, account, rules, journal=journal)

    proposed_analysis = after_analyses[-1]
    before_breaches = [r.rule_id for r in before_rules if r.status == "BREACH"]
    after_breaches = [r.rule_id for r in after_rules if r.status == "BREACH"]
    new_breaches = [b for b in after_breaches if b not in before_breaches]

    reasons: list[str] = []
    verdict: Verdict = "GO"
    rec_qty: float | None = None
    rec_lev: float | None = None

    # Compute sizing options if a valid directional stop exists
    sizing_options: list[SizingRecommendation] = []
    # Collateral already taken out of the cross pool by existing positions (same definition as the LP math)
    other_mm = sum(committed_margin(a) for a in before_analyses)
    if (
        proposed.stop_price is not None
        and ((proposed.side == "long" and proposed.stop_price < proposed.entry_price)
             or (proposed.side == "short" and proposed.stop_price > proposed.entry_price))
    ):
        win_rate = journal.win_rate if (journal and journal.total_trades >= 10) else 0.50
        payoff = journal.payoff_ratio if (journal and journal.payoff_ratio > 0) else 1.5
        sizing_options = recommend_sizes(
            symbol=proposed.symbol,
            side=proposed.side,
            entry_price=proposed.entry_price,
            stop_price=proposed.stop_price,
            account=account,
            atr_1d=proposed_analysis.atr_1d,
            risk_pct=min(1.0, rules.max_risk_per_trade_pct),
            leverage=proposed.leverage,
            margin_mode=proposed.margin_mode,
            k_atr_buffer=k_atr_buffer,
            win_rate=win_rate,
            payoff_ratio=payoff,
            other_positions_mm=other_mm,
            margin_tiers=margin_tiers,
        )

    # Hard NO-GO conditions
    if not cb.trading_allowed:
        verdict = "NO-GO"
        reasons.extend(cb.reasons)

    if proposed_analysis.missing_stop and rules.mandatory_stop:
        verdict = "NO-GO"
        reasons.append("No stop-loss provided while mandatory stop rule is active.")

    if proposed_analysis.missing_stop and not rules.mandatory_stop:
        reasons.append(
            "No stop-loss provided (mandatory stop rule is off): risk is unbounded up to liquidation"
            + (
                f" at ${proposed_analysis.liquidation_price:,.4f}."
                if proposed_analysis.liquidation_price is not None
                else "."
            )
        )

    if proposed_analysis.invalid_stop_direction:
        verdict = "NO-GO"
        reasons.append(
            f"Stop price (${proposed.stop_price:,.4f}) is on the wrong side of the {proposed.side} entry "
            f"(${proposed.entry_price:,.4f})."
        )

    if proposed_analysis.at_or_past_liquidation or proposed_analysis.liquidated:
        verdict = "NO-GO"
        reasons.append("Proposed position is already at or past liquidation price.")

    if proposed_analysis.stop_beyond_liquidation:
        verdict = "NO-GO"
        reasons.append(
            f"Stop (${proposed.stop_price:,.4f}) is beyond liquidation (${proposed_analysis.liquidation_price:,.4f})."
        )
        if sizing_options and sizing_options[-1].max_safe_leverage is not None:
            rec_lev = min(rules.max_leverage, sizing_options[-1].max_safe_leverage)

    if len(combined_inputs) > rules.max_positions:
        verdict = "NO-GO"
        reasons.append(f"Adding this trade brings open positions to {len(combined_inputs)} > max {rules.max_positions}.")

    if exceeds(proposed.leverage, rules.max_leverage):
        if verdict != "NO-GO":
            verdict = "REDUCE SIZE"
        reasons.append(f"Leverage {proposed.leverage:.1f}x exceeds max allowed {rules.max_leverage:.1f}x.")
        rec_lev = min(rules.max_leverage, rec_lev or rules.max_leverage)

    # Size-reducible conditions (REDUCE SIZE unless already NO-GO)
    if verdict != "NO-GO":
        # Check open risk per trade, total open risk, correlated exposure, tight LP
        trade_risk_pct = proposed_analysis.risk_to_stop_pct or 0.0
        remaining_risk_budget_pct = max(0.0, rules.max_total_open_risk_pct - before_portfolio.known_stop_risk_pct)

        if exceeds(trade_risk_pct, rules.max_risk_per_trade_pct):
            verdict = "REDUCE SIZE"
            reasons.append(
                f"Trade stop risk ({trade_risk_pct:.2f}%) exceeds max risk per trade ({rules.max_risk_per_trade_pct:.2f}%)."
            )

        if exceeds(after_portfolio.known_stop_risk_pct, rules.max_total_open_risk_pct):
            verdict = "REDUCE SIZE"
            reasons.append(
                f"Total open risk rises from {before_portfolio.known_stop_risk_pct:.2f}% to "
                f"{after_portfolio.known_stop_risk_pct:.2f}% (limit {rules.max_total_open_risk_pct:.2f}%)."
            )

        if "max_correlated_exposure_pct" in new_breaches or (
            "max_correlated_exposure_pct" in after_breaches
            and after_portfolio.gross_exposure_usd > before_portfolio.gross_exposure_usd
        ):
            verdict = "REDUCE SIZE"
            reasons.append(
                f"Increases correlated exposure into breach of {rules.max_correlated_exposure_pct:.1f}% limit."
            )

        if (
            proposed_analysis.distance_to_liq_atr is not None
            and proposed_analysis.stop_price is not None
            and proposed_analysis.liquidation_price is not None
        ):
            stop_to_lp_atr = abs(proposed_analysis.stop_price - proposed_analysis.liquidation_price) / max(
                1e-9, proposed_analysis.atr_1d
            )
            if stop_to_lp_atr < k_atr_buffer:
                verdict = "REDUCE SIZE"
                reasons.append(
                    f"Liquidation buffer beyond stop is only {stop_to_lp_atr:.2f} ATR (< {k_atr_buffer:.1f} ATR safety target)."
                )
                if sizing_options and sizing_options[-1].max_safe_leverage is not None:
                    rec_lev = min(proposed.leverage, sizing_options[-1].max_safe_leverage)

        if verdict == "REDUCE SIZE":
            # Compute concrete reduced quantity respecting both trade and portfolio open risk caps
            max_allowed_risk_pct = min(rules.max_risk_per_trade_pct, remaining_risk_budget_pct)
            if proposed.stop_price is None:
                # Size cannot be derived from a stop; only leverage can be reduced
                rec_qty = None
            else:
                stop_dist = abs(proposed.entry_price - proposed.stop_price)
                if max_allowed_risk_pct > 0:
                    risk_cap_qty = (account.equity * max_allowed_risk_pct / 100.0) / stop_dist
                    # Largest size allowed by BOTH the risk budget and the liquidation-safety cap
                    # (the cap is independent of the 1% sizing default shown in the sizing table)
                    safety_cap = sizing_options[-1].quantity_cap if sizing_options else None
                    rec_qty = min(proposed.quantity, risk_cap_qty)
                    if safety_cap is not None and safety_cap > 0:
                        rec_qty = min(rec_qty, safety_cap)
                else:
                    rec_qty = 0.0
                    verdict = "NO-GO"
                    reasons.append("Remaining portfolio open risk budget is exhausted (0% left).")

    if verdict == "GO" and not reasons:
        lp_txt = (
            f"LP {proposed_analysis.distance_to_liq_atr:.2f} ATR from entry"
            if proposed_analysis.distance_to_liq_atr is not None
            else f"liquidation price {proposed_analysis.liquidation_note or 'n/a'}"
        )
        reasons.append(
            f"Trade passes all risk rules (risk {proposed_analysis.risk_to_stop_pct or 0.0:.2f}%, "
            f"{lp_txt}, "
            f"95% VaR ${before_portfolio.var_95_hist_usd:,.0f} -> ${after_portfolio.var_95_hist_usd:,.0f})."
        )

    return PreTradeCheckResult(
        verdict=verdict,
        reasons=reasons,
        proposed_analysis=proposed_analysis,
        before_var_95_usd=before_portfolio.var_95_hist_usd,
        after_var_95_usd=after_portfolio.var_95_hist_usd,
        before_var_99_usd=before_portfolio.var_99_hist_usd,
        after_var_99_usd=after_portfolio.var_99_hist_usd,
        before_open_risk_pct=before_portfolio.known_stop_risk_pct,
        after_open_risk_pct=after_portfolio.known_stop_risk_pct,
        before_gross_exposure_usd=before_portfolio.gross_exposure_usd,
        after_gross_exposure_usd=after_portfolio.gross_exposure_usd,
        before_net_exposure_usd=before_portfolio.net_exposure_usd,
        after_net_exposure_usd=after_portfolio.net_exposure_usd,
        before_btc_beta_exposure_usd=before_portfolio.btc_beta_adj_net_usd,
        after_btc_beta_exposure_usd=after_portfolio.btc_beta_adj_net_usd,
        before_breaches=before_breaches,
        after_breaches=after_breaches,
        recommended_quantity=rec_qty,
        recommended_leverage=rec_lev,
        sizing_options=sizing_options,
    )
