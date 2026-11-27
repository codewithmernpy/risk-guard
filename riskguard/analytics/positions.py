"""Per-position risk, unrealized PnL, funding, liquidation, and danger flags."""

from __future__ import annotations

import math
from typing import Mapping, Sequence

from riskguard.analytics.margin import (
    compute_liquidation_price,
    compute_maintenance_margin,
    lookup_margin_tier,
    lookup_margin_tier_with_source,
)
from riskguard.data.providers import DEFAULT_DEMO_FUNDING_8H, MarketDataBundle, SymbolMarketData
from riskguard.models import AccountState, MarginTier, PositionAnalysis, PositionInput


ATR_PROXY_FRACTION = 0.03  # 3% of price, used only when no daily ATR can be computed


def _resolve_market_fields(
    pos: PositionInput,
    market: MarketDataBundle | Mapping[str, SymbolMarketData] | None,
    mark_override: float | None = None,
) -> tuple[float, float, float, bool, bool]:
    """Return (mark_price, funding_8h, atr_1d, used_entry_proxy, used_atr_proxy).

    `mark_override` replaces the market mark (used by the pre-trade check to evaluate a
    proposed order as if filled at its entry price); funding and ATR still come from the feed.
    """
    snap: SymbolMarketData | None = None
    if isinstance(market, MarketDataBundle):
        snap = market.snapshots.get(pos.symbol)
    elif isinstance(market, Mapping):
        snap = market.get(pos.symbol)

    if snap is not None and snap.mark_price > 0:
        mark = float(mark_override) if mark_override is not None and mark_override > 0 else float(snap.mark_price)
        if snap.atr_1d > 0:
            return mark, float(snap.funding_rate_8h), float(snap.atr_1d), False, False
        return mark, float(snap.funding_rate_8h), mark * ATR_PROXY_FRACTION, False, True

    if mark_override is not None and mark_override > 0:
        mark = float(mark_override)
        return mark, DEFAULT_DEMO_FUNDING_8H, mark * ATR_PROXY_FRACTION, False, True
    return float(pos.entry_price), DEFAULT_DEMO_FUNDING_8H, float(pos.entry_price) * ATR_PROXY_FRACTION, True, True


def liquidation_label(a: PositionAnalysis, digits: int = 4) -> str:
    """Human-readable liquidation price: '$x' or the reason there is no liquidation price."""
    if a.liquidation_price is not None:
        return f"${a.liquidation_price:,.{digits}f}"
    return a.liquidation_note or "n/a"


def effective_leverage_label(a: PositionAnalysis, digits: int = 2) -> str:
    """Human-readable effective leverage ('Nx') or 'LIQUIDATED' when liquidated/non-finite."""
    if a.liquidated or not math.isfinite(a.effective_leverage):
        return "LIQUIDATED"
    return f"{a.effective_leverage:.{digits}f}x"


def committed_margin(a: PositionAnalysis) -> float:
    """Collateral position `a` takes out of the shared cross-margin pool.

    Isolated: its own isolated equity (initial margin + uPnL, floored at 0) is ring-fenced.
    Cross: its maintenance margin MM = max(0, notional * MMR - cumB).
    """
    if a.margin_mode == "isolated":
        return max(0.0, a.margin_used + a.unrealized_pnl)
    return max(0.0, a.notional * a.mmr - a.cum_b)


def analyze_position(
    pos: PositionInput,
    account: AccountState,
    market: MarketDataBundle | Mapping[str, SymbolMarketData] | None = None,
    *,
    other_positions_mm: float = 0.0,
    margin_tiers: Mapping[str, Sequence[MarginTier]] | None = None,
    mark_override: float | None = None,
) -> PositionAnalysis:
    """Compute full risk and liquidation metrics for a single perpetual position.

    Cross-margin approximation:
        WB_cross = max(0, account.equity - uPnL_i - other_positions_mm)
        where other_positions_mm is the collateral the other positions take out of the cross pool
        (cross: maintenance margin; isolated: ring-fenced isolated equity), see `committed_margin`.
        Subtracting uPnL_i removes position i's current mark PnL from account equity so the
        Binance entry-anchored LP formula does not double-count it. When mark == entry,
        WB_cross equals (account.equity - other_positions_mm) exactly.
    """
    mark_price, funding_8h, atr_1d, used_proxy, used_atr_proxy = _resolve_market_fields(
        pos, market, mark_override=mark_override
    )
    s = float(pos.side_sign)
    q = float(pos.quantity)
    ep = float(pos.entry_price)

    eq = float(account.equity)
    notional = q * mark_price
    entry_notional = q * ep
    margin_used = entry_notional / float(pos.leverage)
    unrealized_pnl = s * q * (mark_price - ep)
    unrealized_pnl_pct = (unrealized_pnl / margin_used * 100.0) if margin_used > 0 else 0.0

    account_leverage_share = (notional / eq) if eq > 0 else math.nan

    tier, tier_source = lookup_margin_tier_with_source(
        pos.symbol,
        max(notional, entry_notional),
        margin_tiers=margin_tiers,
        fallback_mmr=account.default_mmr,
    )

    if pos.margin_mode == "isolated":
        wallet_balance = margin_used
    else:
        wallet_balance = max(0.0, eq - unrealized_pnl - float(other_positions_mm))

    lp = compute_liquidation_price(
        wallet_balance=wallet_balance,
        quantity=q,
        entry_price=ep,
        side=pos.side,
        mmr=tier.mmr,
        cum_b=tier.cum_b,
    )

    at_or_past_lp = False
    if lp is not None and mark_price > 0:
        if pos.side == "long":
            at_or_past_lp = mark_price <= lp
        else:
            at_or_past_lp = mark_price >= lp

    remaining_equity = (margin_used + unrealized_pnl) if pos.margin_mode == "isolated" else eq
    liquidated = bool(at_or_past_lp or remaining_equity <= 0.0 or eq <= 0.0)

    if liquidated:
        effective_leverage = math.nan
    elif pos.margin_mode == "isolated":
        effective_leverage = notional / remaining_equity
    else:
        effective_leverage = account_leverage_share

    # Distance to liquidation in % and ATR multiples (signed: >0 safe; None once liquidated)
    dist_pct: float | None = None
    dist_atr: float | None = None
    if lp is not None and mark_price > 0 and not liquidated:
        if pos.side == "long":
            dist_pct = (mark_price - lp) / mark_price * 100.0
            dist_atr = (mark_price - lp) / atr_1d if atr_1d > 0 else None
        else:
            dist_pct = (lp - mark_price) / mark_price * 100.0
            dist_atr = (lp - mark_price) / atr_1d if atr_1d > 0 else None

    liquidation_note: str | None = None
    if lp is None:
        if pos.margin_mode == "cross":
            liquidation_note = "none at current equity (cross)"
        else:
            liquidation_note = "none (isolated margin covers a fall to $0)"

    # Stop / TP risk metrics
    missing_stop = pos.stop_price is None
    # A stop at or through the current mark (long: mark <= stop, short: mark >= stop) has already been
    # crossed: it should have filled, so it is not protecting the position (missing on the exchange,
    # gapped through, or entered on the wrong side). It is treated as unprotected, never as zero risk.
    invalid_stop_direction = False
    risk_to_stop_usd: float | None = None
    risk_to_stop_pct: float | None = None
    stop_beyond_liq = False
    reward_to_risk: float | None = None

    if pos.stop_price is not None:
        stop_px = float(pos.stop_price)
        invalid_stop_direction = (pos.side == "long" and mark_price <= stop_px) or (
            pos.side == "short" and mark_price >= stop_px
        )

        if not invalid_stop_direction:
            # Live risk from mark to stop (> 0 here; a stop trailed into profit still gives back mark - stop)
            risk_to_stop_usd = float(s * q * (mark_price - stop_px))
            risk_to_stop_pct = (risk_to_stop_usd / eq * 100.0) if eq > 0 else None

            if lp is not None:
                if pos.side == "long" and stop_px <= lp:
                    stop_beyond_liq = True
                elif pos.side == "short" and stop_px >= lp:
                    stop_beyond_liq = True

            if pos.take_profit is not None and risk_to_stop_usd > 0:
                tp = float(pos.take_profit)
                reward = s * q * (tp - mark_price)
                reward_to_risk = float(reward / risk_to_stop_usd)

    # Funding cost estimate: positive = cost paid by trader, negative = funding earned
    funding_rate_daily = funding_8h * 3.0
    funding_cost_daily_usd = s * notional * funding_rate_daily

    warnings: list[str] = []
    danger_level: str = "OK"

    if liquidated:
        if at_or_past_lp and lp is not None:
            warnings.append(
                f"DANGER: {pos.symbol} mark (${mark_price:,.4f}) is at/past liquidation (${lp:,.4f})!"
            )
        else:
            warnings.append(
                f"DANGER: {pos.symbol} remaining position equity is <= $0 — position is LIQUIDATED!"
            )
        danger_level = "DANGER"
    if stop_beyond_liq and lp is not None and pos.stop_price is not None:
        warnings.append(
            f"DANGER: {pos.symbol} stop (${pos.stop_price:,.4f}) is beyond liquidation (${lp:,.4f}) — "
            f"position liquidates before stop fills."
        )
        danger_level = "DANGER"
    if missing_stop:
        warnings.append(f"DANGER: {pos.symbol} has no stop-loss — risk is unbounded up to liquidation.")
        danger_level = "DANGER"
    if invalid_stop_direction and pos.stop_price is not None:
        where = "below" if pos.side == "long" else "above"
        warnings.append(
            f"DANGER: {pos.symbol} stop already crossed — mark (${mark_price:,.4f}) is at/{where} the "
            f"{pos.side} stop (${pos.stop_price:,.4f}), so the stop is not protecting the position; "
            "risk is unbounded up to liquidation."
        )
        danger_level = "DANGER"
    if dist_atr is not None and 0 < dist_atr < 1.5 and danger_level != "DANGER":
        warnings.append(
            f"WARN: {pos.symbol} liquidation is only {dist_atr:.2f} daily ATRs away ({dist_pct:.2f}%)."
        )
        danger_level = "WARN"
    if lp is None and pos.margin_mode == "cross":
        warnings.append(
            f"INFO: {pos.symbol} (cross) has no liquidation price at current equity — the shared account "
            "equity covers a fall to $0 with other positions held at mark. Account-level cross liquidation "
            "is checked in the stress tests."
        )
    if used_proxy:
        warnings.append(f"INFO: Live/sample price unavailable for {pos.symbol}; using entry price as mark proxy.")
    elif used_atr_proxy:
        warnings.append(f"INFO: Daily ATR unavailable for {pos.symbol}; using 3% of mark as ATR proxy.")

    return PositionAnalysis(
        symbol=pos.symbol,
        side=pos.side,
        margin_mode=pos.margin_mode,
        quantity=q,
        entry_price=ep,
        mark_price=mark_price,
        notional=float(notional),
        margin_used=float(margin_used),
        leverage=float(pos.leverage),
        effective_leverage=float(effective_leverage),
        account_leverage_share=float(account_leverage_share),
        unrealized_pnl=float(unrealized_pnl),
        unrealized_pnl_pct=float(unrealized_pnl_pct),
        stop_price=pos.stop_price,
        take_profit=pos.take_profit,
        risk_to_stop_usd=risk_to_stop_usd,
        risk_to_stop_pct=risk_to_stop_pct,
        reward_to_risk=reward_to_risk,
        funding_rate_8h=float(funding_8h),
        funding_cost_daily_usd=float(funding_cost_daily_usd),
        mmr=float(tier.mmr),
        cum_b=float(tier.cum_b),
        liquidation_price=lp,
        liquidation_note=liquidation_note,
        distance_to_liq_pct=dist_pct,
        distance_to_liq_atr=dist_atr,
        atr_1d=float(atr_1d),
        stop_beyond_liquidation=stop_beyond_liq,
        missing_stop=missing_stop,
        invalid_stop_direction=invalid_stop_direction,
        at_or_past_liquidation=at_or_past_lp,
        liquidated=liquidated,
        used_entry_proxy=used_proxy,
        margin_tier_source=tier_source,
        warnings=warnings,
        danger_level=danger_level,  # type: ignore[arg-type]
    )


def analyze_positions(
    positions: Sequence[PositionInput],
    account: AccountState,
    market: MarketDataBundle | Mapping[str, SymbolMarketData] | None = None,
    margin_tiers: Mapping[str, Sequence[MarginTier]] | None = None,
    mark_overrides: Sequence[float | None] | None = None,
) -> list[PositionAnalysis]:
    """Analyze all open positions, accounting for cross-margin shared equity.

    `mark_overrides[i]`, when given and not None, replaces the market mark for position i
    (the pre-trade check uses it to evaluate a proposed order at its own entry price).
    """
    if not positions:
        return []
    overrides = list(mark_overrides) if mark_overrides is not None else [None] * len(positions)
    if len(overrides) != len(positions):
        raise ValueError("mark_overrides must have one entry per position.")

    # Collateral each position takes out of the cross pool (same definition as `committed_margin`):
    # cross -> maintenance margin; isolated -> ring-fenced isolated equity (margin + uPnL, floored at 0).
    committed_margins: list[float] = []
    for p, ovr in zip(positions, overrides):
        mark_px, _, _, _, _ = _resolve_market_fields(p, market, mark_override=ovr)
        q = abs(p.quantity)
        notional = q * mark_px
        if p.margin_mode == "isolated":
            iso_margin = (q * p.entry_price) / float(p.leverage)
            upnl = p.side_sign * q * (mark_px - p.entry_price)
            committed_margins.append(max(0.0, iso_margin + upnl))
        else:
            tier = lookup_margin_tier(
                p.symbol,
                max(notional, q * p.entry_price),
                margin_tiers=margin_tiers,
                fallback_mmr=account.default_mmr,
            )
            committed_margins.append(compute_maintenance_margin(notional, tier))

    total_committed = sum(committed_margins)
    results: list[PositionAnalysis] = []
    for idx, p in enumerate(positions):
        others_mm = max(0.0, total_committed - committed_margins[idx])
        results.append(
            analyze_position(
                p,
                account,
                market=market,
                other_positions_mm=others_mm,
                margin_tiers=margin_tiers,
                mark_override=overrides[idx],
            )
        )
    return results
