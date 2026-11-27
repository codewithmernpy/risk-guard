"""Maintenance-margin tier tables and Binance USDT-M liquidation price math."""

from __future__ import annotations

import math
from typing import Mapping, Sequence

from riskguard.models import MarginTier, Side, normalize_symbol


def build_continuous_tiers(brackets: Sequence[tuple[float, float]]) -> list[MarginTier]:
    """Build MarginTier objects with continuous cumB maintenance deductions.

    Each tuple is (max_notional_usdt, mmr). For tier k > 0:
        cumB_k = cumB_{k-1} + max_notional_{k-1} * (mmr_k - mmr_{k-1})
    so that MM(N) = N * mmr - cumB is continuous across bracket cutoffs.
    """
    tiers: list[MarginTier] = []
    prev_cum = 0.0
    prev_max = 0.0
    prev_mmr = 0.0
    for idx, (max_notional, mmr) in enumerate(brackets):
        if idx == 0:
            cum_b = 0.0
        else:
            cum_b = prev_cum + prev_max * (mmr - prev_mmr)
        tiers.append(MarginTier(max_notional=float(max_notional), mmr=float(mmr), cum_b=float(cum_b)))
        prev_cum = cum_b
        prev_max = float(max_notional)
        prev_mmr = float(mmr)
    return tiers


# Simplified Binance-inspired USDT-M linear perp maintenance-margin brackets.
# Real exchange brackets can vary by account VIP/leverage config and change dynamically;
# callers may pass custom tier tables via `margin_tiers`.
DEFAULT_MARGIN_TIERS: dict[str, list[MarginTier]] = {
    "BTCUSDT": build_continuous_tiers(
        [
            (50_000.0, 0.004),
            (250_000.0, 0.005),
            (1_000_000.0, 0.010),
            (5_000_000.0, 0.025),
            (math.inf, 0.050),
        ]
    ),
    "ETHUSDT": build_continuous_tiers(
        [
            (50_000.0, 0.004),
            (250_000.0, 0.005),
            (1_000_000.0, 0.010),
            (5_000_000.0, 0.025),
            (math.inf, 0.050),
        ]
    ),
    "BNBUSDT": build_continuous_tiers(
        [
            (25_000.0, 0.005),
            (100_000.0, 0.010),
            (500_000.0, 0.025),
            (math.inf, 0.050),
        ]
    ),
    "SOLUSDT": build_continuous_tiers(
        [
            (25_000.0, 0.005),
            (100_000.0, 0.010),
            (500_000.0, 0.025),
            (math.inf, 0.050),
        ]
    ),
    "XRPUSDT": build_continuous_tiers(
        [
            (25_000.0, 0.005),
            (100_000.0, 0.010),
            (500_000.0, 0.025),
            (math.inf, 0.050),
        ]
    ),
    "DOGEUSDT": build_continuous_tiers(
        [
            (25_000.0, 0.005),
            (100_000.0, 0.010),
            (500_000.0, 0.025),
            (math.inf, 0.050),
        ]
    ),
    "_DEFAULT": build_continuous_tiers(
        [
            (25_000.0, 0.005),
            (100_000.0, 0.010),
            (500_000.0, 0.025),
            (math.inf, 0.050),
        ]
    ),
}


def lookup_margin_tier_with_source(
    symbol: str,
    notional: float,
    margin_tiers: Mapping[str, Sequence[MarginTier]] | None = None,
    fallback_mmr: float | None = None,
) -> tuple[MarginTier, str]:
    """Return (MarginTier, source_name) where source_name is the symbol key, '_DEFAULT', or 'fallback'."""
    norm = normalize_symbol(symbol)
    table = margin_tiers if margin_tiers is not None else DEFAULT_MARGIN_TIERS
    if table.get(norm):
        tiers = table[norm]
        source = norm
    elif table.get("_DEFAULT"):
        tiers = table["_DEFAULT"]
        source = "_DEFAULT"
    else:
        mmr = float(fallback_mmr if fallback_mmr is not None else 0.005)
        return MarginTier(max_notional=math.inf, mmr=mmr, cum_b=0.0), "fallback"

    abs_notional = abs(float(notional))
    for tier in tiers:
        if abs_notional <= tier.max_notional:
            return tier, source
    return tiers[-1], source


def lookup_margin_tier(
    symbol: str,
    notional: float,
    margin_tiers: Mapping[str, Sequence[MarginTier]] | None = None,
    fallback_mmr: float | None = None,
) -> MarginTier:
    """Return the maintenance-margin tier for a symbol and position notional."""
    tier, _ = lookup_margin_tier_with_source(
        symbol,
        notional,
        margin_tiers=margin_tiers,
        fallback_mmr=fallback_mmr,
    )
    return tier


def compute_maintenance_margin(notional: float, tier: MarginTier) -> float:
    """Compute maintenance margin MM(N) = max(0, N * MMR - cumB)."""
    return max(0.0, abs(float(notional)) * tier.mmr - tier.cum_b)


def compute_liquidation_price(
    wallet_balance: float,
    quantity: float,
    entry_price: float,
    side: Side,
    mmr: float,
    cum_b: float = 0.0,
) -> float | None:
    """Compute USDT-M linear perpetual liquidation price via the Binance formula.

    Formula:
        LP = (WB + cumB - side * Q * EP) / (Q * MMR - side * Q)

    where:
        WB   = allocated position margin (isolated) or available cross wallet balance (cross)
        cumB = maintenance margin deduction for the position's tier
        side = +1 for long, -1 for short
        Q    = positive base quantity
        EP   = entry price
        MMR  = maintenance margin rate

    Returns None if LP <= 0 (e.g., <1x long cannot liquidate at a positive price).
    """
    q = abs(float(quantity))
    if q <= 0 or entry_price <= 0:
        return None
    s = 1.0 if side == "long" else -1.0
    denom = q * float(mmr) - s * q
    if abs(denom) <= 1e-15:
        return None
    numer = float(wallet_balance) + float(cum_b) - s * q * float(entry_price)
    lp = numer / denom
    if not math.isfinite(lp) or lp <= 0.0:
        return None
    return float(lp)
