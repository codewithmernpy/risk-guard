"""Deterministic synthetic trade journal generator (SYNTHETIC demo data)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

SYNTHETIC_BANNER = "SYNTHETIC DEMO JOURNAL — generated deterministically for educational/demo use only."

MIN_SYNTHETIC_TRADES = 53  # 10 baseline + 4 planted 10-trade episodes + 3 final-session trades

DEMO_SYMBOLS_PRICES: dict[str, float] = {
    "BTCUSDT": 105_000.0,
    "ETHUSDT": 3_600.0,
    "SOLUSDT": 180.0,
    "XRPUSDT": 2.45,
    "DOGEUSDT": 0.19,
    "BNBUSDT": 980.0,
}


def generate_synthetic_journal(
    *,
    num_trades: int = 400,
    days: int = 180,
    end_date: datetime | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate ~400 synthetic closed trades over 6 months with realistic tilt, revenge, and overtrading episodes.

    Returns a DataFrame with columns:
        opened_at, closed_at, symbol, side, quantity, entry_price, exit_price,
        planned_stop, fees, pnl, source
    """
    if num_trades < MIN_SYNTHETIC_TRADES:
        raise ValueError(
            f"num_trades must be >= {MIN_SYNTHETIC_TRADES}: the generator plants 4 ten-trade behaviour "
            "episodes, a 3-trade final session and at least 10 baseline trades."
        )
    rng = np.random.default_rng(seed)
    end_dt = end_date or datetime(2025, 10, 31, 20, 0, tzinfo=timezone.utc)
    start_dt = end_dt - timedelta(days=days)

    symbols = list(DEMO_SYMBOLS_PRICES.keys())
    symbol_weights = np.array([0.24, 0.22, 0.20, 0.12, 0.12, 0.10])

    # Reserve ~50 trades for explicit overtrading / tilt / revenge clusters across the 6 months
    # and ~350 baseline trades spaced across days.
    rows: list[dict[str, object]] = []

    def _make_trade(
        opened_at: datetime,
        hold_minutes: float,
        sym: str,
        side: str,
        notional: float,
        r_multiple: float,
        stop_pct: float = 0.02,
        include_stop: bool = True,
    ) -> dict[str, object]:
        base_px = DEMO_SYMBOLS_PRICES[sym]
        ep = float(base_px * rng.uniform(0.88, 1.12))
        qty = float(notional / ep)
        stop_dist = ep * stop_pct
        sign = 1.0 if side == "long" else -1.0
        planned_stop = (ep - sign * stop_dist) if include_stop else None
        # Price move corresponds to r_multiple * stop_dist
        xp = max(1e-6, ep + sign * r_multiple * stop_dist)
        gross_pnl = sign * qty * (xp - ep)
        fees = float(notional * 0.0008)  # 4 bps per side round-trip taker
        net_pnl = float(gross_pnl - fees)
        closed_at = (opened_at + timedelta(minutes=float(hold_minutes))).replace(microsecond=0)
        return {
            "opened_at": opened_at.replace(microsecond=0).isoformat(),
            "closed_at": closed_at.isoformat(),
            "symbol": sym,
            "side": side,
            "quantity": round(qty, 6),
            "entry_price": round(ep, 6),
            "exit_price": round(xp, 6),
            "planned_stop": round(planned_stop, 6) if planned_stop is not None else "",
            "fees": round(fees, 4),
            "pnl": round(net_pnl, 4),
            "source": "SYNTHETIC",
        }

    # Pick 4 explicit tilt/overtrading days inside the 180-day span
    episode_days = [32, 78, 129, 168]
    episode_set = set(episode_days)

    # First place ~2 trades/day on non-episode days until reaching target baseline count
    baseline_target = max(10, num_trades - len(episode_days) * 10 - 12)
    cur_t = start_dt
    for i in range(baseline_target):
        day_offset = int((i / max(1, baseline_target)) * (days - 2))
        while day_offset in episode_set:
            day_offset += 1
        hour = int(rng.integers(7, 22))
        minute = int(rng.integers(0, 60))
        opened = start_dt + timedelta(days=day_offset, hours=hour, minutes=minute)
        if opened <= cur_t:
            opened = cur_t + timedelta(minutes=int(rng.integers(45, 240)))
        cur_t = opened

        sym = str(rng.choice(symbols, p=symbol_weights))
        side = "long" if rng.random() < 0.62 else "short"
        notional = float(rng.uniform(2200.0, 4200.0))
        is_win = bool(rng.random() < 0.52)
        if is_win:
            r_mult = float(rng.uniform(0.7, 2.4))
            hold_min = float(rng.uniform(35.0, 260.0))
        else:
            r_mult = -float(rng.uniform(0.7, 1.15))
            # Mild disposition bias even in baseline
            hold_min = float(rng.uniform(80.0, 420.0))
        has_stop = bool(rng.random() < 0.92)
        rows.append(_make_trade(opened, hold_min, sym, side, notional, r_mult, include_stop=has_stop))

    # Plant 4 multi-trade tilt / overtrading / revenge / size-escalation / stop-violation episodes
    for ep_idx, ep_day in enumerate(episode_days):
        t0 = start_dt + timedelta(days=ep_day, hours=13, minutes=0)
        sym = symbols[ep_idx % len(symbols)]
        # Trade 1: initial stop loss
        t1_notional = 3000.0
        t1 = _make_trade(t0, 45.0, sym, "long", t1_notional, -1.05, include_stop=True)
        rows.append(t1)

        # Trade 2: Revenge trade 8 minutes later with 1.8x size -> loses + stop violation (-1.7R)
        t2_open = t0 + timedelta(minutes=53)
        t2 = _make_trade(t2_open, 70.0, sym, "long", t1_notional * 1.8, -1.70, include_stop=True)
        rows.append(t2)

        # Trade 3: Size escalation after 2 consecutive losses (2.2x size!) -> loses again
        t3_open = t2_open + timedelta(minutes=78)
        t3 = _make_trade(t3_open, 190.0, "DOGEUSDT", "long", t1_notional * 2.2, -1.55, include_stop=True)
        rows.append(t3)

        # Trades 4..10: Overtrading chop within the same UTC day (10 total trades on that day)
        chop_t = t3_open + timedelta(minutes=25)
        for k in range(7):
            c_sym = str(rng.choice(symbols, p=symbol_weights))
            c_side = "long" if k % 2 == 0 else "short"
            c_r = float(rng.uniform(0.5, 1.1)) if k % 3 == 0 else -float(rng.uniform(0.5, 1.0))
            c_hold = float(rng.uniform(15.0, 55.0))
            rows.append(_make_trade(chop_t, c_hold, c_sym, c_side, 2800.0, c_r, include_stop=True))
            chop_t = chop_t + timedelta(minutes=int(c_hold + 12))

    # Top up remaining trades to hit exact num_trades cleanly, ending on end_dt's day with a minor losing session
    while len(rows) < num_trades - 3:
        day_offset = int(rng.integers(1, days - 3))
        if day_offset in episode_set:
            continue
        opened = start_dt + timedelta(days=day_offset, hours=int(rng.integers(0, 23)), minutes=int(rng.integers(0, 59)))
        sym = str(rng.choice(symbols, p=symbol_weights))
        side = "long" if rng.random() < 0.58 else "short"
        is_win = bool(rng.random() < 0.53)
        r_mult = float(rng.uniform(0.8, 2.1)) if is_win else -float(rng.uniform(0.75, 1.05))
        hold_min = float(rng.uniform(40.0, 240.0)) if is_win else float(rng.uniform(90.0, 480.0))
        rows.append(_make_trade(opened, hold_min, sym, side, float(rng.uniform(2400.0, 3800.0)), r_mult, include_stop=True))

    # Final day session trades (3 trades ending on end_dt with moderate drawdown/session loss)
    final_day_base = end_dt.replace(hour=12, minute=0, second=0, microsecond=0)
    rows.append(_make_trade(final_day_base, 60.0, "ETHUSDT", "long", 3500.0, -1.0, include_stop=True))
    rows.append(_make_trade(final_day_base + timedelta(minutes=75), 95.0, "SOLUSDT", "long", 4400.0, -1.25, include_stop=True))
    rows.append(_make_trade(final_day_base + timedelta(minutes=190), 50.0, "BTCUSDT", "short", 3200.0, 0.85, include_stop=True))

    df = pd.DataFrame(rows)
    df["_sort_key"] = pd.to_datetime(df["closed_at"], utc=True, format="ISO8601")
    df = df.sort_values("_sort_key").drop(columns=["_sort_key"]).reset_index(drop=True)
    return df


def save_demo_journal(output_path: str | Path, **kwargs: object) -> Path:
    """Generate and save synthetic demo journal CSV to disk."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = generate_synthetic_journal(**kwargs)  # type: ignore[arg-type]
    df.to_csv(path, index=False, lineterminator="\n")  # same bytes on every OS
    return path
