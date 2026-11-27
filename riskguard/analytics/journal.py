"""Trade journal analytics, behavioural detectors, and 0-100 discipline scoring."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Sequence

import numpy as np
import pandas as pd

from riskguard.models import BehaviourEpisode, JournalAnalytics, JournalTrade


def _discipline_grade(score: float) -> str:
    if score >= 90:
        return "A"
    if score >= 80:
        return "B"
    if score >= 70:
        return "C"
    if score >= 55:
        return "D"
    return "F"


def journal_reference_time(trades: Sequence[JournalTrade], now: datetime | None = None) -> datetime | None:
    """Reference time for 'today' / 'this week' PnL and the circuit breaker.

    A real journal is evaluated against the actual current time (so a losing day last week cannot halt
    trading today). A journal made only of SYNTHETIC demo trades is historical sample data, so it is
    evaluated as of its last closed trade (returns None, which `analyze_journal` maps to that time).
    """
    if trades and all((t.source or "").strip().upper() == "SYNTHETIC" for t in trades):
        return None
    return now or datetime.now(timezone.utc)


def _empty_journal_analytics(reference_date: str) -> JournalAnalytics:
    return JournalAnalytics(
        total_trades=0,
        win_count=0,
        loss_count=0,
        win_rate=0.0,
        total_pnl_usd=0.0,
        total_fees_usd=0.0,
        avg_win_usd=0.0,
        avg_loss_usd=0.0,
        payoff_ratio=0.0,
        expectancy_usd=0.0,
        profit_factor=0.0,
        avg_r=None,
        expectancy_r=None,
        r_count=0,
        max_drawdown_pct=0.0,
        max_drawdown_usd=0.0,
        current_drawdown_pct=0.0,
        max_drawdown_duration_hours=0.0,
        current_drawdown_duration_hours=0.0,
        avg_win_hold_minutes=0.0,
        avg_loss_hold_minutes=0.0,
        disposition_ratio=1.0,
        today_pnl_usd=0.0,
        week_pnl_usd=0.0,
        reference_date=reference_date,
        discipline_score=100.0,
        discipline_grade="A",
        discipline_components={},
    )


def analyze_journal(
    trades: Sequence[JournalTrade],
    *,
    starting_equity: float = 10_000.0,
    reference_time: datetime | None = None,
    revenge_window_minutes: float = 30.0,
    revenge_size_ratio: float = 1.15,
    overtrading_min_trades: int = 6,
    overtrading_multiplier: float = 2.0,
    escalation_streak_len: int = 2,
    escalation_size_ratio: float = 1.4,
    disposition_ratio_threshold: float = 1.5,
    stop_slippage_tolerance: float = 0.15,
    tilt_window_hours: float = 4.0,
    tilt_min_losses: int = 3,
    tilt_loss_pct_equity: float = 3.0,
) -> JournalAnalytics:
    """Compute journal performance, equity/drawdown curve, behaviour detectors, and discipline score."""
    if not trades:
        ref = (reference_time or datetime.now(timezone.utc)).date().isoformat()
        return _empty_journal_analytics(ref)

    sorted_trades = sorted(trades, key=lambda t: t.closed_at)
    ref_dt = reference_time if reference_time is not None else sorted_trades[-1].closed_at
    if ref_dt.tzinfo is None:
        ref_dt = ref_dt.replace(tzinfo=timezone.utc)
    else:
        ref_dt = ref_dt.astimezone(timezone.utc)

    # 1. Equity curve & drawdown depth/duration
    # Duration of a drawdown = time from the peak until equity is back at/above that peak (recovery),
    # or until the reference time for a drawdown that is still open.
    equity = float(starting_equity)
    peak_equity = equity
    peak_time = sorted_trades[0].opened_at
    max_dd_usd = 0.0
    max_dd_pct = 0.0
    max_dd_dur_h = 0.0
    in_drawdown = False
    equity_curve: list[dict[str, Any]] = []

    for idx, t in enumerate(sorted_trades, start=1):
        equity += t.pnl
        if equity >= peak_equity:
            if in_drawdown:
                recovery_h = max(0.0, (t.closed_at - peak_time).total_seconds() / 3600.0)
                max_dd_dur_h = max(max_dd_dur_h, recovery_h)
                in_drawdown = False
            peak_equity = equity
            peak_time = t.closed_at
            dd_usd = 0.0
            dd_pct = 0.0
            dd_dur_h = 0.0
        else:
            in_drawdown = True
            dd_usd = peak_equity - equity
            dd_pct = (dd_usd / peak_equity * 100.0) if peak_equity > 0 else 0.0
            dd_dur_h = max(0.0, (t.closed_at - peak_time).total_seconds() / 3600.0)
            max_dd_usd = max(max_dd_usd, dd_usd)
            max_dd_pct = max(max_dd_pct, dd_pct)
            max_dd_dur_h = max(max_dd_dur_h, dd_dur_h)

        equity_curve.append(
            {
                "trade_num": idx,
                "closed_at": t.closed_at.isoformat(),
                "symbol": t.symbol,
                "pnl": round(t.pnl, 4),
                "equity": round(equity, 2),
                "peak_equity": round(peak_equity, 2),
                "drawdown_pct": round(dd_pct, 4),
            }
        )

    current_dd_pct = ((peak_equity - equity) / peak_equity * 100.0) if peak_equity > 0 else 0.0
    current_dd_dur_h = (
        max(0.0, (ref_dt - peak_time).total_seconds() / 3600.0) if equity < peak_equity else 0.0
    )
    max_dd_dur_h = max(max_dd_dur_h, current_dd_dur_h)

    # 2. Core statistics & R distribution
    wins = [t for t in sorted_trades if t.pnl > 0]
    losses = [t for t in sorted_trades if t.pnl < 0]
    n_trades = len(sorted_trades)
    win_rate = len(wins) / n_trades
    total_pnl = sum(t.pnl for t in sorted_trades)
    total_fees = sum(t.fees for t in sorted_trades)
    gross_profit = sum(t.pnl for t in wins)
    gross_loss = abs(sum(t.pnl for t in losses))
    avg_win = (gross_profit / len(wins)) if wins else 0.0
    avg_loss = (gross_loss / len(losses)) if losses else 0.0
    payoff_ratio = (avg_win / avg_loss) if avg_loss > 0 else (99.0 if avg_win > 0 else 0.0)
    expectancy = total_pnl / n_trades
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (99.0 if gross_profit > 0 else 0.0)

    r_multiples: list[float] = []
    stop_violations: list[BehaviourEpisode] = []
    for t in sorted_trades:
        if t.planned_stop is not None:
            risk_unit = abs(t.entry_price - t.planned_stop) * t.quantity
            if risk_unit > 1e-9:
                r_val = t.pnl / risk_unit
                r_multiples.append(float(r_val))

                # Stop violation check: lost significantly more than planned stop distance (ex-fees or price)
                loss_ex_fees = -(t.pnl + t.fees)
                crossed_stop = (t.side == "long" and t.exit_price < t.planned_stop) or (
                    t.side == "short" and t.exit_price > t.planned_stop
                )
                if t.pnl < 0 and crossed_stop and loss_ex_fees > risk_unit * (1.0 + stop_slippage_tolerance):
                    ratio = loss_ex_fees / risk_unit
                    stop_violations.append(
                        BehaviourEpisode(
                            episode_type="stop_violation",
                            timestamp=t.closed_at,
                            symbol=t.symbol,
                            severity="HIGH" if ratio >= 1.5 else "MEDIUM",
                            description=(
                                f"{t.symbol} loss (${loss_ex_fees:,.2f}, -{ratio:.2f}R) exceeded planned stop risk "
                                f"(${risk_unit:,.2f})."
                            ),
                            metrics={"planned_risk_usd": round(risk_unit, 2), "actual_loss_usd": round(loss_ex_fees, 2), "r_loss": round(ratio, 2)},
                        )
                    )

    avg_r = float(np.mean(r_multiples)) if r_multiples else None

    # 3. Today / Week PnL relative to ref_dt
    ref_date = ref_dt.date()
    week_start = (ref_dt - timedelta(days=ref_dt.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)
    today_pnl = sum(t.pnl for t in sorted_trades if t.closed_at.date() == ref_date)
    week_pnl = sum(t.pnl for t in sorted_trades if week_start <= t.closed_at <= ref_dt)

    # 4. Breakdowns by symbol / weekday / hour
    df = pd.DataFrame(
        [
            {
                "symbol": t.symbol,
                "pnl": t.pnl,
                "fees": t.fees,
                "is_win": t.pnl > 0,
                "weekday": t.closed_at.strftime("%a"),
                "weekday_num": t.closed_at.weekday(),
                "hour": t.closed_at.hour,
            }
            for t in sorted_trades
        ]
    )
    by_symbol = (
        df.groupby("symbol", as_index=False)
        .agg(trades=("pnl", "count"), win_rate=("is_win", "mean"), total_pnl=("pnl", "sum"), avg_pnl=("pnl", "mean"))
        .sort_values("total_pnl", ascending=False)
        .to_dict(orient="records")
    )
    by_weekday = (
        df.groupby(["weekday_num", "weekday"], as_index=False)
        .agg(trades=("pnl", "count"), win_rate=("is_win", "mean"), total_pnl=("pnl", "sum"))
        .sort_values("weekday_num")
        .to_dict(orient="records")
    )
    by_hour = (
        df.groupby("hour", as_index=False)
        .agg(trades=("pnl", "count"), win_rate=("is_win", "mean"), total_pnl=("pnl", "sum"))
        .sort_values("hour")
        .to_dict(orient="records")
    )

    # 5. Behavioural detectors
    episodes: list[BehaviourEpisode] = list(stop_violations)

    # 5a. Revenge trading: new trade opened within N minutes of closing a loss with larger size
    trades_by_open = sorted(sorted_trades, key=lambda t: t.opened_at)
    closed_losses = [t for t in sorted_trades if t.pnl < 0]
    for trade in trades_by_open:
        # Most recent closed loss strictly before or at trade.opened_at
        recent_losses = [
            c
            for c in closed_losses
            if c is not trade
            and 0.0 <= (trade.opened_at - c.closed_at).total_seconds() / 60.0 <= revenge_window_minutes
        ]
        if not recent_losses:
            continue
        prev_loss = max(recent_losses, key=lambda c: c.closed_at)
        gap_min = (trade.opened_at - prev_loss.closed_at).total_seconds() / 60.0
        size_ratio = trade.entry_notional / max(1e-9, prev_loss.entry_notional)
        if size_ratio >= revenge_size_ratio:
            episodes.append(
                BehaviourEpisode(
                    episode_type="revenge_trading",
                    timestamp=trade.opened_at,
                    symbol=trade.symbol,
                    severity="HIGH" if size_ratio >= 1.75 or gap_min <= 10 else "MEDIUM",
                    description=(
                        f"Opened {trade.symbol} (${trade.entry_notional:,.0f}) {gap_min:.1f}m after "
                        f"{prev_loss.symbol} loss (${prev_loss.pnl:,.2f}) — {size_ratio:.2f}x larger size."
                    ),
                    metrics={"minutes_after_loss": round(gap_min, 1), "size_ratio": round(size_ratio, 2)},
                )
            )

    # 5b. Overtrading: daily trade counts vs rolling baseline
    trades_per_day: dict[Any, list[JournalTrade]] = defaultdict(list)
    for t in trades_by_open:
        trades_per_day[t.opened_at.date()].append(t)
    ordered_days = sorted(trades_per_day.keys())
    rolling_counts: list[int] = []
    overtrade_days = 0
    for day in ordered_days:
        day_trades = trades_per_day[day]
        count = len(day_trades)
        if len(rolling_counts) >= 5:
            baseline = float(np.median(rolling_counts[-14:]))
            threshold = max(float(overtrading_min_trades), overtrading_multiplier * max(1.0, baseline))
            if count >= threshold:
                overtrade_days += 1
                episodes.append(
                    BehaviourEpisode(
                        episode_type="overtrading",
                        timestamp=day_trades[-1].opened_at,
                        symbol="MULTI",
                        severity="HIGH" if count >= threshold * 1.5 else "MEDIUM",
                        description=(
                            f"Placed {count} trades on {day.isoformat()} vs rolling baseline of {baseline:.1f}/day."
                        ),
                        metrics={"daily_trades": count, "baseline": round(baseline, 1)},
                    )
                )
        rolling_counts.append(count)

    # 5c. Size escalation after losing streaks. The streak only counts trades already CLOSED when the
    # new trade was opened (no lookahead into outcomes the trader could not have known yet).
    consec_losses: list[JournalTrade] = []
    close_ptr = 0
    for t in trades_by_open:
        while close_ptr < n_trades and sorted_trades[close_ptr].closed_at <= t.opened_at:
            c = sorted_trades[close_ptr]
            if c is t:
                break
            if c.pnl < 0:
                consec_losses.append(c)
            elif c.pnl > 0:
                consec_losses = []
            close_ptr += 1
        if len(consec_losses) >= escalation_streak_len:
            baseline_notional = float(np.median([x.entry_notional for x in consec_losses]))
            ratio = t.entry_notional / max(1e-9, baseline_notional)
            if ratio >= escalation_size_ratio:
                episodes.append(
                    BehaviourEpisode(
                        episode_type="size_escalation",
                        timestamp=t.opened_at,
                        symbol=t.symbol,
                        severity="HIGH" if ratio >= 2.0 or len(consec_losses) >= 3 else "MEDIUM",
                        description=(
                            f"After {len(consec_losses)} consecutive losses, sized {t.symbol} at "
                            f"${t.entry_notional:,.0f} ({ratio:.2f}x streak baseline)."
                        ),
                        metrics={"loss_streak": len(consec_losses), "size_ratio": round(ratio, 2)},
                    )
                )

    # 5d. Disposition effect: losers held substantially longer than winners
    avg_win_hold = float(np.mean([t.hold_minutes for t in wins])) if wins else 0.0
    avg_loss_hold = float(np.mean([t.hold_minutes for t in losses])) if losses else 0.0
    disp_ratio = (avg_loss_hold / avg_win_hold) if avg_win_hold > 1e-6 else 1.0
    if len(wins) >= 2 and len(losses) >= 2 and disp_ratio >= disposition_ratio_threshold:
        worst_loser = max(losses, key=lambda t: t.hold_minutes)
        episodes.append(
            BehaviourEpisode(
                episode_type="disposition_effect",
                timestamp=worst_loser.closed_at,
                symbol=worst_loser.symbol,
                severity="HIGH" if disp_ratio >= 2.2 else "MEDIUM",
                description=(
                    f"Losers held {avg_loss_hold:.0f}m avg vs winners {avg_win_hold:.0f}m avg "
                    f"({disp_ratio:.2f}x disposition ratio; worst {worst_loser.symbol} held {worst_loser.hold_minutes:.0f}m)."
                ),
                metrics={"avg_win_hold_min": round(avg_win_hold, 1), "avg_loss_hold_min": round(avg_loss_hold, 1), "ratio": round(disp_ratio, 2)},
            )
        )

    # 5e. Tilt sessions: 4h clusters of >=3 losses + behavioral flags or severe loss
    window_delta = timedelta(hours=tilt_window_hours)
    flagged_times = [
        e.timestamp for e in episodes if e.episode_type in ("revenge_trading", "size_escalation", "stop_violation")
    ]
    last_tilt_end: datetime | None = None
    for i, start_trade in enumerate(sorted_trades):
        if last_tilt_end is not None and start_trade.closed_at <= last_tilt_end:
            continue
        w_end = start_trade.closed_at + window_delta
        win_trades = [t for t in sorted_trades[i:] if t.closed_at <= w_end]
        win_losses = [t for t in win_trades if t.pnl < 0]
        if len(win_losses) < tilt_min_losses:
            continue
        win_pnl = sum(t.pnl for t in win_trades)
        win_flags = sum(1 for ts in flagged_times if start_trade.opened_at <= ts <= w_end)
        loss_pct = max(0.0, -win_pnl / max(1.0, starting_equity) * 100.0)
        if win_flags >= 2 or loss_pct >= tilt_loss_pct_equity:
            last_tilt_end = w_end
            episodes.append(
                BehaviourEpisode(
                    episode_type="tilt_session",
                    timestamp=win_losses[-1].closed_at,
                    symbol="MULTI",
                    severity="HIGH",
                    description=(
                        f"Tilt cluster ({len(win_losses)} losses, {win_flags} rule/size flags, "
                        f"{'+' if (win_pnl) >= 0 else '-'}${abs(win_pnl):,.2f} net PnL inside {tilt_window_hours:.0f}h window)."
                    ),
                    metrics={"losses_in_window": len(win_losses), "flags_in_window": win_flags, "window_pnl_usd": round(win_pnl, 2)},
                )
            )

    episodes.sort(key=lambda e: e.timestamp)
    ep_counts: dict[str, int] = defaultdict(int)
    for e in episodes:
        ep_counts[e.episode_type] += 1

    # 6. Discipline Score (0-100) with transparent weighted components
    with_stop_count = sum(1 for t in sorted_trades if t.planned_stop is not None)
    stop_coverage = with_stop_count / n_trades
    stop_viol_rate = ep_counts.get("stop_violation", 0) / max(1, with_stop_count)
    stop_score = max(0.0, 25.0 * (0.4 * stop_coverage + 0.6 * max(0.0, 1.0 - 4.0 * stop_viol_rate)))

    size_flags = ep_counts.get("revenge_trading", 0) + ep_counts.get("size_escalation", 0)
    size_rate = size_flags / n_trades
    size_score = max(0.0, 25.0 * (1.0 - min(1.0, size_rate * 8.0)))

    active_days = max(1, len(ordered_days))
    pace_penalty = (overtrade_days / active_days) * 0.6 + (ep_counts.get("tilt_session", 0) * 0.12)
    pace_score = max(0.0, 20.0 * (1.0 - min(1.0, pace_penalty * 2.0)))

    disp_excess = max(0.0, disp_ratio - 1.2)
    disp_score = max(0.0, 15.0 * (1.0 - min(1.0, disp_excess / 1.8)))

    if r_multiples:
        catastrophic_r = sum(1 for r in r_multiples if r <= -2.0) / len(r_multiples)
    else:
        catastrophic_r = 0.0
    tail_score = max(0.0, 15.0 * (1.0 - min(1.0, catastrophic_r * 6.0)))

    total_discipline = float(round(min(100.0, max(0.0, stop_score + size_score + pace_score + disp_score + tail_score)), 1))

    components = {
        "Stop Adherence": {"score": round(stop_score, 1), "max": 25.0, "detail": f"{with_stop_count}/{n_trades} trades with stops, {ep_counts.get('stop_violation', 0)} stop violations."},
        "Size & Revenge Control": {"score": round(size_score, 1), "max": 25.0, "detail": f"{ep_counts.get('revenge_trading', 0)} revenge + {ep_counts.get('size_escalation', 0)} escalation episodes."},
        "Pace & Tilt Control": {"score": round(pace_score, 1), "max": 20.0, "detail": f"{overtrade_days} overtrading days, {ep_counts.get('tilt_session', 0)} tilt sessions."},
        "Disposition Control": {"score": round(disp_score, 1), "max": 15.0, "detail": f"Loser/winner hold ratio {disp_ratio:.2f}x."},
        "Tail Loss Consistency": {"score": round(tail_score, 1), "max": 15.0, "detail": f"{catastrophic_r * 100:.1f}% of R-tracked trades lost >= 2R."},
    }

    return JournalAnalytics(
        total_trades=n_trades,
        win_count=len(wins),
        loss_count=len(losses),
        win_rate=float(win_rate),
        total_pnl_usd=float(total_pnl),
        total_fees_usd=float(total_fees),
        avg_win_usd=float(avg_win),
        avg_loss_usd=float(avg_loss),
        payoff_ratio=float(payoff_ratio),
        expectancy_usd=float(expectancy),
        profit_factor=float(profit_factor),
        avg_r=avg_r,
        expectancy_r=avg_r,
        r_count=len(r_multiples),
        max_drawdown_pct=float(max_dd_pct),
        max_drawdown_usd=float(max_dd_usd),
        current_drawdown_pct=float(current_dd_pct),
        max_drawdown_duration_hours=float(max_dd_dur_h),
        current_drawdown_duration_hours=float(current_dd_dur_h),
        avg_win_hold_minutes=float(avg_win_hold),
        avg_loss_hold_minutes=float(avg_loss_hold),
        disposition_ratio=float(disp_ratio),
        today_pnl_usd=float(today_pnl),
        week_pnl_usd=float(week_pnl),
        reference_date=ref_date.isoformat(),
        discipline_score=total_discipline,
        discipline_grade=_discipline_grade(total_discipline),
        discipline_components=components,
        episodes=episodes,
        episode_counts=dict(ep_counts),
        r_multiples=[round(r, 4) for r in r_multiples],
        by_symbol=by_symbol,
        by_weekday=by_weekday,
        by_hour=by_hour,
        equity_curve=equity_curve,
    )
