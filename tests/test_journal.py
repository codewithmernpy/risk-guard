"""Tests for journal metrics and positive/negative cases of all 6 behaviour detectors."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from riskguard.analytics.journal import analyze_journal
from riskguard.models import JournalTrade


def _t(
    open_min: float,
    hold_min: float,
    pnl: float,
    *,
    sym: str = "BTCUSDT",
    side: str = "long",
    qty: float = 1.0,
    ep: float = 100.0,
    stop: float | None = 98.0,
    fees: float = 0.0,
) -> JournalTrade:
    base = datetime(2025, 10, 1, 10, 0, tzinfo=timezone.utc)
    opened = base + timedelta(minutes=open_min)
    closed = opened + timedelta(minutes=hold_min)
    sign = 1.0 if side == "long" else -1.0
    xp = ep + sign * ((pnl + fees) / qty)
    return JournalTrade(
        opened_at=opened,
        closed_at=closed,
        symbol=sym,
        side=side,  # type: ignore[arg-type]
        quantity=qty,
        entry_price=ep,
        exit_price=xp,
        planned_stop=stop,
        fees=fees,
        pnl=pnl,
    )


def test_clean_disciplined_journal_has_zero_flags() -> None:
    # 10 disciplined trades spaced hours apart, uniform size, winners held longer than losers, respected stops
    trades = []
    for i in range(10):
        is_win = i % 2 == 0
        pnl = 4.0 if is_win else -2.0
        hold = 120.0 if is_win else 45.0
        trades.append(_t(i * 300.0, hold, pnl, qty=1.0, ep=100.0, stop=98.0))

    res = analyze_journal(trades, starting_equity=1000.0)
    assert res.total_trades == 10
    assert res.win_rate == pytest.approx(0.5)
    assert res.profit_factor == pytest.approx(2.0)
    assert res.episodes == []
    assert res.discipline_score >= 95.0


def test_revenge_trading_positive_and_negative() -> None:
    # Loss closes at t=30m; revenge trade opens at t=38m (8m gap) with 2x size
    pos_trades = [
        _t(0, 30, -2.0, qty=1.0),
        _t(38, 20, -3.0, qty=2.0),
    ]
    res_pos = analyze_journal(pos_trades)
    assert res_pos.episode_counts.get("revenge_trading", 0) == 1

    # Negative: same size (1.0x) OR opened 90 minutes later (> 30m window)
    neg_trades = [
        _t(0, 30, -2.0, qty=1.0),
        _t(38, 20, -2.0, qty=1.0),    # same size
        _t(150, 20, 2.0, qty=2.0),    # 92m after prev loss
    ]
    res_neg = analyze_journal(neg_trades)
    assert res_neg.episode_counts.get("revenge_trading", 0) == 0


def test_overtrading_positive_and_negative() -> None:
    # 6 baseline days with 1 trade/day, then day 7 has 8 trades
    trades = [_t(d * 1440.0, 30.0, 2.0) for d in range(6)]
    for k in range(8):
        trades.append(_t(6 * 1440.0 + k * 60.0, 20.0, 1.0))

    res_pos = analyze_journal(trades)
    assert res_pos.episode_counts.get("overtrading", 0) == 1

    # Negative: 2 trades/day consistently
    neg_trades = [_t(d * 1440.0 + k * 120.0, 30.0, 2.0) for d in range(8) for k in range(2)]
    res_neg = analyze_journal(neg_trades)
    assert res_neg.episode_counts.get("overtrading", 0) == 0


def test_size_escalation_after_losing_streak() -> None:
    # 2 consecutive losses at qty=1.0, then qty=2.0 -> triggers escalation
    pos_trades = [
        _t(0, 30, -2.0, qty=1.0),
        _t(120, 30, -2.0, qty=1.0),
        _t(240, 30, -3.0, qty=2.0),
    ]
    res_pos = analyze_journal(pos_trades)
    assert res_pos.episode_counts.get("size_escalation", 0) == 1

    # Negative: winning trade breaks streak before bigger size
    neg_trades = [
        _t(0, 30, -2.0, qty=1.0),
        _t(120, 30, 3.0, qty=1.0),
        _t(240, 30, -3.0, qty=2.0),
    ]
    res_neg = analyze_journal(neg_trades)
    assert res_neg.episode_counts.get("size_escalation", 0) == 0


def test_disposition_effect_positive_and_negative() -> None:
    # Losers held 300m vs winners held 40m -> ratio 7.5x >= 1.5x
    pos_trades = [
        _t(0, 40, 3.0),
        _t(200, 40, 3.0),
        _t(400, 300, -2.0),
        _t(900, 300, -2.0),
    ]
    res_pos = analyze_journal(pos_trades)
    assert res_pos.episode_counts.get("disposition_effect", 0) == 1
    assert res_pos.disposition_ratio == pytest.approx(7.5)

    # Negative: losers cut quickly (30m), winners run (120m)
    neg_trades = [
        _t(0, 120, 3.0),
        _t(300, 120, 3.0),
        _t(600, 30, -2.0),
        _t(900, 30, -2.0),
    ]
    res_neg = analyze_journal(neg_trades)
    assert res_neg.episode_counts.get("disposition_effect", 0) == 0


def test_stop_violation_and_tilt_session() -> None:
    # Stop planned at 98 (-$2 risk), but exited at 95 (-$5 loss = 2.5R) -> stop_violation!
    # Plus 3 rapid losses inside 2h with revenge/escalation -> tilt_session!
    trades = [
        _t(0, 20, -4.0, qty=1.0, ep=100.0, stop=98.0),    # 2R stop violation
        _t(30, 20, -8.0, qty=2.0, ep=100.0, stop=98.0),   # revenge + 2R stop violation
        _t(60, 20, -12.0, qty=3.0, ep=100.0, stop=98.0),  # size escalation + revenge + stop violation
    ]
    res = analyze_journal(trades, starting_equity=500.0)
    assert res.episode_counts.get("stop_violation", 0) == 3
    assert res.episode_counts.get("tilt_session", 0) == 1
    assert res.discipline_score < 60.0
