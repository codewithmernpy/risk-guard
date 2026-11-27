"""Tests for the trading rules engine and daily/weekly/drawdown circuit breaker."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from riskguard.analytics.journal import analyze_journal
from riskguard.analytics.portfolio import analyze_portfolio
from riskguard.analytics.positions import analyze_positions
from riskguard.analytics.rules import evaluate_rules
from riskguard.models import AccountState, JournalTrade, PositionInput, RuleConfig


def test_rules_engine_breaches_and_circuit_breaker_tripped() -> None:
    account = AccountState(equity=10_000.0)
    rules = RuleConfig(
        max_risk_per_trade_pct=2.0,
        max_total_open_risk_pct=5.0,
        max_leverage=10.0,
        daily_loss_limit_pct=3.0,
        weekly_loss_limit_pct=7.0,
        max_drawdown_pct=10.0,
        max_positions=2,
        mandatory_stop=True,
    )

    positions = [
        PositionInput(symbol="SOLUSDT", side="long", entry_price=100.0, quantity=50.0, leverage=20.0, margin_mode="isolated", stop_price=92.0),  # 20x > 10x, stop beyond LP
        PositionInput(symbol="XRPUSDT", side="long", entry_price=2.0, quantity=1000.0, leverage=5.0, margin_mode="isolated"),  # missing stop
        PositionInput(symbol="BTCUSDT", side="long", entry_price=100_000.0, quantity=0.03, leverage=3.0, margin_mode="isolated", stop_price=95_000.0),  # 3 positions > max 2
    ]

    # Craft journal with -4% daily loss (-400 on 10k equity) -> trips daily circuit breaker!
    t0 = datetime(2025, 10, 31, 10, 0, tzinfo=timezone.utc)
    trades = [
        JournalTrade(
            opened_at=t0,
            closed_at=t0 + timedelta(minutes=30),
            symbol="BTCUSDT",
            side="long",
            quantity=0.1,
            entry_price=100_000.0,
            exit_price=96_000.0,
            planned_stop=98_000.0,
            fees=0.0,
            pnl=-400.0,
        )
    ]
    journal = analyze_journal(trades, starting_equity=10_000.0, reference_time=t0 + timedelta(hours=1))

    analyses = analyze_positions(positions, account)
    portfolio = analyze_portfolio(analyses, account)
    evals, cb = evaluate_rules(analyses, portfolio, account, rules, journal=journal)
    by_id = {e.rule_id: e for e in evals}

    assert by_id["mandatory_stop"].status == "BREACH"
    assert by_id["liquidation_safety"].status == "BREACH"
    assert by_id["max_leverage"].status == "BREACH"
    assert by_id["max_positions"].status == "BREACH"
    assert by_id["daily_loss_limit_pct"].status == "BREACH"
    assert cb.trading_allowed is False
    assert cb.status == "TRIPPED"
