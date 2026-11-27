"""Tests for the RiskGuard CLI entrypoint."""

from __future__ import annotations

from pathlib import Path

from riskguard.cli import format_cli_report
from riskguard.models import AccountState

ROOT = Path(__file__).resolve().parents[1]


def test_cli_check_detects_planted_demo_dangers() -> None:
    report, code = format_cli_report(
        account=AccountState(equity=10_000.0),
        positions_path=ROOT / "examples" / "positions_demo.csv",
        mode="demo",
        journal_path=ROOT / "examples" / "journal_demo.csv",
        rules_path=ROOT / "examples" / "rules_demo.yaml",
    )
    assert code == 0
    assert "DOGEUSDT" in report
    assert "beyond liquidation" in report
    assert "XRPUSDT" in report
    assert "no stop-loss" in report
    assert "BREACH" in report
