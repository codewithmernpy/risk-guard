"""Command-line interface for RiskGuard (python -m riskguard ...)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from riskguard.analytics.drawdown import compute_drawdown_plan, estimate_risk_of_ruin
from riskguard.analytics.journal import analyze_journal, journal_reference_time
from riskguard.analytics.portfolio import analyze_portfolio
from riskguard.analytics.positions import analyze_positions, liquidation_label
from riskguard.analytics.rules import evaluate_rules
from riskguard.data.providers import MarketDataProvider
from riskguard.io import load_journal_csv, load_positions_csv, load_positions_json, load_rules_yaml
from riskguard.models import AccountState, RuleConfig
from riskguard.synthetic import save_demo_journal


# Console-safe text: Windows consoles often use cp1252/cp437, which garble or reject typographic
# characters. The text report is reduced to plain ASCII before printing.
_ASCII_MAP = str.maketrans(
    {
        "—": "-",  # em dash
        "–": "-",  # en dash
        "·": "|",  # middle dot
        "→": "->",
        "≥": ">=",
        "≤": "<=",
        "×": "x",
        "…": "...",
        "‘": "'",
        "’": "'",
        "“": '"',
        "”": '"',
    }
)


def to_console_text(text: str) -> str:
    """Translate common typographic characters to ASCII; replace anything else non-ASCII with '?'."""
    return text.translate(_ASCII_MAP).encode("ascii", errors="replace").decode("ascii")


def _load_positions_auto(path: Path):
    if path.suffix.lower() == ".json":
        return load_positions_json(path)
    return load_positions_csv(path)


def format_cli_report(
    *,
    account: AccountState,
    positions_path: Path,
    mode: str,
    journal_path: Path | None = None,
    rules_path: Path | None = None,
    as_json: bool = False,
) -> tuple[str, int]:
    """Run full risk check and format report. Returns (output_text, exit_code)."""
    # A path the user passed must exist: never fall back to defaults silently
    for label, path in (("positions", positions_path), ("rules", rules_path), ("journal", journal_path)):
        if path is not None and not Path(path).is_file():
            raise FileNotFoundError(f"{label} file not found: {path}")
    positions = _load_positions_auto(positions_path)
    rules = load_rules_yaml(rules_path) if rules_path is not None else RuleConfig()

    journal_trades = load_journal_csv(journal_path) if journal_path is not None else []
    journal = (
        analyze_journal(
            journal_trades,
            starting_equity=account.equity,
            reference_time=journal_reference_time(journal_trades),
        )
        if journal_trades
        else None
    )

    provider = MarketDataProvider()
    symbols = [p.symbol for p in positions] or ["BTCUSDT"]
    market = provider.get_market_data(symbols, mode="demo" if mode == "demo" else "auto")

    analyses = analyze_positions(positions, account, market=market)
    portfolio = analyze_portfolio(analyses, account, market=market)
    rule_evals, cb = evaluate_rules(analyses, portfolio, account, rules, journal=journal)
    dd_plan = compute_drawdown_plan(
        current_equity=account.equity,
        current_dd_pct=journal.current_drawdown_pct if journal else 0.0,
        base_risk_pct=min(1.0, rules.max_risk_per_trade_pct),
        max_dd_limit_pct=rules.max_drawdown_pct,
    )
    ror = estimate_risk_of_ruin(
        win_rate=journal.win_rate if (journal and journal.total_trades >= 5) else 0.50,
        payoff_ratio=journal.payoff_ratio if (journal and journal.payoff_ratio > 0) else 1.5,
        risk_per_trade_pct=dd_plan.recommended_risk_pct or dd_plan.base_risk_pct,
        max_dd_limit_pct=rules.max_drawdown_pct,
        r_multiples=journal.r_multiples if journal else None,
        starting_equity=account.equity,
    )

    if as_json:
        payload = {
            "data_source": {"source": market.source, "badge": market.badge_label, "reason": market.reason},
            "account": account.model_dump(),
            "circuit_breaker": cb.model_dump(),
            "positions": [a.model_dump() for a in analyses],
            "portfolio": portfolio.model_dump(),
            "rules": [r.model_dump() for r in rule_evals],
            "drawdown_plan": dd_plan.model_dump(),
            "risk_of_ruin": ror.model_dump(),
        }
        return json.dumps(payload, indent=2, default=str), 0

    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("RISKGUARD — PORTFOLIO RISK & DISCIPLINE REPORT")
    lines.append("=" * 72)
    lines.append(
        f"Equity: ${account.equity:,.2f} {account.currency} | Positions: {portfolio.position_count} | "
        f"Data: {market.badge_label}"
    )
    lines.append(
        f"Trading Allowed Today: {'YES' if cb.trading_allowed else 'NO (HALTED)'} "
        f"[{cb.status}{', as of ' + cb.reference_date if cb.reference_date else ', no journal'}] | "
        f"Gross: ${portfolio.gross_exposure_usd:,.2f} ({portfolio.gross_leverage:.2f}x) | "
        f"Net: ${portfolio.net_exposure_usd:+,.2f} | BTC-Beta Net: ${portfolio.btc_beta_adj_net_usd:+,.2f}"
    )
    lines.append(
        f"1d 95%/99% Hist VaR: ${portfolio.var_95_hist_usd:,.2f} / ${portfolio.var_99_hist_usd:,.2f} | "
        f"CF VaR 95%/99%: ${portfolio.var_95_cf_usd:,.2f} / ${portfolio.var_99_cf_usd:,.2f} | "
        f"HHI: {portfolio.hhi:.3f} | Max 30d Corr: {portfolio.max_pairwise_corr_30d:.2f}"
    )

    # Planted / active position dangers
    dangers = [w for a in analyses for w in a.warnings if w.startswith("DANGER")]
    lines.append("\n--- CRITICAL DANGERS ---")
    if dangers:
        for d in dangers:
            lines.append(f"  [!] {d}")
    else:
        lines.append("  None detected.")

    lines.append("\n--- POSITIONS ---")
    for a in analyses:
        lp_txt = liquidation_label(a)
        stop_txt = f"${a.stop_price:,.4f}" if a.stop_price is not None else "NONE"
        dist_txt = (
            "LIQUIDATED"
            if a.liquidated
            else (
                f"{a.distance_to_liq_pct:.2f}% ({a.distance_to_liq_atr:.2f} ATR)"
                if a.distance_to_liq_pct is not None and a.distance_to_liq_atr is not None
                else "-"
            )
        )
        if a.risk_to_stop_usd is not None:
            risk_txt = f"${a.risk_to_stop_usd:,.2f} ({a.risk_to_stop_pct:.2f}%)"
        elif a.invalid_stop_direction:
            risk_txt = "UNBOUNDED (stop already crossed)"
        else:
            risk_txt = "UNBOUNDED (no stop)"
        lines.append(
            f"  {a.symbol:<9} {a.side.upper():<5} {a.leverage:>4.1f}x {a.margin_mode:<8} "
            f"Notional=${a.notional:>8,.2f} Mark=${a.mark_price:<10,.4f} uPnL=${a.unrealized_pnl:>+8,.2f} "
            f"Stop={stop_txt:<10} LP={lp_txt:<10} DistLP={dist_txt:<16} Risk={risk_txt} [{a.danger_level}]"
        )

    lines.append("\n--- STRESS SCENARIOS ---")
    for s in portfolio.stress_scenarios:
        liq = ", ".join(s.liquidated_symbols) if s.liquidated_symbols else "None"
        lines.append(
            f"  {s.scenario:<28} PnL=${s.portfolio_pnl_usd:>+9,.2f} ({s.portfolio_pnl_pct:>+6.2f}%) "
            f"Liquidations: {liq}"
        )

    lines.append("\n--- RULES AUDIT ---")
    for r in rule_evals:
        lines.append(f"  [{r.status:<6}] {r.name}: {r.current_value} (limit {r.limit_value}) — {r.explanation}")

    if journal is not None:
        lines.append("\n--- JOURNAL & DISCIPLINE ---")
        lines.append(
            f"  Trades: {journal.total_trades} | Win Rate: {journal.win_rate * 100:.1f}% | "
            f"PF: {journal.profit_factor:.2f} | Discipline Score: {journal.discipline_score:.1f}/100 ({journal.discipline_grade})"
        )
        if journal.episode_counts:
            flags = ", ".join(f"{k}={v}" for k, v in sorted(journal.episode_counts.items()))
            lines.append(f"  Behaviour Flags: {flags}")

    lines.append("\n*Educational analytics only — not financial advice.*")
    return to_console_text("\n".join(lines)), 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="riskguard",
        description="RiskGuard — Crypto Perpetuals Risk Desk & Discipline Coach CLI",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    check_p = sub.add_parser("check", help="Run full risk check on positions CSV/JSON.")
    check_p.add_argument("--positions", required=True, type=Path, help="Path to positions CSV or JSON.")
    check_p.add_argument("--equity", type=float, default=10_000.0, help="Account equity in USDT.")
    check_p.add_argument("--journal", type=Path, default=None, help="Optional closed-trade journal CSV.")
    check_p.add_argument("--rules", type=Path, default=None, help="Optional YAML rules file.")
    check_p.add_argument("--demo", action="store_true", help="Force offline bundled sample data.")
    check_p.add_argument("--json", action="store_true", help="Output structured JSON.")

    gen_p = sub.add_parser("generate-demo", help="Generate synthetic journal CSV (examples/journal_demo.csv).")
    gen_p.add_argument("--out", type=Path, default=Path("examples/journal_demo.csv"), help="Output CSV path.")
    gen_p.add_argument("--trades", type=int, default=400, help="Number of synthetic trades.")
    gen_p.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    # Never crash on a console that cannot encode a character (e.g. cp1252 stdout on Windows)
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")  # type: ignore[union-attr]
        except (AttributeError, ValueError):
            pass

    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.command == "generate-demo":
        try:
            out = save_demo_journal(args.out, num_trades=args.trades, seed=args.seed)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        with open(out, encoding="utf-8") as fh:
            written = sum(1 for _ in fh) - 1  # data rows (header excluded)
        print(f"Generated synthetic journal ({written} trades, seed={args.seed}) at {out}")
        return 0

    if args.command == "check":
        mode = "demo" if args.demo else "auto"
        try:
            account = AccountState(equity=args.equity)
            report, code = format_cli_report(
                account=account,
                positions_path=args.positions,
                mode=mode,
                journal_path=args.journal,
                rules_path=args.rules,
                as_json=args.json,
            )
        except (FileNotFoundError, ValueError) as exc:  # missing file, invalid CSV/YAML/JSON or equity
            print(to_console_text(f"error: {exc}"), file=sys.stderr)
            return 2
        print(report)
        return code

    return 1


if __name__ == "__main__":
    sys.exit(main())
