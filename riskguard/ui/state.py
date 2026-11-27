"""Session state management, demo loading, and cached analytics pipeline."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Sequence

import pandas as pd
import streamlit as st

from riskguard.ai.tools import CoachContext
from riskguard.analytics.drawdown import compute_drawdown_plan, estimate_risk_of_ruin
from riskguard.analytics.journal import analyze_journal, journal_reference_time
from riskguard.analytics.portfolio import analyze_portfolio
from riskguard.analytics.positions import analyze_positions
from riskguard.analytics.rules import evaluate_rules
from riskguard.data.providers import (
    DEFAULT_DEMO_SYMBOLS,
    MarketDataBundle,
    MarketDataProvider,
    SymbolResolution,
)
from riskguard.io import (
    load_journal_csv,
    load_positions_csv,
    load_rules_yaml,
    positions_to_dataframe,
)
from riskguard.models import AccountState, JournalTrade, PositionInput, RuleConfig

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEMO_POSITIONS_PATH = PROJECT_ROOT / "examples" / "positions_demo.csv"
DEMO_JOURNAL_PATH = PROJECT_ROOT / "examples" / "journal_demo.csv"
DEMO_RULES_PATH = PROJECT_ROOT / "examples" / "rules_demo.yaml"


@st.cache_resource
def get_market_provider() -> MarketDataProvider:
    """Shared cached MarketDataProvider instance across Streamlit reruns."""
    return MarketDataProvider()


def resolve_other_symbol(raw: str, mode: str) -> SymbolResolution:
    """Normalize and validate an arbitrary Binance USDT-M symbol for Live or Demo mode."""
    provider = get_market_provider()
    prov_mode = "demo" if mode == "Demo" else "auto"
    return provider.resolve_symbol(raw, mode=prov_mode)


@st.cache_data(ttl=60, show_spinner=False)
def fetch_market_bundle_cached(
    symbols_tuple: tuple[str, ...],
    mode: str,
    lookback_days: int = 120,
) -> MarketDataBundle:
    """Fetch market data with Streamlit TTL caching."""
    provider = get_market_provider()
    prov_mode = "demo" if mode == "Demo" else "auto"
    return provider.get_market_data(list(symbols_tuple), mode=prov_mode, lookback_days=lookback_days)


def load_demo_defaults() -> None:
    """Reset session state to bundled demo positions, synthetic journal, and default rules."""
    positions = load_positions_csv(DEMO_POSITIONS_PATH) if DEMO_POSITIONS_PATH.exists() else []
    df_pos = positions_to_dataframe(positions)
    trades = load_journal_csv(DEMO_JOURNAL_PATH) if DEMO_JOURNAL_PATH.exists() else []
    rules = load_rules_yaml(DEMO_RULES_PATH) if DEMO_RULES_PATH.exists() else RuleConfig()

    st.session_state["equity"] = 10_000.0
    # Drop the sidebar equity widget's own state so it re-initialises from the reset equity on the next run
    st.session_state.pop("sidebar_equity_input", None)
    st.session_state["positions_df"] = df_pos.copy()
    st.session_state["demo_positions_fp"] = df_pos.to_json()
    st.session_state["positions_editor_base"] = df_pos.copy()
    st.session_state["positions_editor_ver"] = st.session_state.get("positions_editor_ver", 0) + 1
    st.session_state["journal_trades"] = trades
    st.session_state["journal_is_synthetic"] = any((t.source or "").upper() == "SYNTHETIC" for t in trades)
    st.session_state["rules"] = rules
    st.session_state["ai_history"] = []
    st.session_state["ai_display_messages"] = []
    st.session_state["ai_tool_logs"] = []
    st.session_state["_coach_running"] = False
    st.session_state.pop("pt_other_symbol", None)
    st.session_state.pop("last_pre_trade", None)
    # The demo positions were opened at the bundled sample's prices (Oct 2025): value them with that same
    # offline sample. Valuing them at today's live prices would show fake losses / liquidations.
    st.session_state["data_mode"] = "Demo"
    st.session_state.pop("sidebar_data_mode", None)


def init_session_state() -> None:
    """Initialize session state keys on first visit (demo positions + Demo market data)."""
    if "initialized" in st.session_state:
        return
    st.session_state["anthropic_api_key"] = ""
    st.session_state["positions_editor_ver"] = 0
    load_demo_defaults()
    if os.getenv("RG_DATA_MODE", "").strip().lower() in ("auto", "live"):
        st.session_state["data_mode"] = "Live auto"
    # Start clean on version 0 so Page 2 editor key is predictable ("positions_editor")
    st.session_state["positions_editor_ver"] = 0
    st.session_state["initialized"] = True


def _journal_fingerprint(trades: Sequence[JournalTrade]) -> str:
    """Content hash of the loaded journal (so replacing it with a same-length journal is detected)."""
    h = hashlib.sha256()
    for t in trades:
        h.update(
            f"{t.opened_at.isoformat()}|{t.closed_at.isoformat()}|{t.symbol}|{t.side}|{t.quantity}|"
            f"{t.entry_price}|{t.exit_price}|{t.planned_stop}|{t.fees}|{t.pnl}|{t.source}\n".encode("utf-8")
        )
    return h.hexdigest()[:16]


def _digest_state(
    equity: float,
    positions_df: pd.DataFrame,
    rules: RuleConfig,
    journal_fp: str,
    market_source: str,
    market_ts: str,
    today: str,
    market_symbols: tuple[str, ...] = (),
    missing_symbols: tuple[str, ...] = (),
) -> str:
    payload = {
        "equity": equity,
        "pos": positions_df.to_json(orient="records"),
        "rules": rules.model_dump(),
        "journal": journal_fp,
        "m_src": market_source,
        "m_ts": market_ts,
        "today": today,
        "m_syms": list(market_symbols),
        "m_miss": list(missing_symbols),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def _format_validation_error(exc: Exception) -> str:
    errors = getattr(exc, "errors", None)
    if callable(errors):
        try:
            parts = [f"{'.'.join(str(x) for x in e.get('loc', ())) or 'row'}: {e.get('msg', '')}" for e in errors()]
            return "; ".join(parts)
        except Exception:  # pragma: no cover - defensive formatting only
            pass
    return str(exc)


def get_parsed_positions() -> list[PositionInput]:
    """Parse the positions table. Blank draft rows (no symbol / entry / quantity yet) are skipped;
    rows that fail validation are excluded AND recorded in st.session_state['positions_errors'] so the
    UI can show which row was dropped and why (never silently)."""
    df: pd.DataFrame = st.session_state.get("positions_df", pd.DataFrame())
    errors: list[str] = []
    valid_rows: list[PositionInput] = []
    if df is not None and not df.empty:
        for row_no, raw in enumerate(df.to_dict(orient="records"), start=1):
            if not raw.get("symbol") or pd.isna(raw.get("entry_price")) or pd.isna(raw.get("quantity")):
                continue
            try:
                cleaned = {k: (None if pd.isna(v) else v) for k, v in raw.items()}
                valid_rows.append(PositionInput.model_validate(cleaned))
            except Exception as exc:  # pydantic ValidationError or bad value conversion
                errors.append(f"Row {row_no} ({raw.get('symbol')}): {_format_validation_error(exc)}")
    st.session_state["positions_errors"] = errors
    return valid_rows


def refresh_market_cache() -> None:
    """Clear market caches so the next fetch pulls fresh live prices."""
    fetch_market_bundle_cached.clear()
    get_market_provider().cache.clear()
    st.session_state.pop("_ctx_cache_digest", None)


def sync_sidebar_widgets() -> None:
    """Copy the sidebar widget values into the canonical session keys BEFORE analytics run.

    Pages compute their analytics before the sidebar is drawn; without this, a changed equity / data
    mode / API key would only take effect one rerun later. Keyed widget state is already updated when
    the rerun starts, so reading it here gives the value the user just entered.
    """
    if "sidebar_equity_input" in st.session_state:
        st.session_state["equity"] = float(st.session_state["sidebar_equity_input"])
    if "sidebar_data_mode" in st.session_state:
        st.session_state["data_mode"] = st.session_state["sidebar_data_mode"]
    if "sidebar_api_key_input" in st.session_state:
        st.session_state["anthropic_api_key"] = str(st.session_state["sidebar_api_key_input"] or "").strip()


def build_session_context(extra_symbols: Sequence[str] = ()) -> CoachContext:
    """Compute or return memoized full analytics CoachContext for the current session."""
    init_session_state()
    sync_sidebar_widgets()

    equity = float(st.session_state.get("equity", 10_000.0))
    account = AccountState(equity=equity)
    positions = get_parsed_positions()
    rules: RuleConfig = st.session_state.get("rules", RuleConfig())
    journal_trades: Sequence[JournalTrade] = st.session_state.get("journal_trades", [])

    symbols = tuple(sorted({p.symbol for p in positions} | set(DEFAULT_DEMO_SYMBOLS) | set(extra_symbols)))
    data_mode = st.session_state.get("data_mode", "Live auto")
    market = fetch_market_bundle_cached(symbols, data_mode, lookback_days=120)

    journal_fp = _journal_fingerprint(journal_trades)
    reference_time = journal_reference_time(journal_trades)
    today = reference_time.date().isoformat() if reference_time is not None else "journal-last-trade"
    digest = _digest_state(
        equity=equity,
        positions_df=st.session_state["positions_df"],
        rules=rules,
        journal_fp=journal_fp,
        market_source=market.source,
        market_ts=market.fetched_at.isoformat(),
        today=today,
        market_symbols=tuple(sorted(market.snapshots.keys())),
        missing_symbols=tuple(sorted(market.missing_symbols)),
    )
    if st.session_state.get("_ctx_cache_digest") == digest and "_ctx_cache" in st.session_state:
        return st.session_state["_ctx_cache"]

    analyses = analyze_positions(positions, account, market=market)
    portfolio = analyze_portfolio(analyses, account, market=market)

    # Memoize journal analysis separately so position edits don't recompute the journal.
    # Real journals are evaluated against the actual current time ("today"); the synthetic demo
    # journal is evaluated as of its last trade (see journal_reference_time).
    j_digest = f"{equity:.2f}:{journal_fp}:{today}"
    if st.session_state.get("_journal_digest") == j_digest and "_journal_cache" in st.session_state:
        journal = st.session_state["_journal_cache"]
    else:
        journal = (
            analyze_journal(journal_trades, starting_equity=equity, reference_time=reference_time)
            if journal_trades
            else None
        )
        st.session_state["_journal_digest"] = j_digest
        st.session_state["_journal_cache"] = journal

    rule_evals, cb = evaluate_rules(analyses, portfolio, account, rules, journal=journal)
    dd_plan = compute_drawdown_plan(
        current_equity=equity,
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
        starting_equity=equity,
        simulations=1500,
    )

    ctx = CoachContext(
        account=account,
        positions=positions,
        analyses=analyses,
        portfolio=portfolio,
        rules=rules,
        rule_evals=rule_evals,
        circuit_breaker=cb,
        journal=journal,
        market=market,
        drawdown_plan=dd_plan,
        risk_of_ruin=ror,
    )
    st.session_state["_ctx_cache_digest"] = digest
    st.session_state["_ctx_cache"] = ctx
    return ctx


def update_positions_from_editor(edited_df: pd.DataFrame) -> None:
    """Persist edits from st.data_editor into canonical session state."""
    st.session_state["positions_df"] = edited_df.copy()
    st.session_state.pop("_ctx_cache_digest", None)


def apply_uploaded_positions(positions: list[PositionInput]) -> None:
    """Load uploaded CSV/JSON positions and bump editor version cleanly."""
    df = positions_to_dataframe(positions)
    st.session_state["positions_df"] = df.copy()
    st.session_state["positions_editor_base"] = df.copy()
    st.session_state["positions_editor_ver"] = st.session_state.get("positions_editor_ver", 0) + 1
    st.session_state.pop("_ctx_cache_digest", None)


def is_new_upload(slot: str, uploaded_file) -> bool:
    """True only the first time a given uploaded file is seen for this uploader `slot`.

    st.file_uploader keeps returning the same file on every rerun; applying it (and calling st.rerun())
    each time would loop forever and keep overwriting later edits. Remember the processed file id.
    """
    if uploaded_file is None:
        return False
    file_key = getattr(uploaded_file, "file_id", None) or f"{uploaded_file.name}:{uploaded_file.size}"
    state_key = f"_processed_upload_{slot}"
    if st.session_state.get(state_key) == file_key:
        return False
    st.session_state[state_key] = file_key
    return True
