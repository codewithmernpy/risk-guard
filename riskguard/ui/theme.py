"""Shared Crypto Quant Suite dark theme, sidebar controls, badges, and footer."""

from __future__ import annotations

from typing import Literal

import streamlit as st

from riskguard.data.providers import MarketDataBundle

SUITE_CSS = """
<style>
    .rg-card {
        background: #131A2E;
        border: 1px solid rgba(99, 102, 241, 0.22);
        border-radius: 10px;
        padding: 1rem 1.15rem;
        margin-bottom: 0.75rem;
    }
    .rg-badge {
        display: inline-block;
        padding: 0.22rem 0.65rem;
        border-radius: 999px;
        font-weight: 700;
        font-size: 0.80rem;
        letter-spacing: 0.03em;
    }
    .rg-badge-ok { background: rgba(16, 185, 129, 0.16); color: #34D399; border: 1px solid rgba(16, 185, 129, 0.45); }
    .rg-badge-warn { background: rgba(245, 158, 11, 0.18); color: #FBBF24; border: 1px solid rgba(245, 158, 11, 0.50); }
    .rg-badge-danger { background: rgba(239, 68, 68, 0.18); color: #F87171; border: 1px solid rgba(239, 68, 68, 0.55); }
    .rg-badge-demo { background: rgba(99, 102, 241, 0.20); color: #A5B4FC; border: 1px solid rgba(99, 102, 241, 0.50); }
    .rg-footer {
        margin-top: 2.5rem;
        padding-top: 1rem;
        border-top: 1px solid rgba(230, 233, 242, 0.12);
        font-size: 0.85rem;
        color: #94A3B8;
        text-align: center;
    }
    .rg-footer a {
        color: #818CF8;
        text-decoration: none;
        margin: 0 0.25rem;
    }
    [data-testid="stMetricLabel"],
    [data-testid="stMetricLabel"] * {
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: clip !important;
    }
    [data-testid="stMetricValue"] {
        font-size: clamp(1.1rem, 2.2vw, 1.8rem);
    }
</style>
"""


def configure_page(title: str, icon: str = "🛡️") -> None:
    """Configure wide dark layout and inject shared suite CSS."""
    try:
        st.set_page_config(
            page_title=f"RiskGuard · {title}",
            page_icon=icon,
            layout="wide",
            initial_sidebar_state="expanded",
        )
    except st.errors.StreamlitAPIException as exc:
        if "set_page_config" not in str(exc) and "once" not in str(exc):
            raise
    st.markdown(SUITE_CSS, unsafe_allow_html=True)


def status_badge_html(label: str, level: Literal["OK", "WARN", "DANGER", "DEMO"]) -> str:
    """Render a pill-shaped traffic-light or mode badge."""
    cls_map = {
        "OK": "rg-badge-ok",
        "WARN": "rg-badge-warn",
        "DANGER": "rg-badge-danger",
        "DEMO": "rg-badge-demo",
    }
    return f'<span class="rg-badge {cls_map[level]}">{label}</span>'


def render_data_source_badge(market: MarketDataBundle) -> None:
    """Display active data source status badge in sidebar."""
    if market.source == "binance":
        html = status_badge_html(market.badge_label, "OK")
    elif market.source == "hyperliquid":
        html = status_badge_html(market.badge_label, "WARN")
    else:
        html = status_badge_html(market.badge_label, "DEMO")
    st.markdown(html, unsafe_allow_html=True)
    st.caption(f"{market.reason} · As of {market.fetched_at.strftime('%Y-%m-%d %H:%M UTC')}")


def _positions_are_demo() -> bool:
    """True while the loaded positions are still the unmodified bundled demo positions."""
    df = st.session_state.get("positions_df")
    fp = st.session_state.get("demo_positions_fp")
    return df is not None and fp is not None and df.to_json() == fp


def render_sidebar(
    *,
    market: MarketDataBundle | None = None,
    on_load_demo: callable | None = None,
    on_refresh_market: callable | None = None,
) -> None:
    """Render shared sidebar across all RiskGuard pages."""
    with st.sidebar:
        st.markdown("## 🛡️ RiskGuard")
        st.caption("Your risk desk and discipline coach.")
        st.divider()

        options = ["Live auto", "Demo"]
        current_mode = st.session_state.get("data_mode", "Live auto")
        idx = options.index(current_mode) if current_mode in options else 0
        chosen_mode = st.radio(
            "Market Data Mode",
            options=options,
            index=idx,
            key="sidebar_data_mode",
            help="Live auto queries Binance USDT-M -> Hyperliquid -> Demo sample. Demo stays 100% offline.",
        )
        st.session_state["data_mode"] = chosen_mode
        if chosen_mode == "Live auto" and _positions_are_demo():
            st.warning(
                "The demo positions were opened at Oct-2025 sample prices. Live mode values them at TODAY's prices, "
                "so their PnL / liquidation numbers are not meaningful. Use Demo mode for the demo, or enter your "
                "own positions for live analysis."
            )

        equity_val = st.number_input(
            "Account Equity (USDT)",
            min_value=100.0,
            max_value=100_000_000.0,
            value=float(st.session_state.get("equity", 10_000.0)),
            step=500.0,
            key="sidebar_equity_input",
        )
        st.session_state["equity"] = float(equity_val)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Load Demo Data", key="sidebar_load_demo_btn", width="stretch"):
                if on_load_demo is not None:
                    on_load_demo()
                st.rerun()
        with col2:
            if st.button("Refresh Feed", key="sidebar_refresh_btn", width="stretch"):
                if on_refresh_market is not None:
                    on_refresh_market()
                st.rerun()

        st.divider()
        api_key = st.text_input(
            "Anthropic API Key (Session Only)",
            value=st.session_state.get("anthropic_api_key", ""),
            type="password",
            key="sidebar_api_key_input",
            placeholder="sk-ant-... (optional; offline coach works without key)",
            help="Stored in session memory only. Never written to disk.",
        )
        st.session_state["anthropic_api_key"] = api_key.strip()

        if market is not None:
            st.divider()
            st.markdown("**Data Source**")
            render_data_source_badge(market)


def render_footer() -> None:
    """Render shared Crypto Quant Suite footer."""
    st.markdown(
        '<div class="rg-footer">'
        "<strong>RiskGuard</strong> · Crypto Quant Suite"
        "<br/><span>Read-only risk analytics. Educational use only — not financial advice.</span>"
        "</div>",
        unsafe_allow_html=True,
    )
