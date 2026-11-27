"""RiskGuard — Page 2: Editable Positions, Liquidation & Per-Position Risk."""

from __future__ import annotations

import math

import pandas as pd
import streamlit as st

from riskguard.analytics.positions import effective_leverage_label, liquidation_label
from riskguard.io import load_positions_csv, load_positions_json, positions_to_dataframe
from riskguard.ui.state import (
    apply_uploaded_positions,
    build_session_context,
    init_session_state,
    is_new_upload,
    load_demo_defaults,
    refresh_market_cache,
    update_positions_from_editor,
)
from riskguard.ui.theme import configure_page, render_footer, render_sidebar

configure_page("Positions", icon="📌")
init_session_state()

st.title("📌 Positions & Liquidation Risk")
st.caption("Edit positions inline, upload CSV/JSON, and inspect Binance USDT-M liquidation prices, ATR buffers, funding costs, and stop safety.")

# Upload / Paste controls
with st.expander("Import Positions (CSV / JSON) or Export", expanded=False):
    up_col1, up_col2 = st.columns(2)
    with up_col1:
        uploaded_file = st.file_uploader("Upload Positions CSV or JSON", type=["csv", "json"], key="pos_uploader")
        if is_new_upload("positions", uploaded_file):
            try:
                raw_bytes = uploaded_file.getvalue()
                if uploaded_file.name.lower().endswith(".json"):
                    parsed = load_positions_json(raw_bytes)
                else:
                    parsed = load_positions_csv(raw_bytes)
                apply_uploaded_positions(parsed)
                st.success(f"Loaded {len(parsed)} position(s) from {uploaded_file.name}.")
                st.rerun()
            except Exception as exc:
                st.error(f"Could not parse uploaded positions: {exc}")

    with up_col2:
        pasted = st.text_area(
            "Or paste CSV / JSON positions",
            height=110,
            key="pos_paste_area",
            placeholder='[{"symbol": "BTCUSDT", "side": "long", "entry_price": 100000, "quantity": 0.05, "leverage": 5, "margin_mode": "isolated", "stop_price": 96000}]',
        )
        if st.button("Apply Pasted Positions", key="apply_pasted_pos_btn"):
            try:
                txt = pasted.strip()
                parsed = load_positions_json(txt) if txt.startswith(("[", "{")) else load_positions_csv(txt)
                apply_uploaded_positions(parsed)
                st.success(f"Loaded {len(parsed)} pasted position(s).")
                st.rerun()
            except Exception as exc:
                st.error(f"Could not parse pasted positions: {exc}")

st.subheader("Live Editable Portfolio Table")
ver = st.session_state.get("positions_editor_ver", 0)
editor_key = "positions_editor" if ver == 0 else f"positions_editor_{ver}"
if editor_key not in st.session_state:
    st.session_state["positions_editor_base"] = st.session_state["positions_df"].copy()

edited_df = st.data_editor(
    st.session_state["positions_editor_base"],
    key=editor_key,
    num_rows="dynamic",
    width="stretch",
    column_config={
        "symbol": st.column_config.TextColumn("Symbol", required=True),
        "side": st.column_config.SelectboxColumn("Side", options=["long", "short"], required=True),
        "entry_price": st.column_config.NumberColumn("Entry Price", min_value=1e-9, format="%.6f", required=True),
        "quantity": st.column_config.NumberColumn("Quantity", min_value=0.000001, format="%.6f", required=True),
        "leverage": st.column_config.NumberColumn("Leverage", min_value=1.0, max_value=150.0, step=1.0, format="%.1f", required=True),
        "margin_mode": st.column_config.SelectboxColumn("Margin Mode", options=["isolated", "cross"], required=True),
        "stop_price": st.column_config.NumberColumn("Stop Price", min_value=0.0, format="%.6f"),
        "take_profit": st.column_config.NumberColumn("Take Profit", min_value=0.0, format="%.6f"),
        "opened_at": st.column_config.TextColumn("Opened At (UTC ISO)"),
    },
)
update_positions_from_editor(edited_df)

ctx = build_session_context()
render_sidebar(
    market=ctx.market,
    on_load_demo=load_demo_defaults,
    on_refresh_market=refresh_market_cache,
)

# Rows that could not be validated are excluded from every calculation: say so explicitly
for err in st.session_state.get("positions_errors", []):
    st.error(f"Excluded from analysis — invalid position row. {err}")

# Explicitly list symbols that fell back to entry-price proxy
proxy_symbols = list(dict.fromkeys(a.symbol for a in ctx.analyses if a.used_entry_proxy))
if proxy_symbols:
    if ctx.market is not None and not ctx.market.is_live:
        st.warning(
            f"Entry-price proxy used for: **{', '.join(proxy_symbols)}** "
            "(not in the offline bundled sample — switch Market Data Mode to **Live auto** in the sidebar for live mark price, funding, ATR, and beta)."
        )
    else:
        st.warning(
            f"Entry-price proxy used for: **{', '.join(proxy_symbols)}** "
            "(unavailable on the active live feed; using entry price as mark, 3% ATR proxy, and default funding)."
        )

# Position Warnings
all_warnings = [(a.danger_level, w) for a in ctx.analyses for w in a.warnings]
if all_warnings:
    st.subheader("⚠️ Position Risk Alerts")
    for lvl, msg in all_warnings:
        if msg.startswith("DANGER"):
            st.error(msg)
        elif msg.startswith("WARN"):
            st.warning(msg)
        else:
            st.info(msg)

st.subheader("Per-Position Liquidation & Risk Analytics")
if ctx.analyses:
    rows = []
    for a in ctx.analyses:
        tier_label = (
            "_DEFAULT (default)"
            if a.margin_tier_source == "_DEFAULT"
            else ("fallback (default)" if a.margin_tier_source == "fallback" else (a.margin_tier_source or "_DEFAULT (default)"))
        )
        mark_source = "Entry proxy" if a.used_entry_proxy else (ctx.market.source if ctx.market is not None else "Feed")
        px_digits = 4 if a.mark_price >= 0.01 else 8
        rows.append(
            {
                "Symbol": a.symbol,
                "Side": a.side.upper(),
                "Mode": a.margin_mode,
                "Lev": a.leverage,
                "Eff. Lev": effective_leverage_label(a),
                "Notional ($)": round(a.notional, 2),
                "Margin Used ($)": round(a.margin_used, 2),
                "Mark ($)": round(a.mark_price, px_digits),
                "Mark Source": mark_source,
                "uPnL ($)": round(a.unrealized_pnl, 2),
                "ROE %": round(a.unrealized_pnl_pct, 2),
                "Stop ($)": round(a.stop_price, px_digits) if a.stop_price else None,
                "Liq Price": liquidation_label(a),
                "Dist to LP %": round(a.distance_to_liq_pct, 2) if (a.distance_to_liq_pct is not None and math.isfinite(a.distance_to_liq_pct)) else None,
                "Dist to LP (ATR)": round(a.distance_to_liq_atr, 2) if (a.distance_to_liq_atr is not None and math.isfinite(a.distance_to_liq_atr)) else None,
                "Risk to Stop": (
                    f"${a.risk_to_stop_usd:,.2f}"
                    if a.risk_to_stop_usd is not None
                    else ("UNBOUNDED (stop crossed)" if a.invalid_stop_direction else "UNBOUNDED (no stop)")
                ),
                "Risk % Eq": round(a.risk_to_stop_pct, 2) if a.risk_to_stop_pct is not None else None,
                "R:R": round(a.reward_to_risk, 2) if a.reward_to_risk is not None else None,
                "Funding/Day ($)": round(a.funding_cost_daily_usd, 2),
                "MMR %": round(a.mmr * 100.0, 2),
                "Margin Tier": tier_label,
                "Alert": "LIQUIDATED" if a.liquidated else a.danger_level,
            }
        )
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    csv_export = positions_to_dataframe(ctx.positions).to_csv(index=False)
    st.download_button(
        "Download Positions CSV",
        data=csv_export,
        file_name="riskguard_positions.csv",
        mime="text/csv",
    )
else:
    st.info("Add at least one valid position above to inspect liquidation & risk metrics.")

render_footer()
