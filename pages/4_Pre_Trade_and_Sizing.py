"""RiskGuard — Page 4: Pre-Trade Risk Check & Position Sizing Calculator."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from riskguard.analytics.positions import liquidation_label
from riskguard.analytics.sizing import pre_trade_check
from riskguard.data.providers import DEFAULT_DEMO_SYMBOLS, SymbolResolution
from riskguard.models import PositionInput, normalize_symbol
from riskguard.ui.state import (
    build_session_context,
    init_session_state,
    load_demo_defaults,
    refresh_market_cache,
    resolve_other_symbol,
    sync_sidebar_widgets,
)
from riskguard.ui.theme import configure_page, render_footer, render_sidebar, status_badge_html

configure_page("Pre-Trade Check & Sizing", icon="⚖️")
init_session_state()
sync_sidebar_widgets()

data_mode = st.session_state.get("data_mode", "Live auto")
raw_other = str(st.session_state.get("pt_other_symbol", "") or "").strip()
other_resolution: SymbolResolution | None = None
other_error: str | None = None
extra_symbols: tuple[str, ...] = ()

if raw_other:
    try:
        other_resolution = resolve_other_symbol(raw_other, data_mode)
        extra_symbols = (other_resolution.symbol,)
    except ValueError as exc:
        other_error = str(exc)

ctx = build_session_context(extra_symbols=extra_symbols)
render_sidebar(
    market=ctx.market,
    on_load_demo=load_demo_defaults,
    on_refresh_market=refresh_market_cache,
)

st.title("⚖️ Pre-Trade Risk Check & Position Sizing")
st.caption("Simulate adding a new position before clicking buy: check GO / REDUCE SIZE / NO-GO, before/after VaR, open risk, and 4 sizing models.")

avail_syms = sorted(set(DEFAULT_DEMO_SYMBOLS) | {a.symbol for a in ctx.analyses})
default_sym = "BTCUSDT" if "BTCUSDT" in avail_syms else avail_syms[0]

st.subheader("Proposed Trade Parameters")
# Symbol, Other coin, and side live outside the form so the entry / stop defaults follow them immediately.
sel1, sel_other, sel2 = st.columns([1.0, 1.3, 0.9])
with sel1:
    selected_symbol = st.selectbox("Symbol", options=avail_syms, index=avail_syms.index(default_sym), key="pt_symbol")
with sel_other:
    st.text_input(
        "Other coin (optional)",
        key="pt_other_symbol",
        placeholder="e.g. pepe, LINK, 1000PEPEUSDT",
        help="Leave blank to use Symbol dropdown. Accepts any Binance USDT-M perpetual ticker or symbol (e.g. 'pepe' -> 1000PEPEUSDT in Live mode).",
    )
with sel2:
    side = st.selectbox("Side", options=["long", "short"], key="pt_side")

symbol = other_resolution.symbol if other_resolution is not None else selected_symbol

if other_error is not None:
    st.error(other_error)
elif other_resolution is not None:
    if other_resolution.validated_against_live_list:
        if other_resolution.symbol != normalize_symbol(raw_other):
            st.info(
                f"Resolved `{raw_other}` -> **`{symbol}`** (validated against Binance USDT-M contracts). "
                "Clear *Other coin* to use the dropdown."
            )
        else:
            st.caption(f"Using **`{symbol}`** (validated against Binance USDT-M contracts). Clear *Other coin* to use the dropdown.")
    elif other_resolution.used_fallback_format_only:
        st.warning(f"Using **`{symbol}`** ({other_resolution.note})")
    else:
        st.caption(f"Using **`{symbol}`** (offline format validated; Live mode resolves exchange aliases such as `1000PEPEUSDT`).")

snap = ctx.market.snapshots.get(symbol) if ctx.market else None
if other_error is None:
    if snap is not None and snap.mark_price > 0:
        suggested_entry = float(snap.mark_price)
        price_fmt = "%.8f" if suggested_entry < 0.01 else ("%.6f" if suggested_entry < 1 else "%.4f")
        st.caption(f"{symbol} mark ${suggested_entry:,.6g} ({ctx.market.badge_label}). Entry defaults to the mark; edit it to test a limit entry.")
    else:
        suggested_entry = None
        price_fmt = "%.6f"
        if data_mode == "Demo" or (ctx.market is not None and not ctx.market.is_live):
            st.warning(
                f"{symbol} is not in the bundled offline sample — **Live auto** mode is needed for its live mark price, "
                "funding, and candles. You can still enter an entry price manually below to run the analysis with an explicit proxy warning."
            )
        else:
            st.warning(
                f"No market price for {symbol} on the active feed ({ctx.market.badge_label if ctx.market else 'offline'}): "
                "enter the entry price manually (analysis will use entry price and a 3% ATR proxy)."
            )
else:
    suggested_entry = None
    price_fmt = "%.6f"

with st.form("pre_trade_form"):
    c2, c3, c4 = st.columns(3)
    with c2:
        # Keyed per symbol so each symbol keeps its own entry, initialised to its own mark
        entry_price = st.number_input(
            "Entry Price ($)",
            min_value=1e-9,
            value=suggested_entry,
            format=price_fmt,
            key=f"pt_entry_{symbol}",
        )
        notional_usd = st.number_input("Proposed Notional ($)", min_value=10.0, value=2500.0, step=250.0, key="pt_notional")
    with c3:
        leverage = st.number_input("Leverage (x)", min_value=1.0, max_value=125.0, value=5.0, step=1.0, key="pt_leverage")
        margin_mode = st.selectbox("Margin Mode", options=["isolated", "cross"], key="pt_margin_mode")
    with c4:
        default_stop = (
            round(suggested_entry * (0.97 if side == "long" else 1.03), 10) if suggested_entry is not None else None
        )
        stop_price = st.number_input(
            "Stop Loss Price ($, 0=none)",
            min_value=0.0,
            value=default_stop,
            format=price_fmt,
            key=f"pt_stop_{symbol}_{side}",
        )
        take_profit = st.number_input(
            "Take Profit Price ($, 0=none)", min_value=0.0, value=0.0, format=price_fmt, key=f"pt_tp_{symbol}_{side}"
        )

    k_atr = st.slider("Liquidation Safety Buffer (k x Daily ATR beyond stop)", min_value=0.5, max_value=3.0, value=1.5, step=0.25, key="pt_katr")
    submitted = st.form_submit_button("Evaluate Pre-Trade Risk", type="primary")

# Evaluate on submit (and once on first visit as a default preview)
missing_entry_on_submit = (submitted or "last_pre_trade" not in st.session_state) and entry_price is None
if other_error is not None:
    if submitted:
        st.error(f"Fix the 'Other coin' symbol before evaluating: {other_error}")
elif missing_entry_on_submit:
    st.error("Enter an entry price to evaluate the trade.")
elif submitted or "last_pre_trade" not in st.session_state:
    try:
        proposed = PositionInput(
            symbol=symbol,
            side=side,  # type: ignore[arg-type]
            entry_price=entry_price,
            notional=notional_usd,
            leverage=leverage,
            margin_mode=margin_mode,  # type: ignore[arg-type]
            stop_price=stop_price if stop_price else None,
            take_profit=take_profit if take_profit else None,
        )
        res = pre_trade_check(
            proposed=proposed,
            existing_positions=ctx.positions,
            account=ctx.account,
            market=ctx.market,
            rules=ctx.rules,
            journal=ctx.journal,
            k_atr_buffer=k_atr,
        )
        st.session_state["last_pre_trade"] = res
    except Exception as exc:
        st.error(f"Invalid trade inputs: {exc}")

res = st.session_state.get("last_pre_trade")
if res is not None and other_error is None and not (submitted and entry_price is None):
    st.divider()
    pa_in = res.proposed_analysis
    verdict_lvl = "OK" if res.verdict == "GO" else ("WARN" if res.verdict == "REDUCE SIZE" else "DANGER")
    st.markdown(f"### Verdict: {status_badge_html(res.verdict, verdict_lvl)}", unsafe_allow_html=True)
    st.caption(
        f"Evaluated: {pa_in.side.upper()} {pa_in.quantity:.6g} {pa_in.symbol} @ ${pa_in.entry_price:,.6g} "
        f"({pa_in.leverage:.1f}x {pa_in.margin_mode}, stop "
        + (f"${pa_in.stop_price:,.6g}" if pa_in.stop_price is not None else "none")
        + "), as if filled at the entry price."
    )
    if ctx.market is None or pa_in.symbol not in ctx.market.snapshots:
        mode_note = (
            "Live auto mode is needed for its live price, funding, and candles."
            if (ctx.market is not None and not ctx.market.is_live)
            else "Active live feed had no data for this symbol."
        )
        st.warning(
            f"**Proxy Warning:** {pa_in.symbol} has no feed data ({mode_note}) — "
            f"analysis ran on the entered entry price (${pa_in.entry_price:,.6g}) using a 3% ATR proxy "
            f"(${pa_in.atr_1d:,.6g}) and default funding."
        )
    for reason in res.reasons:
        if res.verdict == "NO-GO":
            st.error(reason)
        elif res.verdict == "REDUCE SIZE":
            st.warning(reason)
        else:
            st.success(reason)

    if res.recommended_quantity is not None or res.recommended_leverage is not None:
        rec_parts = []
        if res.recommended_quantity is not None:
            rec_parts.append(f"Max Recommended Quantity: `{res.recommended_quantity:.6g}` (${res.recommended_quantity * res.proposed_analysis.entry_price:,.2f})")
        if res.recommended_leverage is not None:
            rec_parts.append(f"Max Safe Leverage: `{res.recommended_leverage:.2f}x`")
        st.info(" · ".join(rec_parts))

    col_ba, col_pos = st.columns([1.2, 1.0])
    with col_ba:
        st.subheader("Before vs After Portfolio Impact")
        ba_df = pd.DataFrame(
            [
                {"Metric": "1d 95% Hist VaR ($)", "Before": round(res.before_var_95_usd, 2), "After": round(res.after_var_95_usd, 2), "Delta": round(res.after_var_95_usd - res.before_var_95_usd, 2)},
                {"Metric": "1d 99% Hist VaR ($)", "Before": round(res.before_var_99_usd, 2), "After": round(res.after_var_99_usd, 2), "Delta": round(res.after_var_99_usd - res.before_var_99_usd, 2)},
                {"Metric": "Total Open Risk (% Eq)", "Before": round(res.before_open_risk_pct, 2), "After": round(res.after_open_risk_pct, 2), "Delta": round(res.after_open_risk_pct - res.before_open_risk_pct, 2)},
                {"Metric": "Gross Exposure ($)", "Before": round(res.before_gross_exposure_usd, 2), "After": round(res.after_gross_exposure_usd, 2), "Delta": round(res.after_gross_exposure_usd - res.before_gross_exposure_usd, 2)},
                {"Metric": "Net Exposure ($)", "Before": round(res.before_net_exposure_usd, 2), "After": round(res.after_net_exposure_usd, 2), "Delta": round(res.after_net_exposure_usd - res.before_net_exposure_usd, 2)},
                {"Metric": "BTC-Beta Net ($)", "Before": round(res.before_btc_beta_exposure_usd, 2), "After": round(res.after_btc_beta_exposure_usd, 2), "Delta": round(res.after_btc_beta_exposure_usd - res.before_btc_beta_exposure_usd, 2)},
            ]
        )
        st.dataframe(ba_df, width="stretch", hide_index=True)

    with col_pos:
        st.subheader("Proposed Position Metrics")
        pa = res.proposed_analysis
        tier_label = (
            "_DEFAULT (default)"
            if pa.margin_tier_source == "_DEFAULT"
            else ("fallback (default)" if pa.margin_tier_source == "fallback" else (pa.margin_tier_source or "_DEFAULT (default)"))
        )
        pm_df = pd.DataFrame(
            [
                {"Field": "Liquidation Price", "Value": liquidation_label(pa)},
                {"Field": "Dist to Liquidation", "Value": f"{pa.distance_to_liq_pct:.2f}% ({pa.distance_to_liq_atr:.2f} ATR)" if pa.distance_to_liq_pct is not None and pa.distance_to_liq_atr is not None else "-"},
                {"Field": "Stop Beyond Liquidation?", "Value": "YES (DANGER)" if pa.stop_beyond_liquidation else "No"},
                {"Field": "Stop Wrong Side of Entry?", "Value": "YES (DANGER)" if pa.invalid_stop_direction else "No"},
                {"Field": "Risk to Stop", "Value": f"${pa.risk_to_stop_usd:,.2f} ({pa.risk_to_stop_pct:.2f}%)" if pa.risk_to_stop_usd is not None else "Unbounded"},
                {"Field": "Reward / Risk", "Value": f"{pa.reward_to_risk:.2f}R" if pa.reward_to_risk is not None else "N/A"},
                {"Field": "Margin Tier", "Value": f"{pa.mmr * 100.0:.2f}% MMR ({tier_label})"},
                {"Field": "New Rule Breaches", "Value": ", ".join([b for b in res.after_breaches if b not in res.before_breaches]) or "None"},
            ]
        )
        st.dataframe(pm_df, width="stretch", hide_index=True)

    if res.sizing_options:
        st.subheader("📐 Recommended Position Sizes (4 Sizing Models)")
        sz_df = pd.DataFrame(
            [
                {
                    "Method": s.method,
                    "Quantity": round(s.quantity, 6),
                    "Notional ($)": round(s.notional_usd, 2),
                    "Risk ($)": round(s.risk_usd, 2),
                    "Risk % Eq": round(s.risk_pct, 2),
                    "Max Safe Lev": f"{s.max_safe_leverage:.2f}x" if s.max_safe_leverage else "-",
                    "Rationale": s.notes,
                }
                for s in res.sizing_options
            ]
        )
        st.dataframe(sz_df, width="stretch", hide_index=True)

render_footer()
