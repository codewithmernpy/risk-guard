"""RiskGuard — Page 3: Portfolio Exposure, Correlation, VaR/CVaR & Stress Testing."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from riskguard.ui.charts import (
    chart_correlation_heatmap,
    chart_exposure_bars,
    chart_stress_scenarios,
    chart_var_comparison,
)
from riskguard.ui.state import build_session_context, load_demo_defaults, refresh_market_cache
from riskguard.ui.theme import configure_page, render_footer, render_sidebar

configure_page("Exposure & Stress", icon="📉")
ctx = build_session_context()
render_sidebar(
    market=ctx.market,
    on_load_demo=load_demo_defaults,
    on_refresh_market=refresh_market_cache,
)

p = ctx.portfolio
st.title("📉 Portfolio Exposure, Correlation, VaR & Stress Tests")
st.caption("Gross/net & 90d BTC-beta-adjusted exposure, 30d/90d hourly correlations, HHI concentration, Historical & Cornish-Fisher VaR/CVaR, and crash simulations.")

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Gross Exposure", f"${p.gross_exposure_usd:,.0f}", f"{p.gross_leverage:.2f}x equity", delta_color="off", delta_arrow="off")
m2.metric("Net Exposure", f"{'+' if (p.net_exposure_usd) >= 0 else '-'}${abs(p.net_exposure_usd):,.0f}", f"Long ${p.long_exposure_usd:,.0f} / Short ${p.short_exposure_usd:,.0f}", delta_color="off", delta_arrow="off")
m3.metric("BTC-Beta Net", f"{'+' if (p.btc_beta_adj_net_usd) >= 0 else '-'}${abs(p.btc_beta_adj_net_usd):,.0f}", f"{p.btc_beta_adj_net_pct:+.1f}% equity", delta_color="off", delta_arrow="off")
m4.metric("Concentration (HHI)", f"{p.hhi:.3f}", f"{p.effective_num_positions:.2f} effective bets", delta_color="off", delta_arrow="off")
m5.metric("Worst Case Stops", f"${p.worst_case_stops_usd:,.0f}", f"{p.worst_case_stops_pct:.2f}% equity", delta_color="off", delta_arrow="off")

if not ctx.analyses:
    st.info("No active positions to analyze. Load demo data or add positions on Page 2.")
    render_footer()
    st.stop()

c1, c2 = st.columns(2)
with c1:
    st.plotly_chart(chart_exposure_bars(p), width="stretch")
with c2:
    corr_win = st.radio("Correlation Window", ["30d (720h)", "90d (2160h)"], horizontal=True, key="corr_window_radio")
    corr_data = p.corr_30d if corr_win.startswith("30d") else p.corr_90d
    st.plotly_chart(chart_correlation_heatmap(corr_data, f"Hourly Return Correlation ({corr_win})"), width="stretch")

st.divider()
st.subheader("1-Day Tail Risk: Historical vs Parametric Cornish-Fisher VaR / CVaR")
v_col1, v_col2 = st.columns([1.1, 1.3])
with v_col1:
    var_table = pd.DataFrame(
        [
            {"Metric": "1-Day 95% VaR", "Historical ($)": round(p.var_95_hist_usd, 2), "Cornish-Fisher ($)": round(p.var_95_cf_usd, 2), "Hist % Eq": round(p.var_95_hist_usd / p.equity * 100, 2)},
            {"Metric": "1-Day 99% VaR", "Historical ($)": round(p.var_99_hist_usd, 2), "Cornish-Fisher ($)": round(p.var_99_cf_usd, 2), "Hist % Eq": round(p.var_99_hist_usd / p.equity * 100, 2)},
            {"Metric": "1-Day 95% CVaR (ES)", "Historical ($)": round(p.cvar_95_hist_usd, 2), "Cornish-Fisher ($)": round(p.cvar_95_cf_usd, 2), "Hist % Eq": round(p.cvar_95_hist_usd / p.equity * 100, 2)},
            {"Metric": "1-Day 99% CVaR (ES)", "Historical ($)": round(p.cvar_99_hist_usd, 2), "Cornish-Fisher ($)": round(p.cvar_99_cf_usd, 2), "Hist % Eq": round(p.cvar_99_hist_usd / p.equity * 100, 2)},
        ]
    )
    st.dataframe(var_table, width="stretch", hide_index=True)
    st.caption(f"Computed over {p.var_lookback_days} aligned daily return observations. Positive figures represent 1-day loss potential.")
with v_col2:
    st.plotly_chart(chart_var_comparison(p), width="stretch")

st.divider()
st.subheader("Stress Tests & Forced Liquidation Check")
s_col1, s_col2 = st.columns([1.2, 1.2])
with s_col1:
    stress_df = pd.DataFrame(
        [
            {
                "Scenario": s.scenario,
                "PnL ($)": round(s.portfolio_pnl_usd, 2),
                "PnL %": round(s.portfolio_pnl_pct, 2),
                "Shocked Equity ($)": round(s.shocked_equity_usd, 2),
                "Liquidated": ", ".join(s.liquidated_symbols) if s.liquidated_symbols else "None",
            }
            for s in p.stress_scenarios
        ]
    )
    st.dataframe(stress_df, width="stretch", hide_index=True)
with s_col2:
    st.plotly_chart(chart_stress_scenarios(p), width="stretch")
st.caption(
    "PnL is linear mark-to-market of current notionals (BTC scenarios use each asset's 90d hourly beta; "
    "funding: longs pay positive funding). Isolated positions are liquidated when the shocked mark reaches "
    "their own liquidation price; cross positions are checked at account level: the cross pool (equity minus "
    "ring-fenced isolated equity) plus the shocked cross PnL versus the cross maintenance margin. "
    "Correlation -> 1 is shown in whichever direction (crash or squeeze) hurts this portfolio more."
)

with st.expander("90-Day Hourly Betas vs BTCUSDT & Per-Asset Breakdown"):
    asset_df = pd.DataFrame(
        [
            {"Symbol": sym, **{k: round(v, 3) for k, v in vals.items()}}
            for sym, vals in p.per_asset_exposure.items()
        ]
    )
    st.dataframe(asset_df, width="stretch", hide_index=True)

render_footer()
