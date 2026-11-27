"""RiskGuard — Page 1: Risk Desk & Discipline Dashboard."""

from __future__ import annotations

import math

import pandas as pd
import streamlit as st

from riskguard.analytics.positions import effective_leverage_label, liquidation_label
from riskguard.ui.state import build_session_context, load_demo_defaults, refresh_market_cache
from riskguard.ui.theme import configure_page, render_footer, render_sidebar, status_badge_html

configure_page("Dashboard", icon="🛡️")
ctx = build_session_context()
render_sidebar(
    market=ctx.market,
    on_load_demo=load_demo_defaults,
    on_refresh_market=refresh_market_cache,
)

p = ctx.portfolio
cb = ctx.circuit_breaker
breaches = [r for r in ctx.rule_evals if r.status == "BREACH"]
warns = [r for r in ctx.rule_evals if r.status == "WARN"]
dangers = [w for a in ctx.analyses for w in a.warnings if w.startswith("DANGER")]

if breaches or dangers or not cb.trading_allowed:
    desk_status = ("BREACH / ACTION REQUIRED", "DANGER")
elif warns or cb.status == "CAUTION":
    desk_status = ("WARN / ELEVATED RISK", "WARN")
else:
    desk_status = ("OK / WITHIN LIMITS", "OK")

st.title("🛡️ RiskGuard — Risk Desk & Discipline Dashboard")
st.caption("Your risk desk and discipline coach. Read-only crypto perpetuals portfolio, liquidation, tail-risk, and behavioural audit.")

badge_col1, badge_col2, badge_col3 = st.columns([2, 2, 3])
with badge_col1:
    st.markdown(f"**Desk Status:** {status_badge_html(desk_status[0], desk_status[1])}", unsafe_allow_html=True)
with badge_col2:
    cb_level = "OK" if cb.status == "OPEN" else ("WARN" if cb.status == "CAUTION" else "DANGER")
    cb_text = f"TRADING ALLOWED ({cb.status})" if cb.trading_allowed else f"TRADING HALTED ({cb.status})"
    st.markdown(f"**Circuit Breaker:** {status_badge_html(cb_text, cb_level)}", unsafe_allow_html=True)
    st.caption(
        f"Daily/weekly PnL evaluated for {cb.reference_date}"
        + (" (synthetic demo journal: its last trading day)" if st.session_state.get("journal_is_synthetic") else "")
        if cb.reference_date
        else "No journal loaded: daily/weekly loss limits have no PnL to evaluate."
    )
with badge_col3:
    if ctx.market is not None:
        m_lvl = "OK" if ctx.market.source == "binance" else ("WARN" if ctx.market.source == "hyperliquid" else "DEMO")
        st.markdown(f"**Market Feed:** {status_badge_html(ctx.market.badge_label, m_lvl)}", unsafe_allow_html=True)

for err in st.session_state.get("positions_errors", []):
    st.error(f"Excluded from analysis — invalid position row. {err}")

st.divider()

# Top KPI row
k1, k2, k3, k4, k5, k6 = st.columns(6)
upnl = p.total_unrealized_pnl_usd
k1.metric(
    "Account Equity",
    f"${p.equity:,.2f}",
    f"uPnL ${upnl:+,.0f}",
    delta_color="off",
    delta_arrow="off",
)
k2.metric("Gross Exposure", f"${p.gross_exposure_usd:,.0f}", f"{p.gross_leverage:.2f}x equity", delta_color="off", delta_arrow="off")
k3.metric("BTC-Beta Net", f"{'+' if (p.btc_beta_adj_net_usd) >= 0 else '-'}${abs(p.btc_beta_adj_net_usd):,.0f}", f"{p.btc_beta_adj_net_pct:+.1f}% equity", delta_color="off", delta_arrow="off")
k4.metric("Known Stop Risk", f"${p.known_stop_risk_usd:,.0f}", f"{p.known_stop_risk_pct:.2f}% ({p.unstopped_positions} unstopped)", delta_color="off", delta_arrow="off")
k5.metric("1d 95% Hist VaR", f"${p.var_95_hist_usd:,.0f}", f"CVaR ${p.cvar_95_hist_usd:,.0f}", delta_color="off", delta_arrow="off")
if ctx.journal is not None:
    k6.metric("Discipline Score", f"{ctx.journal.discipline_score:.0f}/100 ({ctx.journal.discipline_grade})", f"DD {ctx.journal.current_drawdown_pct:.2f}%", delta_color="off", delta_arrow="off")
else:
    k6.metric("Discipline Score", "N/A", "No journal loaded", delta_color="off", delta_arrow="off")

st.subheader("🚨 Top 3 Dangers Requiring Immediate Action")
prioritized_dangers: list[tuple[str, str]] = []
for a in ctx.analyses:
    if a.at_or_past_liquidation or a.liquidated:
        prioritized_dangers.append(("CRITICAL", f"{a.symbol} ({a.side.upper()} {a.leverage:.0f}x) is LIQUIDATED / AT/PAST liquidation ({liquidation_label(a)})."))
    elif a.stop_beyond_liquidation:
        prioritized_dangers.append(
            ("CRITICAL", f"{a.symbol} ({a.side.upper()} {a.leverage:.0f}x): stop (${a.stop_price:,.4f}) is beyond liquidation ({liquidation_label(a)}) — forced liquidation happens first.")
        )
    elif a.invalid_stop_direction:
        prioritized_dangers.append(
            ("CRITICAL", f"{a.symbol} ({a.side.upper()} {a.leverage:.0f}x): mark (${a.mark_price:,.4f}) has already crossed the stop (${a.stop_price:,.4f}) — the stop is not protecting the position.")
        )
for a in ctx.analyses:
    if a.missing_stop:
        prioritized_dangers.append(("HIGH", f"{a.symbol} (${a.notional:,.0f} notional) has NO stop-loss — risk is unbounded up to liquidation."))
for s in sorted(p.stress_scenarios, key=lambda x: x.liquidation_count, reverse=True):
    if s.liquidation_count > 0:
        prioritized_dangers.append(
            ("HIGH", f"Stress '{s.scenario}' causes ${s.portfolio_pnl_usd:+,.0f} ({s.portfolio_pnl_pct:+.1f}%) and liquidates: {', '.join(s.liquidated_symbols)}.")
        )
for r in breaches:
    if r.rule_id not in ("liquidation_safety", "mandatory_stop"):
        prioritized_dangers.append(("BREACH", f"{r.name}: {r.explanation}"))
if ctx.journal and ctx.journal.episode_counts:
    tilt_n = ctx.journal.episode_counts.get("tilt_session", 0)
    rev_n = ctx.journal.episode_counts.get("revenge_trading", 0)
    if tilt_n > 0 or rev_n > 0:
        prioritized_dangers.append(
            ("BEHAVIOUR", f"Journal flagged {rev_n} revenge-trading and {tilt_n} tilt episodes (Discipline {ctx.journal.discipline_score:.0f}/100).")
        )

if not prioritized_dangers:
    st.success("No critical dangers detected across open positions, rules, or stress scenarios.")
else:
    for idx, (sev, msg) in enumerate(prioritized_dangers[:3], start=1):
        if sev in ("CRITICAL", "HIGH"):
            st.error(f"**#{idx} [{sev}]** {msg}")
        else:
            st.warning(f"**#{idx} [{sev}]** {msg}")

col_left, col_right = st.columns([3, 2])
with col_left:
    st.subheader("📊 Open Positions Snapshot")
    if ctx.analyses:
        pos_rows = [
            {
                "Symbol": a.symbol,
                "Side": a.side.upper(),
                "Lev": f"{a.leverage:.0f}x {a.margin_mode[:4]}",
                "Eff. Lev": effective_leverage_label(a),
                "Notional ($)": round(a.notional, 2),
                "Mark": round(a.mark_price, 4),
                "Stop": round(a.stop_price, 4) if a.stop_price else None,
                "Liq Price": liquidation_label(a),
                "Dist LP %": round(a.distance_to_liq_pct, 2) if (a.distance_to_liq_pct is not None and math.isfinite(a.distance_to_liq_pct)) else None,
                "Risk % Eq": round(a.risk_to_stop_pct, 2) if a.risk_to_stop_pct is not None else None,
                "Status": "LIQUIDATED" if a.liquidated else a.danger_level,
            }
            for a in ctx.analyses
        ]
        st.dataframe(pd.DataFrame(pos_rows), width="stretch", hide_index=True)
    else:
        st.info("No open positions. Add positions on the Positions page or click Load Demo Data.")

with col_right:
    st.subheader("📏 Rules & Drawdown Schedule")
    rule_rows = [
        {"Rule": r.name, "Status": r.status, "Current": r.current_value, "Limit": r.limit_value}
        for r in ctx.rule_evals
    ]
    st.dataframe(pd.DataFrame(rule_rows), width="stretch", hide_index=True)
    if ctx.drawdown_plan is not None:
        dp = ctx.drawdown_plan
        st.info(
            f"**Drawdown Risk Throttle:** DD `{dp.current_dd_pct:.2f}%` -> "
            f"Recommended risk/trade **`{dp.recommended_risk_pct:.2f}%`** "
            f"(Step-down `{dp.step_down_multiplier:.2f}x`, CPPI `{dp.cppi_multiplier:.2f}x`)"
        )

render_footer()
