"""RiskGuard — Page 6: Configurable Trading Rules & Live Breaches Audit."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from riskguard.io import dump_rules_yaml, load_rules_yaml
from riskguard.models import RuleConfig
from riskguard.ui.state import (
    build_session_context,
    init_session_state,
    is_new_upload,
    load_demo_defaults,
    refresh_market_cache,
)
from riskguard.ui.theme import configure_page, render_footer, render_sidebar, status_badge_html

configure_page("Rules", icon="📏")
init_session_state()

st.title("📏 Risk Rules & Circuit Breaker")
st.caption("Configure portfolio, leverage, stop-loss, correlation, and loss-limit rules via form or YAML.")

rules: RuleConfig = st.session_state.get("rules", RuleConfig())

col_form, col_yaml = st.columns([1.3, 1.0])
with col_form:
    with st.form("rules_form"):
        st.subheader("Edit Risk Rules")
        c1, c2 = st.columns(2)
        # Widget bounds match RuleConfig's validation bounds, so any valid rules YAML can be shown and edited
        with c1:
            max_risk_trade = st.number_input("Max Risk Per Trade (%)", 0.01, 100.0, float(rules.max_risk_per_trade_pct), 0.25)
            max_total_risk = st.number_input("Max Total Open Risk (%)", 0.01, 100.0, float(rules.max_total_open_risk_pct), 0.5)
            max_lev = st.number_input("Max Leverage (x)", 1.0, 150.0, float(rules.max_leverage), 1.0)
            max_pos = st.number_input("Max Open Positions", 1, 50, int(rules.max_positions), 1)
            mandatory_stop = st.checkbox("Mandatory Stop-Loss", value=bool(rules.mandatory_stop))
        with c2:
            daily_loss = st.number_input("Daily Loss Limit (%)", 0.01, 100.0, float(rules.daily_loss_limit_pct), 0.5)
            weekly_loss = st.number_input("Weekly Loss Limit (%)", 0.01, 100.0, float(rules.weekly_loss_limit_pct), 0.5)
            max_dd = st.number_input("Max Drawdown (%)", 0.01, 100.0, float(rules.max_drawdown_pct), 1.0)
            max_corr_exp = st.number_input("Max Correlated Exposure (% Eq)", 0.01, 2000.0, float(rules.max_correlated_exposure_pct), 10.0)
            corr_thresh = st.slider("Correlation Threshold (r)", 0.10, 0.99, float(rules.correlation_threshold), 0.01)
            warn_frac = st.slider("Warn at fraction of limit", 0.11, 0.99, float(rules.warn_fraction), 0.01)

        if st.form_submit_button("Apply Rules", type="primary"):
            try:
                st.session_state["rules"] = RuleConfig(
                    max_risk_per_trade_pct=max_risk_trade,
                    max_total_open_risk_pct=max_total_risk,
                    max_leverage=max_lev,
                    daily_loss_limit_pct=daily_loss,
                    weekly_loss_limit_pct=weekly_loss,
                    max_drawdown_pct=max_dd,
                    max_positions=int(max_pos),
                    max_correlated_exposure_pct=max_corr_exp,
                    correlation_threshold=corr_thresh,
                    mandatory_stop=mandatory_stop,
                    warn_fraction=warn_frac,
                )
            except ValueError as exc:
                st.error(f"Invalid rules: {exc}")
            else:
                st.session_state.pop("_ctx_cache_digest", None)
                st.rerun()

with col_yaml:
    st.subheader("Import / Export YAML")
    yaml_file = st.file_uploader("Upload Rules YAML", type=["yaml", "yml"], key="rules_yaml_uploader")
    if is_new_upload("rules", yaml_file):
        try:
            loaded_rules = load_rules_yaml(yaml_file.getvalue())
            st.session_state["rules"] = loaded_rules
            st.session_state.pop("_ctx_cache_digest", None)
            st.success(f"Loaded rules from {yaml_file.name}.")
            st.rerun()
        except Exception as exc:
            st.error(f"Invalid YAML: {exc}")

    yaml_text = dump_rules_yaml(st.session_state["rules"])
    st.code(yaml_text, language="yaml")
    st.download_button(
        "Download Rules YAML",
        data=yaml_text,
        file_name="riskguard_rules.yaml",
        mime="text/yaml",
    )

ctx = build_session_context()
render_sidebar(
    market=ctx.market,
    on_load_demo=load_demo_defaults,
    on_refresh_market=refresh_market_cache,
)

st.divider()
cb = ctx.circuit_breaker
cb_lvl = "OK" if cb.status == "OPEN" else ("WARN" if cb.status == "CAUTION" else "DANGER")
cb_label = f"Trading Allowed Today: {'YES' if cb.trading_allowed else 'NO'} ({cb.status})"
st.markdown(f"### {status_badge_html(cb_label, cb_lvl)}", unsafe_allow_html=True)
st.caption(
    f"Daily / weekly loss evaluated for {cb.reference_date} (UTC). Loss limits trip when the loss reaches the limit."
    if cb.reference_date
    else "No journal loaded: daily / weekly loss limits have no closed-trade PnL to evaluate."
)
for reason in cb.reasons:
    if cb.status == "TRIPPED":
        st.error(reason)
    elif cb.status == "CAUTION":
        st.warning(reason)
    else:
        st.success(reason)

st.subheader("Live Rule Evaluation Audit")
audit_df = pd.DataFrame(
    [
        {
            "Rule": r.name,
            "Status": r.status,
            "Current": r.current_value,
            "Limit": r.limit_value,
            "Affected": ", ".join(r.affected_symbols) or "-",
            "Explanation": r.explanation,
        }
        for r in ctx.rule_evals
    ]
)
st.dataframe(audit_df, width="stretch", hide_index=True)

render_footer()
