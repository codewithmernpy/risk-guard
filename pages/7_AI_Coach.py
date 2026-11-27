"""RiskGuard — Page 7: AI Risk & Discipline Coach (Claude + Offline Fallback)."""

from __future__ import annotations

import hashlib
import json
import os

import streamlit as st

from riskguard.ai.coach import run_coach_turn
from riskguard.ui.state import build_session_context, load_demo_defaults, refresh_market_cache
from riskguard.ui.theme import configure_page, render_footer, render_sidebar, status_badge_html

configure_page("AI Coach", icon="🤖")
ctx = build_session_context()
ctx.k_atr_buffer = float(st.session_state.get("pt_katr", 1.5))
render_sidebar(
    market=ctx.market,
    on_load_demo=load_demo_defaults,
    on_refresh_market=refresh_market_cache,
)

st.title("🤖 RiskGuard AI Coach")
st.caption("Powered by Claude tool_runner over live portfolio, stress, rule, sizing, and journal tools — with full deterministic offline fallback.")

api_key = (st.session_state.get("anthropic_api_key") or os.getenv("ANTHROPIC_API_KEY") or "").strip()
if api_key:
    st.markdown(
        f"**Coach Mode:** {status_badge_html('LIVE CLAUDE · Tool-Calling Coach Ready', 'OK')}",
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        f"**Coach Mode:** {status_badge_html('OFFLINE DETERMINISTIC COACH (No API Key)', 'DEMO')}",
        unsafe_allow_html=True,
    )
    st.info(
        "No Anthropic API key is set in the sidebar or `ANTHROPIC_API_KEY`. "
        "You can still run full one-click reviews and ask questions using RiskGuard's deterministic offline risk coach."
    )

st.session_state.setdefault("ai_history", [])
st.session_state.setdefault("ai_display_messages", [])
st.session_state.setdefault("_coach_running", False)

_ONE_CLICK_REVIEW_PROMPT = (
    "Perform a comprehensive risk and discipline review of my open positions, liquidation risks, "
    "stress scenarios, rule breaches, journal behaviour, and drawdown schedule. Cite exact numbers and "
    "prioritize the most dangerous issues first."
)


def _coach_state_signature() -> str:
    base_digest = st.session_state.get("_ctx_cache_digest")
    if not base_digest:
        payload = {
            "eq": ctx.account.equity,
            "pos": [(a.symbol, a.side, a.quantity, a.mark_price, a.leverage, a.stop_price) for a in ctx.analyses],
            "rules": ctx.rules.model_dump(),
        }
        base_digest = hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]
    return f"{base_digest}:{ctx.k_atr_buffer}:{bool(api_key)}"


btn_col1, btn_col2, _ = st.columns([1.8, 1.0, 2.2])
with btn_col1:
    review_clicked = st.button("Run One-Click Portfolio & Discipline Review", type="primary", width="stretch", key="btn_one_click_review")
with btn_col2:
    if st.button("Clear Conversation", width="stretch", key="btn_clear_ai_chat"):
        st.session_state["ai_history"] = []
        st.session_state["ai_display_messages"] = []
        st.session_state["_coach_running"] = False
        st.rerun()


def _handle_prompt(prompt_text: str, *, deduplicate_same_state: bool = False) -> None:
    if st.session_state.get("_coach_running"):
        return

    msgs = st.session_state["ai_display_messages"]
    state_sig = _coach_state_signature()
    if deduplicate_same_state and len(msgs) >= 2:
        last_user = msgs[-2]
        last_assistant = msgs[-1]
        if (
            last_user.get("role") == "user"
            and last_user.get("text") == prompt_text
            and last_assistant.get("role") == "assistant"
            and last_assistant.get("state_sig") == state_sig
        ):
            return

    st.session_state["_coach_running"] = True
    try:
        with st.spinner("Analyzing positions, stress tests, rules, and discipline..."):
            result = run_coach_turn(
                ctx=ctx,
                history=st.session_state["ai_history"],
                user_message=prompt_text,
                api_key=api_key or None,
            )
    finally:
        st.session_state["_coach_running"] = False

    if (
        msgs
        and msgs[-1].get("role") == "assistant"
        and msgs[-1].get("text") == result.reply_text
        and msgs[-1].get("state_sig") == state_sig
    ):
        if len(st.session_state["ai_history"]) >= 2:
            del st.session_state["ai_history"][-2:]
        return

    msgs.append({"role": "user", "text": prompt_text})
    msgs.append(
        {
            "role": "assistant",
            "text": result.reply_text,
            "tool_calls": result.tool_calls,
            "notice": result.notice,
            "offline": result.used_fallback_offline,
            "state_sig": state_sig,
        }
    )


if review_clicked:
    _handle_prompt(_ONE_CLICK_REVIEW_PROMPT, deduplicate_same_state=True)

# Render chat history
for msg in st.session_state["ai_display_messages"]:
    with st.chat_message(msg["role"]):
        if msg.get("notice"):
            st.caption(f"ℹ️ {msg['notice']}")
        st.markdown(msg["text"])
        if msg.get("tool_calls"):
            with st.expander(f"Tool Calls Executed ({len(msg['tool_calls'])})", expanded=False):
                st.json(msg["tool_calls"])

if user_prompt := st.chat_input("Ask RiskGuard AI Coach (e.g., 'What happens if BTC drops 15%?' or 'How do I fix my DOGEUSDT trade?')"):
    _handle_prompt(user_prompt)
    st.rerun()

render_footer()
