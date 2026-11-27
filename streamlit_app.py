"""RiskGuard — Multipage Streamlit Navigation Entrypoint."""

from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="RiskGuard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

pg = st.navigation(
    [
        st.Page("app_pages/dashboard.py", title="Dashboard", icon="🛡️", default=True),
        st.Page("pages/2_Positions.py", title="Positions", icon="📌"),
        st.Page("pages/3_Exposure_and_Stress.py", title="Exposure & Stress", icon="📉"),
        st.Page("pages/4_Pre_Trade_and_Sizing.py", title="Pre-Trade & Sizing", icon="⚖️"),
        st.Page("pages/5_Journal_and_Discipline.py", title="Journal & Discipline", icon="🧠"),
        st.Page("pages/6_Rules.py", title="Rules", icon="📏"),
        st.Page("pages/7_AI_Coach.py", title="AI Coach", icon="🤖"),
        st.Page("pages/8_About.py", title="About", icon="📘"),
    ]
)
pg.run()
