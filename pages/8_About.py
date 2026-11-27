"""RiskGuard — Page 8: Methodology, Architecture & Limitations."""

from __future__ import annotations

import streamlit as st

from riskguard.ui.state import build_session_context, load_demo_defaults, refresh_market_cache
from riskguard.ui.theme import configure_page, render_footer, render_sidebar

configure_page("About", icon="📘")
ctx = build_session_context()
render_sidebar(
    market=ctx.market,
    on_load_demo=load_demo_defaults,
    on_refresh_market=refresh_market_cache,
)

st.title("📘 About RiskGuard — Methodology & Architecture")
st.caption("Read-only crypto perpetuals risk analytics and trading discipline engine.")

st.markdown(
    """
### What RiskGuard Solves
Crypto perpetual traders rarely blow up from a single bad indicator — they blow up from:
1. **Stops placed beyond liquidation price** under high isolated leverage.
2. **Hidden directional correlation** across altcoin longs (effectively 1 massive BTC-beta bet).
3. **Fat-tailed drawdowns** underestimated by Gaussian-only assumptions.
4. **Behavioural tilt** — revenge trading, size escalation after loss streaks, stop widening, and disposition bias.

### Key Quantitative Methods
- **USDT-M Liquidation Price:** `LP = (WB + cumB - side * Q * EP) / (Q * MMR - side * Q)` using a configurable per-symbol maintenance-margin tier table with continuous `cumB`.
- **Cross-Margin Approximation:** `WB_cross = max(0, equity - uPnL_i - other_committed_margin)`, documented cleanly as an analytical approximation holding other assets at mark while stress scenarios explicitly test simultaneous multi-asset margin calls.
- **Portfolio Tail Risk:** 1-day 95%/99% Historical VaR/CVaR and Parametric Cornish-Fisher VaR/CVaR incorporating sample skewness and excess kurtosis.
- **90d Hourly BTC Beta & 30d/90d Correlations:** Computed on aligned hourly returns against `BTCUSDT`, driving beta-adjusted net exposure and crash stress propagation.
- **Sizing & Pre-Trade Gate:** Fixed fractional, ATR volatility-targeted, capped fractional Kelly, and max-leverage-safe sizing keeping liquidation `k` daily ATRs beyond the stop.
- **Drawdown Risk Throttle & Risk of Ruin:** Step-down schedule + CPPI cushion scaling, paired with root-solved analytical Risk of Ruin and vectorized Monte Carlo simulation.

### Explicit Limitations & Disclaimers
- **Simplified Margin Tiers:** Default MMR brackets are simplified Binance-inspired tiers and may differ from your exchange VIP tier or dynamic leverage limits.
- **Cross-Margin Approximation:** Isolated vs cross mixed accounts are approximated cleanly; exact exchange margin engines also incorporate unrealized funding, ADL queues, and order-book liquidation fees.
- **VaR Assumptions:** Historical and Cornish-Fisher VaR assume recent return distributions remain informative during regime shifts; always cross-check with stress scenarios.
- **Read-Only Analytics:** RiskGuard contains **zero exchange API keys** and **zero order-placement code**. Educational and analytical tool only — **not financial advice**.
"""
)

render_footer()
