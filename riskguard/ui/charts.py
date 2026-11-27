"""Plotly interactive visualizations styled for the RiskGuard dark theme."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from riskguard.models import JournalAnalytics, PortfolioSummary, RiskOfRuinResult

DARK_BG = "#0B1020"
SURFACE_BG = "#131A2E"
TEXT_COLOR = "#E6E9F2"
ACCENT = "#6366F1"
GREEN = "#10B981"
AMBER = "#F59E0B"
RED = "#EF4444"


def _apply_dark_layout(fig: go.Figure, title: str, height: int = 380) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color=TEXT_COLOR)),
        paper_bgcolor=DARK_BG,
        plot_bgcolor=SURFACE_BG,
        font=dict(color=TEXT_COLOR),
        height=height,
        margin=dict(l=40, r=20, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
    )
    fig.update_xaxes(gridcolor="rgba(230,233,242,0.08)")
    fig.update_yaxes(gridcolor="rgba(230,233,242,0.08)")
    return fig


def chart_exposure_bars(portfolio: PortfolioSummary) -> go.Figure:
    """Bar chart comparing Gross, Net, and BTC-Beta-Adjusted exposures per asset."""
    assets = list(portfolio.per_asset_exposure.keys())
    if not assets:
        return _apply_dark_layout(go.Figure(), "Per-Asset Exposure ($)")
    net_vals = [portfolio.per_asset_exposure[s]["net_usd"] for s in assets]
    beta_vals = [
        portfolio.per_asset_exposure[s]["net_usd"] * portfolio.betas_vs_btc.get(s, 1.0)
        for s in assets
    ]
    fig = go.Figure(
        data=[
            go.Bar(name="Net Exposure ($)", x=assets, y=net_vals, marker_color=ACCENT),
            go.Bar(name="BTC-Beta Adjusted ($)", x=assets, y=beta_vals, marker_color=AMBER),
        ]
    )
    fig.update_layout(barmode="group")
    return _apply_dark_layout(fig, "Per-Asset Net vs BTC-Beta Adjusted Exposure ($)")


def chart_correlation_heatmap(corr_dict: dict[str, dict[str, float]], title: str) -> go.Figure:
    """Annotated Plotly correlation heatmap."""
    if not corr_dict:
        return _apply_dark_layout(go.Figure(), title)
    df = pd.DataFrame(corr_dict)
    fig = go.Figure(
        data=go.Heatmap(
            z=df.to_numpy(),
            x=list(df.columns),
            y=list(df.index),
            zmin=-1,
            zmax=1,
            colorscale="RdBu_r",
            text=df.round(2).to_numpy(),
            texttemplate="%{text}",
            colorbar=dict(title="r"),
        )
    )
    return _apply_dark_layout(fig, title, height=380)


def chart_var_comparison(portfolio: PortfolioSummary) -> go.Figure:
    """Grouped bar chart comparing Historical vs Cornish-Fisher VaR and CVaR."""
    labels = ["95% VaR", "99% VaR", "95% CVaR", "99% CVaR"]
    hist_vals = [
        portfolio.var_95_hist_usd,
        portfolio.var_99_hist_usd,
        portfolio.cvar_95_hist_usd,
        portfolio.cvar_99_hist_usd,
    ]
    cf_vals = [
        portfolio.var_95_cf_usd,
        portfolio.var_99_cf_usd,
        portfolio.cvar_95_cf_usd,
        portfolio.cvar_99_cf_usd,
    ]
    fig = go.Figure(
        data=[
            go.Bar(name="Historical 1d ($)", x=labels, y=hist_vals, marker_color=ACCENT),
            go.Bar(name="Cornish-Fisher 1d ($)", x=labels, y=cf_vals, marker_color=RED),
        ]
    )
    fig.update_layout(barmode="group")
    return _apply_dark_layout(fig, "1-Day Portfolio Tail Risk: Historical vs Cornish-Fisher ($)")


def chart_stress_scenarios(portfolio: PortfolioSummary) -> go.Figure:
    """Horizontal bar chart of stress scenario PnL with liquidation annotations."""
    scenarios = portfolio.stress_scenarios
    if not scenarios:
        return _apply_dark_layout(go.Figure(), "Stress Scenario PnL ($)")
    names = [s.scenario for s in scenarios]
    pnls = [s.portfolio_pnl_usd for s in scenarios]
    colors = [RED if s.liquidation_count > 0 else (AMBER if s.portfolio_pnl_usd < 0 else GREEN) for s in scenarios]
    text_labels = [
        f"{'+' if (s.portfolio_pnl_usd) >= 0 else '-'}${abs(s.portfolio_pnl_usd):,.0f} ({s.portfolio_pnl_pct:+.1f}%)"
        + (f" | LIQ: {', '.join(s.liquidated_symbols)}" if s.liquidated_symbols else "")
        for s in scenarios
    ]
    fig = go.Figure(
        go.Bar(
            y=names,
            x=pnls,
            orientation="h",
            marker_color=colors,
            text=text_labels,
            textposition="auto",
        )
    )
    return _apply_dark_layout(fig, "Stress Test Impact & Forced Liquidations", height=400)


def chart_equity_and_drawdown(journal: JournalAnalytics) -> go.Figure:
    """Equity curve + drawdown underwater subplot with flagged behavioural episode markers."""
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.65, 0.35],
        subplot_titles=("Cumulative Equity Curve ($)", "Drawdown Underwater (%)"),
    )
    if not journal.equity_curve:
        return _apply_dark_layout(fig, "Equity Curve & Drawdown", height=480)

    df = pd.DataFrame(journal.equity_curve)
    df["closed_dt"] = pd.to_datetime(df["closed_at"], utc=True, format="ISO8601")

    fig.add_trace(
        go.Scatter(x=df["closed_dt"], y=df["equity"], mode="lines", name="Equity", line=dict(color=ACCENT, width=2.2)),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df["closed_dt"],
            y=df["peak_equity"],
            mode="lines",
            name="Peak Equity",
            line=dict(color="rgba(148,163,184,0.45)", dash="dot", width=1.2),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df["closed_dt"],
            y=-df["drawdown_pct"],
            mode="lines",
            fill="tozeroy",
            name="Drawdown %",
            line=dict(color=RED, width=1.5),
            fillcolor="rgba(239,68,68,0.22)",
        ),
        row=2,
        col=1,
    )

    # Overlay behavioural episode markers onto equity curve
    if journal.episodes:
        eq_series = df.set_index("closed_dt")["equity"]
        ep_x = []
        ep_y = []
        ep_txt = []
        for ep in journal.episodes:
            ts = pd.Timestamp(ep.timestamp)
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")
            idx_pos = eq_series.index.get_indexer([ts], method="nearest")[0]
            ep_x.append(ts)
            ep_y.append(float(eq_series.iloc[idx_pos]))
            ep_txt.append(f"{ep.episode_type}: {ep.description}")
        fig.add_trace(
            go.Scatter(
                x=ep_x,
                y=ep_y,
                mode="markers",
                name="Behaviour Flags",
                marker=dict(color=AMBER, size=8, symbol="diamond"),
                text=ep_txt,
                hoverinfo="text+x",
            ),
            row=1,
            col=1,
        )

    _apply_dark_layout(fig, "Journal Equity Curve, Drawdown & Behaviour Episodes", height=500)
    return fig


def chart_r_distribution(journal: JournalAnalytics) -> go.Figure:
    """Histogram of realized R-multiples."""
    if not journal.r_multiples:
        return _apply_dark_layout(go.Figure(), "Realized R-Multiple Distribution")
    fig = go.Figure(
        data=go.Histogram(
            x=journal.r_multiples,
            nbinsx=30,
            marker_color=ACCENT,
            name="Trades",
        )
    )
    fig.add_vline(x=0, line_dash="dash", line_color=RED)
    return _apply_dark_layout(fig, "R-Multiple Distribution (Planned Stop Risk Units)", height=340)


def chart_risk_of_ruin(ror: RiskOfRuinResult) -> go.Figure:
    """Compare Analytical, Monte Carlo Flat, and Monte Carlo Drawdown-Scheduled ruin odds."""
    methods = ["Analytical", "MC Flat Risk", "MC DD-Scheduled"]
    probs = [
        ror.analytical_ruin_prob * 100.0,
        ror.mc_ruin_prob_flat * 100.0,
        ror.mc_ruin_prob_scheduled * 100.0,
    ]
    colors = [AMBER, RED, GREEN]
    fig = go.Figure(
        data=go.Bar(
            x=methods,
            y=probs,
            marker_color=colors,
            text=[f"{p:.2f}%" for p in probs],
            textposition="outside",
            textfont=dict(color=TEXT_COLOR),
            cliponaxis=False,
        )
    )
    max_prob = max(probs) if probs else 0.0
    upper_y = max(1.0, max_prob * 1.25)
    fig = _apply_dark_layout(fig, "Risk of Ruin: Flat Sizing vs Drawdown Step-Down Schedule", height=320)
    fig.update_yaxes(
        title_text=f"P(equity falls {ror.ruin_drawdown_pct:.0f}% below start) (%)",
        range=[0.0, upper_y],
    )
    return fig
