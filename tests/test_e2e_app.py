"""End-to-end Streamlit AppTest across all 8 pages in offline demo mode."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest
import requests
from streamlit.proto.Metric_pb2 import Metric as MetricProto
from streamlit.testing.v1 import AppTest

from riskguard.models import RiskOfRuinResult
from riskguard.ui.charts import chart_risk_of_ruin

ROOT = Path(__file__).resolve().parents[1]
MAIN_SCRIPT = str(ROOT / "streamlit_app.py")


@pytest.fixture(autouse=True)
def _block_real_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Guarantee zero live network requests run during tests."""
    monkeypatch.setenv("RG_DATA_MODE", "demo")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    def _forbid_network(*args, **kwargs):
        raise AssertionError("Real network call attempted during E2E test!")

    monkeypatch.setattr(requests.Session, "request", _forbid_network)


def test_all_pages_render_and_interactive_flows_succeed() -> None:
    # Page 1: Dashboard
    at = AppTest.from_file(MAIN_SCRIPT, default_timeout=20).run()
    assert not at.exception
    assert any("RiskGuard" in t.value for t in at.title)
    assert len(at.metric) >= 5

    # Page 2: Positions + edit a position via st.data_editor state
    at.switch_page("pages/2_Positions.py").run()
    assert not at.exception
    assert len(at.data_editor) == 1
    assert any("beyond liquidation" in e.value for e in at.error)

    # Lower DOGEUSDT leverage from 25x to 3x and raise stop inside safe zone -> liquidation warning disappears
    at.session_state["positions_editor"] = {
        "edited_rows": {0: {"leverage": 3.0, "stop_price": 0.18}},
        "added_rows": [],
        "deleted_rows": [],
    }
    at.run()
    assert not at.exception
    assert not any("DOGEUSDT stop" in e.value and "beyond liquidation" in e.value for e in at.error)

    # Page 3: Exposure & Stress
    at.switch_page("pages/3_Exposure_and_Stress.py").run()
    assert not at.exception
    assert any("Exposure" in t.value for t in at.title)
    assert len(at.dataframe) >= 2

    # Page 4: Pre-Trade Check & Sizing + submit form
    at.switch_page("pages/4_Pre_Trade_and_Sizing.py").run()
    assert not at.exception
    at.number_input(key="pt_notional").set_value(1500.0)
    at.number_input(key="pt_leverage").set_value(3.0)
    eval_btn = next(b for b in at.button if "Evaluate Pre-Trade Risk" in b.label)
    eval_btn.click().run()
    assert not at.exception
    assert any("Verdict:" in m.value for m in at.markdown)
    assert len(at.dataframe) >= 2

    # Page 5: Journal & Discipline
    at.switch_page("pages/5_Journal_and_Discipline.py").run()
    assert not at.exception
    assert any("SYNTHETIC DEMO JOURNAL" in i.value for i in at.info)
    assert any("Discipline Score" in m.value for m in at.markdown)

    # Page 6: Rules
    at.switch_page("pages/6_Rules.py").run()
    assert not at.exception
    assert any("Trading Allowed Today" in m.value for m in at.markdown)

    # Page 7: AI Coach (one-click offline review without API key)
    at.switch_page("pages/7_AI_Coach.py").run()
    assert not at.exception
    review_btn = next(b for b in at.button if "One-Click" in b.label)
    review_btn.click().run()
    assert not at.exception
    assert any("not financial advice" in m.value.lower() for m in at.markdown)

    # Page 8: About
    at.switch_page("pages/8_About.py").run()
    assert not at.exception
    assert any("About RiskGuard" in t.value for t in at.title)


def test_entrypoint_navigation_titles_config_and_no_page_config_errors() -> None:
    cfg = tomllib.loads((ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8"))
    assert cfg["client"]["toolbarMode"] == "viewer"

    at = AppTest.from_file(MAIN_SCRIPT, default_timeout=60).run()
    assert not at.exception
    assert any("Dashboard" in t.value for t in at.title)

    registered_titles = [info["page_name"] for info in at._registered_pages.values()]
    assert registered_titles == [
        "Dashboard",
        "Positions",
        "Exposure & Stress",
        "Pre-Trade & Sizing",
        "Journal & Discipline",
        "Rules",
        "AI Coach",
        "About",
    ]

    # Loading dashboard.py standalone and calling configure_page twice in one run raise no set_page_config errors
    direct_dash = AppTest.from_file(str(ROOT / "app_pages" / "dashboard.py"), default_timeout=60).run()
    assert not direct_dash.exception
    assert any("Dashboard" in t.value for t in direct_dash.title)

    def _double_configure() -> None:
        from riskguard.ui.theme import configure_page

        configure_page("Entry", icon="🛡️")
        configure_page("Dashboard", icon="🛡️")

    dup_at = AppTest.from_function(_double_configure, default_timeout=30).run()
    assert not dup_at.exception


def test_metric_delta_colors_and_risk_of_ruin_zero_bar_labels() -> None:
    at = AppTest.from_file(MAIN_SCRIPT, default_timeout=60).run()
    assert not at.exception
    dash_metrics = {m.label: m for m in at.metric}
    assert dash_metrics["Account Equity"].proto.color == MetricProto.MetricColor.GRAY
    assert dash_metrics["Account Equity"].proto.direction == MetricProto.MetricDirection.NONE
    for label in ("Gross Exposure", "BTC-Beta Net", "Known Stop Risk", "1d 95% Hist VaR", "Discipline Score"):
        assert dash_metrics[label].proto.color == MetricProto.MetricColor.GRAY
        assert dash_metrics[label].proto.direction == MetricProto.MetricDirection.NONE

    at.switch_page("pages/3_Exposure_and_Stress.py").run()
    assert not at.exception
    for m in at.metric:
        assert m.proto.color == MetricProto.MetricColor.GRAY
        assert m.proto.direction == MetricProto.MetricDirection.NONE

    at.switch_page("pages/5_Journal_and_Discipline.py").run()
    assert not at.exception
    j_metrics = {m.label: m for m in at.metric}
    assert j_metrics["Profit Factor"].proto.color == MetricProto.MetricColor.GREEN
    for label in ("Total Trades", "Total Net PnL", "Avg R / Payoff", "Max Drawdown", "Disposition Ratio"):
        assert j_metrics[label].proto.color == MetricProto.MetricColor.GRAY
        assert j_metrics[label].proto.direction == MetricProto.MetricDirection.NONE

    ror_zero = RiskOfRuinResult(
        ruin_drawdown_pct=15.0,
        risk_per_trade_pct=1.0,
        win_rate=0.55,
        payoff_ratio=1.5,
        expectancy_r=0.375,
        analytical_ruin_prob=0.0,
        mc_ruin_prob_flat=0.02,
        mc_ruin_prob_scheduled=0.0,
        mc_median_terminal_equity=12000.0,
        mc_p05_terminal_equity=9200.0,
        mc_p95_max_dd_pct=10.0,
        simulations=500,
        horizon_trades=100,
    )
    fig = chart_risk_of_ruin(ror_zero)
    bar = fig.data[0]
    assert list(bar.text) == ["0.00%", "2.00%", "0.00%"]
    assert bar.textposition == "outside"
    assert bar.cliponaxis is False
    assert list(fig.layout.yaxis.range) == [0.0, 2.5]

    fig_all_zero = chart_risk_of_ruin(ror_zero.model_copy(update={"mc_ruin_prob_flat": 0.0}))
    assert list(fig_all_zero.data[0].text) == ["0.00%", "0.00%", "0.00%"]
    assert list(fig_all_zero.layout.yaxis.range) == [0.0, 1.0]


def test_apptest_offline_ai_coach_chat_exchange() -> None:
    at = AppTest.from_file(str(ROOT / "pages" / "7_AI_Coach.py"), default_timeout=60).run()
    assert not at.exception

    at.chat_input[0].set_value("How do I fix my DOGEUSDT trade?").run()
    assert not at.exception
    assistant_msgs = [m for m in at.chat_message if m.name == "assistant"]
    assert len(assistant_msgs) == 1
    reply1 = "\n".join(md.value for md in assistant_msgs[0].markdown)
    assert "Position Diagnosis & Concrete Fixes — DOGEUSDT" in reply1
    assert "4.89x" in reply1
    assert "13,937" in reply1

    at.chat_input[0].set_value("can I buy BTC at 5x").run()
    assert not at.exception
    assistant_msgs = [m for m in at.chat_message if m.name == "assistant"]
    assert len(assistant_msgs) == 2
    reply2 = "\n".join(md.value for md in assistant_msgs[1].markdown)
    assert "Pre-Trade Check — LONG BTCUSDT @ 5.0x" in reply2
    assert "Verdict:" in reply2


def test_apptest_ai_coach_double_click_review_produces_single_message() -> None:
    at = AppTest.from_file(str(ROOT / "pages" / "7_AI_Coach.py"), default_timeout=60).run()
    assert not at.exception

    # First click produces 1 user + 1 assistant review message
    next(b for b in at.button if "One-Click" in b.label).click().run()
    assert not at.exception
    assistant_msgs = [m for m in at.chat_message if m.name == "assistant"]
    assert len(assistant_msgs) == 1
    assert len(at.session_state["ai_display_messages"]) == 2
    assert len(at.session_state["ai_history"]) == 2

    # Second quick click in the same input state must NOT append a duplicate review
    next(b for b in at.button if "One-Click" in b.label).click().run()
    assert not at.exception
    assistant_msgs = [m for m in at.chat_message if m.name == "assistant"]
    assert len(assistant_msgs) == 1
    assert len(at.session_state["ai_display_messages"]) == 2
    assert len(at.session_state["ai_history"]) == 2

    # Click while _coach_running is True must also be ignored even if state changes
    at.session_state["_coach_running"] = True
    at.number_input(key="sidebar_equity_input").set_value(15_000.0)
    next(b for b in at.button if "One-Click" in b.label).click().run()
    assert not at.exception
    assert len([m for m in at.chat_message if m.name == "assistant"]) == 1

    # Once not running and state has changed, a fresh review IS generated
    at.session_state["_coach_running"] = False
    next(b for b in at.button if "One-Click" in b.label).click().run()
    assert not at.exception
    assert len([m for m in at.chat_message if m.name == "assistant"]) == 2


def test_apptest_positions_shows_liquidated_for_past_lp_row() -> None:
    at = AppTest.from_file(MAIN_SCRIPT, default_timeout=60).run()
    assert not at.exception
    at.switch_page("pages/2_Positions.py").run()
    assert not at.exception

    # Set DOGEUSDT entry to 0.30 at 25x isolated while sample mark is ~0.1846 -> way past LP (~0.2894)
    at.session_state["positions_editor"] = {
        "edited_rows": {0: {"entry_price": 0.30, "leverage": 25.0, "stop_price": 0.25}},
        "added_rows": [],
        "deleted_rows": [],
    }
    at.run()
    assert not at.exception
    pos_df = next(d.value for d in at.dataframe if "Eff. Lev" in d.value.columns).set_index("Symbol")
    assert pos_df.loc["DOGEUSDT", "Eff. Lev"] == "LIQUIDATED"
    assert pos_df.loc["DOGEUSDT", "Alert"] == "LIQUIDATED"
    assert pos_df.loc["SOLUSDT", "Eff. Lev"] != "LIQUIDATED"
    assert pos_df.loc["SOLUSDT", "Eff. Lev"].endswith("x")


def test_theme_css_includes_metric_label_and_value_rules() -> None:
    from riskguard.ui.theme import SUITE_CSS

    assert '[data-testid="stMetricLabel"]' in SUITE_CSS
    assert "white-space: normal !important;" in SUITE_CSS
    assert "text-overflow: clip !important;" in SUITE_CSS
    assert "clamp(1.1rem, 2.2vw, 1.8rem)" in SUITE_CSS
