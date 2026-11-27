"""Tests for AI Coach tools (no API key), Anthropic SDK call contract, refusal & error fallbacks."""

from __future__ import annotations

import functools
import json
from types import SimpleNamespace

import anthropic
import httpx2 as httpx  # the anthropic 2.x SDK ships httpx2 (plain httpx is not installed with it)
import pytest

from riskguard.ai.coach import DEFAULT_MODEL, build_offline_coach_report, run_coach_turn
from riskguard.ai.prompts import COACH_PROMPT
from riskguard.ai.tools import CoachContext, build_coach_tools
from riskguard.analytics.drawdown import compute_drawdown_plan, estimate_risk_of_ruin
from riskguard.analytics.portfolio import _evaluate_shock_scenario, analyze_portfolio
from riskguard.analytics.positions import analyze_positions
from riskguard.analytics.rules import evaluate_rules
from riskguard.analytics.sizing import max_safe_leverage_for_stop, pre_trade_check, recommend_sizes
from riskguard.data.providers import MarketDataProvider
from riskguard.models import AccountState, PositionInput, RuleConfig


def _sample_context() -> CoachContext:
    account = AccountState(equity=10_000.0)
    rules = RuleConfig()
    positions = [
        PositionInput(
            symbol="DOGEUSDT",
            side="long",
            entry_price=0.20,
            quantity=25_000.0,
            leverage=25.0,
            margin_mode="isolated",
            stop_price=0.18,  # beyond LP (~0.193)
        ),
        PositionInput(
            symbol="XRPUSDT",
            side="long",
            entry_price=2.50,
            quantity=1_000.0,
            leverage=5.0,
            margin_mode="isolated",
            stop_price=None,
        ),
    ]
    analyses = analyze_positions(positions, account)
    portfolio = analyze_portfolio(analyses, account)
    rule_evals, cb = evaluate_rules(analyses, portfolio, account, rules)
    dd_plan = compute_drawdown_plan(current_equity=10_000.0, current_dd_pct=2.0)
    ror = estimate_risk_of_ruin(win_rate=0.52, payoff_ratio=1.4, simulations=500, horizon_trades=100)
    return CoachContext(
        account=account,
        positions=positions,
        analyses=analyses,
        portfolio=portfolio,
        rules=rules,
        rule_evals=rule_evals,
        circuit_breaker=cb,
        dd_plan=dd_plan,
        risk_of_ruin=ror,
    ) if False else CoachContext(
        account=account,
        positions=positions,
        analyses=analyses,
        portfolio=portfolio,
        rules=rules,
        rule_evals=rule_evals,
        circuit_breaker=cb,
        drawdown_plan=dd_plan,
        risk_of_ruin=ror,
    )


def test_ai_tools_build_with_valid_schemas_without_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    ctx = _sample_context()
    tools = build_coach_tools(ctx)
    names = [t.name for t in tools]
    assert names == [
        "get_portfolio_risk",
        "get_position_details",
        "run_stress",
        "check_rules",
        "evaluate_new_trade",
        "journal_behaviour",
        "size_trade",
        "drawdown_plan",
    ]
    for tool in tools:
        defn = tool.to_dict()
        assert defn["name"]
        assert defn["description"]
        assert defn["input_schema"]["type"] == "object"

    # Execute every tool directly and verify valid compact JSON < 3000 chars
    outputs = [
        tools[0].call({}),
        tools[1].call({"symbol": "DOGEUSDT"}),
        tools[2].call({"scenario": "all"}),
        tools[3].call({}),
        tools[4].call({"trade_json": json.dumps({"symbol": "BTCUSDT", "side": "long", "entry_price": 100000, "notional": 2000, "leverage": 5, "margin_mode": "isolated", "stop_price": 96000})}),
        tools[5].call({}),
        tools[6].call({"symbol": "BTCUSDT", "side": "long", "entry": 100000.0, "stop": 96000.0, "method": "all"}),
        tools[7].call({}),
    ]
    for out in outputs:
        assert isinstance(out, str)
        assert len(out) < 3000
        json.loads(out)


def test_offline_coach_without_key_cites_numbers_and_disclaimer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    ctx = _sample_context()
    history: list[dict] = []
    res = run_coach_turn(ctx, history, "Audit my portfolio", api_key=None)

    assert res.used_fallback_offline is True
    assert "DOGEUSDT" in res.reply_text
    assert "XRPUSDT" in res.reply_text
    assert "not financial advice" in res.reply_text.lower()
    assert len(history) == 2
    assert history[1]["role"] == "assistant"


def test_tool_runner_exact_sdk_contract_and_history_append() -> None:
    ctx = _sample_context()
    captured_kwargs: dict = {}

    class _FakeBlock:
        def __init__(self, btype: str, **kwargs):
            self.type = btype
            for k, v in kwargs.items():
                setattr(self, k, v)

    turn1_msg = SimpleNamespace(
        role="assistant",
        stop_reason="tool_use",
        model=DEFAULT_MODEL,
        content=[_FakeBlock("tool_use", id="tu_1", name="get_portfolio_risk", input={})],
    )
    turn2_msg = SimpleNamespace(
        role="assistant",
        stop_reason="end_turn",
        model=DEFAULT_MODEL,
        content=[
            _FakeBlock("thinking", thinking="Checking DOGEUSDT stop vs LP."),
            _FakeBlock("text", text="Close or reduce DOGEUSDT (stop 0.18 is beyond LP). Not financial advice."),
        ],
    )

    class _FakeClient:
        def __init__(self, api_key=None):
            self.beta = SimpleNamespace(
                messages=SimpleNamespace(
                    tool_runner=self._tool_runner,
                )
            )

        def _tool_runner(self, **kwargs):
            captured_kwargs.update(kwargs)
            return iter([turn1_msg, turn2_msg])

    history: list[dict] = []
    res = run_coach_turn(
        ctx,
        history,
        "What is my top danger?",
        api_key="sk-ant-fake-test",
        client_factory=_FakeClient,
    )

    assert captured_kwargs["model"] == DEFAULT_MODEL
    assert captured_kwargs["max_tokens"] == 16000
    assert captured_kwargs["max_iterations"] == 16
    assert captured_kwargs["system"] == COACH_PROMPT
    assert captured_kwargs["output_config"] == {"effort": "high"}
    assert captured_kwargs["betas"] == ["server-side-fallback-2026-07-01"]
    assert captured_kwargs["fallbacks"] == "default"
    assert "temperature" not in captured_kwargs
    assert "top_p" not in captured_kwargs
    assert "thinking" not in captured_kwargs
    assert "tool_choice" not in captured_kwargs

    assert res.used_fallback_offline is False
    assert len(res.tool_calls) == 1
    assert res.tool_calls[0]["tool"] == "get_portfolio_risk"
    assert len(history) == 2
    # Full assistant content (thinking + text) preserved
    assert len(history[1]["content"]) == 2


@pytest.mark.parametrize("err_kind", ["refusal", "conn", "rate", "status"])
def test_coach_handles_refusal_and_api_errors(err_kind: str) -> None:
    ctx = _sample_context()

    class _FakeErrClient:
        def __init__(self, api_key=None):
            self.beta = SimpleNamespace(messages=SimpleNamespace(tool_runner=self._tool_runner))

        def _tool_runner(self, **kwargs):
            req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
            if err_kind == "refusal":
                return iter([SimpleNamespace(role="assistant", stop_reason="refusal", content=[])])
            if err_kind == "conn":
                raise anthropic.APIConnectionError(request=req)
            if err_kind == "rate":
                resp = httpx.Response(429, request=req)
                raise anthropic.RateLimitError("Rate limited", response=resp, body=None)
            resp = httpx.Response(500, request=req)
            raise anthropic.APIStatusError("Server error", response=resp, body=None)

    history: list[dict] = []
    res = run_coach_turn(ctx, history, "Review risk", api_key="sk-ant-fake", client_factory=_FakeErrClient)
    assert res.used_fallback_offline is True
    assert res.notice is not None
    assert "DOGEUSDT" in res.reply_text
    assert "not financial advice" in res.reply_text.lower()


@functools.lru_cache(maxsize=1)
def _demo_market_context() -> CoachContext:
    """Offline context using bundled demo market snapshots and realistic multi-coin portfolio."""
    account = AccountState(equity=10_000.0)
    rules = RuleConfig()
    positions = [
        PositionInput(
            symbol="DOGEUSDT",
            side="long",
            entry_price=0.1865,
            quantity=26_810.0,
            leverage=25.0,
            margin_mode="isolated",
            stop_price=0.1720,
        ),
        PositionInput(
            symbol="SOLUSDT",
            side="long",
            entry_price=186.50,
            quantity=22.0,
            leverage=8.0,
            margin_mode="isolated",
            stop_price=178.0,
        ),
        PositionInput(
            symbol="XRPUSDT",
            side="long",
            entry_price=2.4950,
            quantity=1_400.0,
            leverage=10.0,
            margin_mode="isolated",
            stop_price=None,
        ),
    ]
    market = MarketDataProvider().get_market_data(["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "BNBUSDT"], mode="demo")
    analyses = analyze_positions(positions, account, market=market)
    portfolio = analyze_portfolio(analyses, account, market=market)
    rule_evals, cb = evaluate_rules(analyses, portfolio, account, rules)
    dd_plan = compute_drawdown_plan(current_equity=10_000.0, current_dd_pct=1.8)
    ror = estimate_risk_of_ruin(win_rate=0.52, payoff_ratio=1.4, simulations=300, horizon_trades=100)
    return CoachContext(
        account=account,
        positions=positions,
        analyses=analyses,
        portfolio=portfolio,
        rules=rules,
        rule_evals=rule_evals,
        circuit_breaker=cb,
        market=market,
        drawdown_plan=dd_plan,
        risk_of_ruin=ror,
    )


@pytest.mark.parametrize("prompt", ["How do I fix my DOGEUSDT trade?", "doge"])
def test_offline_coach_intent_a_held_position_diagnosis_and_fixes(prompt: str) -> None:
    ctx = _demo_market_context()
    doge = next(a for a in ctx.analyses if a.symbol == "DOGEUSDT")
    expected_safe_lev = max_safe_leverage_for_stop(
        doge.entry_price, doge.stop_price, doge.atr_1d, doge.side, doge.mmr, k_atr=ctx.k_atr_buffer
    )
    expected_max_qty = (ctx.account.equity * ctx.rules.max_risk_per_trade_pct / 100.0) / abs(doge.mark_price - doge.stop_price)

    reply = build_offline_coach_report(ctx, user_question=prompt)
    assert "Position Diagnosis & Concrete Fixes — DOGEUSDT" in reply
    assert f"{expected_safe_lev:.2f}x" in reply
    assert f"{expected_max_qty:,.4f}" in reply
    assert f"${doge.liquidation_price:,.4f}" in reply
    assert "Not financial advice" in reply


def test_offline_coach_intent_a_unstopped_position_gives_safe_stop_and_qty() -> None:
    ctx = _demo_market_context()
    xrp = next(a for a in ctx.analyses if a.symbol == "XRPUSDT")
    safe_stop = xrp.liquidation_price + ctx.k_atr_buffer * xrp.atr_1d
    reply = build_offline_coach_report(ctx, user_question="fix my xrp position")
    assert "Position Diagnosis & Concrete Fixes — XRPUSDT" in reply
    assert f"${safe_stop:,.4f}" in reply
    assert "Max Safe Leverage" in reply
    assert "Risk-Compliant Quantity" in reply


@pytest.mark.parametrize(
    ("prompt", "shock"),
    [
        ("what if BTC drops 15%", -0.15),
        ("btc crash 20%", -0.20),
        ("btc -10", -0.10),
    ],
)
def test_offline_coach_intent_b_custom_btc_shock(prompt: str, shock: float) -> None:
    ctx = _demo_market_context()
    shocks = {a.symbol: ctx.portfolio.betas_vs_btc.get(a.symbol, 1.0) * shock for a in ctx.analyses}
    expected = _evaluate_shock_scenario("test", "test", ctx.analyses, ctx.account, shocks)

    reply = build_offline_coach_report(ctx, user_question=prompt)
    assert f"BTC {shock * 100:+.1f}%" in reply
    assert f"${expected.portfolio_pnl_usd:+,.2f}" in reply
    assert f"{expected.portfolio_pnl_pct:+.2f}%" in reply
    for sym in expected.liquidated_symbols:
        assert sym in reply
    assert "Not financial advice" in reply


@pytest.mark.parametrize(
    ("prompt", "sym", "side", "lev"),
    [
        ("can I open sol", "SOLUSDT", "long", 10.0),
        ("can I buy ETH at 3x", "ETHUSDT", "long", 3.0),
        ("Can I long BTC at 25x?", "BTCUSDT", "long", 25.0),
        ("can i short bnb at 5 x", "BNBUSDT", "short", 5.0),
    ],
)
def test_offline_coach_intent_c_pre_trade_check(prompt: str, sym: str, side: str, lev: float) -> None:
    ctx = _demo_market_context()
    snap = ctx.market.snapshots[sym]
    entry = snap.mark_price
    atr = snap.atr_1d
    stop = (entry - 2.0 * atr) if side == "long" else (entry + 2.0 * atr)
    recs = recommend_sizes(
        symbol=sym,
        side=side,
        entry_price=entry,
        stop_price=stop,
        account=ctx.account,
        atr_1d=atr,
        risk_pct=min(1.0, ctx.rules.max_risk_per_trade_pct),
        leverage=lev,
        margin_mode="isolated",
    )
    proposed = PositionInput(
        symbol=sym,
        side=side,
        entry_price=entry,
        notional=recs[0].notional_usd,
        leverage=lev,
        margin_mode="isolated",
        stop_price=stop,
    )
    expected = pre_trade_check(proposed, ctx.positions, ctx.account, ctx.market, ctx.rules, journal=ctx.journal)

    reply = build_offline_coach_report(ctx, user_question=prompt)
    assert f"Pre-Trade Check — {side.upper()} {sym} @ {lev:.1f}x" in reply
    assert f"`{expected.verdict}`" in reply
    for reason in expected.reasons:
        assert reason in reply
    assert "Not financial advice" in reply


@pytest.mark.parametrize(
    "prompt",
    [
        "How is my LINKUSDT trade?",
        "What about NOTACOIN?",
        "can I buy PEPE at 3x",
        "BTC",
    ],
)
def test_offline_coach_intent_d_unheld_or_unknown_coin(prompt: str) -> None:
    ctx = _demo_market_context()
    reply = build_offline_coach_report(ctx, user_question=prompt)
    assert "Unknown / Unheld Coin" in reply
    assert "Not financial advice" in reply


@pytest.mark.parametrize(
    "prompt",
    [
        "Audit my portfolio",
        "How is my win rate?",
        "HOW IS MY RISK?",
        "Perform a comprehensive risk and discipline review of my open positions, liquidation risks, stress scenarios, rule breaches, journal behaviour, and drawdown schedule.",
    ],
)
def test_offline_coach_intent_e_general_questions_return_full_briefing(prompt: str) -> None:
    ctx = _demo_market_context()
    reply = build_offline_coach_report(ctx, user_question=prompt)
    assert "RiskGuard Risk & Discipline Briefing" in reply
    assert "1. Prioritized Actions" in reply
    assert "2. Portfolio Exposure, VaR & Stress Reality" in reply
    assert "Not financial advice" in reply
