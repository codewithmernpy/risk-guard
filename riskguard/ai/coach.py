"""Anthropic SDK tool_runner AI coach with deterministic offline fallback."""

from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from typing import Any, Callable, Sequence

import anthropic
from pydantic import BaseModel, Field

from riskguard.ai.prompts import COACH_PROMPT
from riskguard.ai.tools import CoachContext, build_coach_tools
from riskguard.analytics.portfolio import _evaluate_shock_scenario
from riskguard.analytics.positions import committed_margin, effective_leverage_label, liquidation_label
from riskguard.analytics.sizing import max_safe_leverage_for_stop, pre_trade_check, recommend_sizes
from riskguard.data.providers import (
    BINANCE_MULTIPLIER_PREFIXES,
    MarketDataBundle,
    SymbolMarketData,
    symbol_to_hyperliquid_coin,
)
from riskguard.logging_config import get_logger
from riskguard.models import PositionAnalysis, PositionInput, Side, normalize_symbol

logger = get_logger("riskguard.ai")

DEFAULT_MODEL = "claude-opus-5-5"
DISCLAIMER = "Not financial advice."

_NEG_SHOCK_VERBS = (
    r"drop|drops|dropped|fall|falls|fell|crash|crashes|crashed|dump|dumps|dumped|"
    r"down|plunge|plunges|plunged|tank|tanks|tanked|decline|declines|declined|"
    r"sell\s*off|selloff|slump|slumps|slip|slips|bleed|bleeds|dip|dips|shock"
)
_POS_SHOCK_VERBS = (
    r"rally|rallies|rallied|pump|pumps|pumped|up|rise|rises|rose|"
    r"surge|surges|surged|jump|jumps|jumped|gain|gains|gained|spike|spikes|spiked"
)
_BTC_REF = r"(?:btc|btcusdt|bitcoin)"

_PRE_TRADE_FILLERS = frozenset(
    {
        "a",
        "an",
        "the",
        "new",
        "position",
        "pos",
        "trade",
        "perp",
        "perpetual",
        "long",
        "short",
        "buy",
        "sell",
        "on",
        "in",
        "for",
        "of",
        "at",
        "with",
        "using",
        "leverage",
        "lev",
        "order",
        "some",
        "more",
    }
)

_KNOWN_TICKERS = frozenset(
    {
        "btc",
        "eth",
        "sol",
        "xrp",
        "doge",
        "bnb",
        "ada",
        "avax",
        "link",
        "dot",
        "pepe",
        "arb",
        "op",
        "sui",
        "ton",
        "matic",
        "pol",
        "uni",
        "aave",
        "ltc",
        "bch",
        "atom",
        "near",
        "apt",
        "inj",
        "wif",
        "bonk",
        "shib",
        "sei",
        "tia",
        "ftm",
        "fil",
        "trx",
        "xlm",
        "hbar",
        "vet",
        "icp",
        "etc",
        "render",
        "fet",
        "tao",
    }
)

_NON_COIN_WORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "my",
        "me",
        "it",
        "this",
        "that",
        "all",
        "everything",
        "now",
        "today",
        "week",
        "month",
        "portfolio",
        "positions",
        "position",
        "trades",
        "trade",
        "risk",
        "risks",
        "rules",
        "rule",
        "breach",
        "breaches",
        "journal",
        "discipline",
        "drawdown",
        "stress",
        "exposure",
        "margin",
        "leverage",
        "var",
        "cvar",
        "hhi",
        "beta",
        "score",
        "status",
        "summary",
        "briefing",
        "report",
        "review",
        "audit",
        "help",
        "hello",
        "hi",
        "hey",
        "account",
        "equity",
        "stops",
        "stop",
        "liquidation",
        "funding",
        "correlation",
        "actions",
        "dangers",
        "danger",
        "episodes",
        "behaviour",
        "behavior",
        "sizing",
        "sizes",
        "ruin",
        "plan",
        "schedule",
        "market",
        "data",
        "what",
        "how",
        "why",
        "when",
        "where",
        "who",
        "ok",
        "okay",
        "thanks",
        "thank",
        "you",
        "current",
        "open",
        "full",
        "comprehensive",
    }
)


class CoachTurnResult(BaseModel):
    """Output of a single AI Coach interaction (live Claude or offline fallback)."""

    reply_text: str
    assistant_content: list[Any] = Field(default_factory=list)
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    used_fallback_offline: bool = False
    notice: str | None = None
    stop_reason: str | None = None
    model: str | None = None


def _extract_text_from_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    parts: list[str] = []
    if isinstance(content, list):
        for block in content:
            b_type = getattr(block, "type", None) or (block.get("type") if isinstance(block, dict) else None)
            if b_type == "text":
                txt = getattr(block, "text", None) or (block.get("text") if isinstance(block, dict) else "")
                if txt:
                    parts.append(str(txt))
    return "\n\n".join(parts).strip()


def _serialize_content_for_history(content: list[Any], strip_tool_use: bool = False) -> list[Any]:
    """Preserve full assistant content blocks cleanly while guarding against dangling tool_use."""
    serialized: list[Any] = []
    for block in content:
        b_type = getattr(block, "type", None) or (block.get("type") if isinstance(block, dict) else None)
        if strip_tool_use and b_type == "tool_use":
            continue
        serialized.append(block)
    return serialized


def _fmt_price(px: float) -> str:
    digits = 6 if abs(px) < 0.01 else 4
    return f"${px:,.{digits}f}"


def _tokenize_lower(text: str) -> set[str]:
    tokens = set(re.findall(r"[a-z0-9]+", text.lower()))
    for compound in re.findall(r"[a-z0-9]+(?:[/-][a-z0-9]+)+", text.lower()):
        cleaned = compound.replace("/", "").replace("-", "")
        for suffix in ("perp", "swap"):
            if cleaned.endswith(suffix):
                cleaned = cleaned[: -len(suffix)]
        if cleaned:
            tokens.add(cleaned)
    return tokens


def _symbol_aliases(symbol: str) -> set[str]:
    sym_low = symbol.lower()
    base_low = symbol_to_hyperliquid_coin(symbol).lower()
    aliases = {sym_low, base_low}
    for prefix in BINANCE_MULTIPLIER_PREFIXES:
        p_low = prefix.lower()
        if base_low.startswith(p_low) and len(base_low) > len(p_low):
            aliases.add(base_low[len(p_low) :])
    return aliases


def _match_held_positions(q_low: str, analyses: Sequence[PositionAnalysis]) -> list[PositionAnalysis]:
    tokens = _tokenize_lower(q_low)
    return [a for a in analyses if _symbol_aliases(a.symbol) & tokens]


def _parse_btc_shock(q_low: str) -> float | None:
    """Parse a BTC percentage shock (e.g. 'what if BTC drops 15%', 'btc crash 20%', 'btc -10')."""
    if not re.search(rf"\b{_BTC_REF}\b", q_low):
        return None

    m = re.search(
        rf"\b{_BTC_REF}\b[^.?!\n]{{0,40}}?\b({_NEG_SHOCK_VERBS})\b[^0-9+-]{{0,12}}([+-]?\d+(?:\.\d+)?)\s*%?(?!\s*x\b)",
        q_low,
    )
    if m:
        raw_num = m.group(2)
        val = float(raw_num)
        return (val / 100.0) if raw_num.startswith("+") else (-abs(val) / 100.0)

    m = re.search(
        rf"\b{_BTC_REF}\b[^.?!\n]{{0,40}}?\b({_POS_SHOCK_VERBS})\b[^0-9+-]{{0,12}}([+-]?\d+(?:\.\d+)?)\s*%?(?!\s*x\b)",
        q_low,
    )
    if m:
        return abs(float(m.group(2))) / 100.0

    m = re.search(
        rf"\b{_BTC_REF}\b\s*(?:at|by|to|moves?|changes?)?\s*([+-]\s*\d+(?:\.\d+)?)\s*%?(?!\s*x\b)",
        q_low,
    )
    if m:
        return float(m.group(1).replace(" ", "")) / 100.0

    m = re.search(
        rf"\b({_NEG_SHOCK_VERBS}|{_POS_SHOCK_VERBS})\b[^0-9+-]{{0,12}}(\d+(?:\.\d+)?)\s*%?\s*(?:in|on|for)?\s*\b{_BTC_REF}\b",
        q_low,
    )
    if m:
        verb, val_s = m.group(1), m.group(2)
        is_neg = bool(re.fullmatch(_NEG_SHOCK_VERBS, verb))
        return (-1.0 if is_neg else 1.0) * abs(float(val_s)) / 100.0

    return None


def _parse_pre_trade_query(q_low: str) -> tuple[Side, str, float | None] | None:
    """Parse 'can I open/buy/long/short <coin> [at N x]' pre-trade questions."""
    m = re.search(
        r"\b(?:can\s+i|should\s+i|could\s+i|may\s+i|is\s+it\s+(?:ok|okay|safe)\s+to)\s+"
        r"(open|buy|long|short|sell|enter|take)\b(.*)$",
        q_low,
    )
    if not m:
        return None
    verb, rest = m.group(1), m.group(2)

    lev: float | None = None
    lev_m = re.search(r"(?:(?:at|with|@|using)\s*)?(\d+(?:\.\d+)?)\s*x\b", rest)
    if lev_m:
        lev = float(lev_m.group(1))
        rest = rest[: lev_m.start()] + " " + rest[lev_m.end() :]

    side: Side = "short" if (verb in ("short", "sell") or re.search(r"\b(?:short|sell)\b", rest)) else "long"

    coin_token: str | None = None
    for raw_tok in re.findall(r"[a-z0-9/_-]+", rest):
        tok = raw_tok.strip("?.,!:;\"'()[]")
        if not tok or tok in _PRE_TRADE_FILLERS or tok in _NON_COIN_WORDS:
            continue
        coin_token = tok
        break

    if not coin_token:
        return None
    return side, coin_token, lev


def _resolve_symbol_snapshot(
    raw_coin: str,
    ctx: CoachContext,
) -> tuple[str, SymbolMarketData | None, MarketDataBundle | dict[str, SymbolMarketData] | None]:
    """Resolve raw coin/symbol against active market snapshots or held positions without network calls."""
    cleaned = raw_coin.strip().upper().replace("/", "").replace("-", "").replace("_", "")
    for suffix in ("PERP", "SWAP"):
        if cleaned.endswith(suffix):
            cleaned = cleaned[: -len(suffix)]
    try:
        norm = normalize_symbol(cleaned)
    except ValueError:
        return raw_coin.strip().upper(), None, ctx.market

    snapshots = ctx.market.snapshots if ctx.market is not None else {}
    if norm in snapshots:
        return norm, snapshots[norm], ctx.market

    if norm.endswith("USDT"):
        base = norm[:-4]
        for prefix in BINANCE_MULTIPLIER_PREFIXES:
            cand = f"{prefix}{base}USDT"
            if cand in snapshots:
                return cand, snapshots[cand], ctx.market

    # Fallback for unit tests where MarketDataBundle is None but analyses already carry mark & ATR
    analysis_map = {a.symbol: a for a in ctx.analyses}
    if norm in analysis_map:
        synthetic_snaps: dict[str, SymbolMarketData] = dict(snapshots)
        now = datetime.now(timezone.utc)
        for a in ctx.analyses:
            if a.symbol not in synthetic_snaps:
                synthetic_snaps[a.symbol] = SymbolMarketData(
                    symbol=a.symbol,
                    mark_price=a.mark_price,
                    index_price=a.mark_price,
                    funding_rate_8h=a.funding_rate_8h,
                    funding_rate_daily=a.funding_rate_8h * 3.0,
                    atr_1d=a.atr_1d,
                    source="sample",
                    as_of=now,
                )
        return norm, synthetic_snaps[norm], synthetic_snaps

    return norm, None, ctx.market


_COMMON_ACRONYMS = frozenset(
    {
        "VAR",
        "CVAR",
        "HHI",
        "ATR",
        "ROE",
        "PNL",
        "UPNL",
        "ROI",
        "MMR",
        "YAML",
        "CSV",
        "JSON",
        "API",
        "USD",
        "USDT",
        "USDC",
        "BUSD",
        "DD",
        "CPPI",
        "MC",
        "OK",
        "NO",
        "GO",
        "AI",
        "LP",
        "TP",
        "SL",
        "UI",
        "CLI",
        "UTC",
        "PR",
        "R",
    }
)


def _extract_unheld_coin_query(q_raw: str, q_low: str, ctx: CoachContext) -> str | None:
    """Detect when the user is asking about a coin that is not in open positions."""
    tokens = _tokenize_lower(q_low)
    market_aliases: set[str] = set()
    if ctx.market is not None:
        for sym in ctx.market.snapshots:
            market_aliases.update(_symbol_aliases(sym))

    # 1. Explicit pair suffix like LINKUSDT / ADA-USDT / PEPE/USDT
    for tok in sorted(tokens):
        if tok.endswith(("usdt", "usdc", "busd")) and len(tok) > 4:
            return normalize_symbol(tok)

    # 2. Known market snapshot symbol or common crypto ticker
    for tok in re.findall(r"[a-z0-9]+", q_low):
        if tok in market_aliases or tok in _KNOWN_TICKERS:
            sym, _, _ = _resolve_symbol_snapshot(tok, ctx)
            return sym

    # 3. Explicit "<coin> trade/position/pos/bag/coin/perp" phrasing
    m = re.search(
        r"\b(?:fix|check|analyze|analyse|diagnose|status\s+of|how\s+is|how's|"
        r"what\s+about|tell\s+me\s+about|my)\s+(?:my\s+)?([a-z0-9/_-]+)\s+"
        r"(?:trade|position|pos|bag|coin|perp)\b",
        q_low,
    )
    if m:
        cand = m.group(1).strip("?.,!:;\"'()[]")
        if cand and cand not in _NON_COIN_WORDS and not cand.isdigit():
            sym, _, _ = _resolve_symbol_snapshot(cand, ctx)
            return sym

    # 4. Explicit uppercase ticker token in a mixed-case question (e.g. "What about NOTACOIN?")
    if any(c.islower() for c in q_raw):
        for raw_tok in re.findall(r"\b[A-Z][A-Z0-9]{1,15}\b", q_raw):
            if raw_tok in _COMMON_ACRONYMS or raw_tok.lower() in _NON_COIN_WORDS:
                continue
            sym, _, _ = _resolve_symbol_snapshot(raw_tok, ctx)
            return sym

    # 5. Single bare token prompt (e.g. "notacoin" or "pepe")
    bare_tokens = [t for t in re.findall(r"[a-z0-9/_-]+", q_low) if t not in _NON_COIN_WORDS]
    if len(bare_tokens) == 1 and len(re.findall(r"[a-z0-9]+", q_low)) == 1:
        cand = bare_tokens[0]
        if cand not in _NON_COIN_WORDS and not cand.isdigit() and 2 <= len(cand) <= 20:
            sym, _, _ = _resolve_symbol_snapshot(cand, ctx)
            return sym

    return None


def _append_notice(lines: list[str], notice: str | None) -> None:
    if notice:
        lines.append(f"> **Offline Coach Notice:** {notice}\n")


def _format_position_fix_report(
    ctx: CoachContext,
    matched: Sequence[PositionAnalysis],
    *,
    notice: str | None = None,
) -> str:
    """Intent (a): Diagnosis + concrete leverage, stop, and risk-sized quantity fixes."""
    lines: list[str] = []
    _append_notice(lines, notice)
    k_buf = float(ctx.k_atr_buffer)
    max_risk_pct = float(ctx.rules.max_risk_per_trade_pct)
    max_risk_usd = float(ctx.account.equity) * (max_risk_pct / 100.0)
    total_committed = sum(committed_margin(x) for x in ctx.analyses)

    for idx, a in enumerate(matched):
        if idx > 0:
            lines.append("")
        lines.append(f"### Position Diagnosis & Concrete Fixes — {a.symbol} (Deterministic Offline Coach)")
        stop_txt = _fmt_price(a.stop_price) if a.stop_price is not None else "NONE"
        lp_txt = liquidation_label(a)
        dist_txt = (
            "LIQUIDATED"
            if a.liquidated
            else (
                f"{a.distance_to_liq_pct:.2f}% ({a.distance_to_liq_atr:.2f} daily ATR)"
                if a.distance_to_liq_pct is not None and a.distance_to_liq_atr is not None
                else lp_txt
            )
        )
        risk_txt = (
            f"${a.risk_to_stop_usd:,.2f} ({a.risk_to_stop_pct:.2f}% equity)"
            if a.risk_to_stop_usd is not None and a.risk_to_stop_pct is not None
            else "UNBOUNDED (no protective stop)"
        )

        lines.append("#### Diagnosis")
        lines.append(
            f"- **Position:** {a.side.upper()} `{a.quantity:,.6g}` {a.symbol} @ entry {_fmt_price(a.entry_price)} | "
            f"mark {_fmt_price(a.mark_price)} | notional `${a.notional:,.2f}` | "
            f"**{a.leverage:.1f}x {a.margin_mode}** (effective `{effective_leverage_label(a)}`)"
        )
        lines.append(
            f"- **PnL & Risk:** uPnL `${a.unrealized_pnl:+,.2f}` (`{a.unrealized_pnl_pct:+.2f}%` ROE) | "
            f"Stop `{stop_txt}` | Liquidation `{lp_txt}` | Distance to LP `{dist_txt}` | "
            f"Risk to Stop `{risk_txt}` | Status **{a.danger_level}**"
        )
        if a.warnings:
            for w in a.warnings:
                lines.append(f"- **Flag:** {w}")
        else:
            lines.append("- **Flags:** No critical position flags active.")

        for r in ctx.rule_evals:
            if r.status != "OK" and a.symbol in r.affected_symbols:
                lines.append(f"- **Rule {r.status} ({r.name}):** {r.explanation} (Current `{r.current_value}` vs Limit `{r.limit_value}`).")

        # Determine reference stop for max-safe-leverage & quantity calculations
        has_valid_active_stop = (
            a.stop_price is not None
            and not a.missing_stop
            and not a.invalid_stop_direction
        )
        atr = max(1e-9, float(a.atr_1d))
        fallback_stop = (
            max(a.mark_price * 0.01, a.mark_price - 2.0 * atr)
            if a.side == "long"
            else a.mark_price + 2.0 * atr
        )
        ref_stop = float(a.stop_price) if has_valid_active_stop and a.stop_price is not None else fallback_stop
        stop_label = "current stop" if has_valid_active_stop else "suggested 2-ATR stop"

        other_mm = max(0.0, total_committed - committed_margin(a))
        directional_for_entry = (a.side == "long" and ref_stop < a.entry_price) or (
            a.side == "short" and ref_stop > a.entry_price
        )
        if directional_for_entry:
            recs = recommend_sizes(
                symbol=a.symbol,
                side=a.side,
                entry_price=a.entry_price,
                stop_price=ref_stop,
                account=ctx.account,
                atr_1d=atr,
                risk_pct=max_risk_pct,
                leverage=a.leverage,
                margin_mode=a.margin_mode,
                k_atr_buffer=k_buf,
                other_positions_mm=other_mm,
            )
            max_lev_rec = recs[-1]
            safe_lev = float(max_lev_rec.max_safe_leverage) if max_lev_rec.max_safe_leverage is not None else max_safe_leverage_for_stop(
                entry_price=a.entry_price,
                stop_price=ref_stop,
                atr_1d=atr,
                side=a.side,
                mmr=a.mmr,
                k_atr=k_buf,
            )
        else:
            safe_lev = max_safe_leverage_for_stop(
                entry_price=a.entry_price,
                stop_price=ref_stop,
                atr_1d=atr,
                side=a.side,
                mmr=a.mmr,
                k_atr=k_buf,
            )

        lines.append("\n#### Concrete Fixes")
        # 1. Max safe leverage
        rule_capped_lev = min(safe_lev, float(ctx.rules.max_leverage))
        lev_note = f" (and rule cap is `{ctx.rules.max_leverage:.1f}x` -> target `<= {rule_capped_lev:.2f}x`)" if safe_lev > ctx.rules.max_leverage else ""
        lines.append(
            f"1. **Max Safe Leverage:** **`{safe_lev:.2f}x`** keeps liquidation at least `{k_buf:.2f}` daily ATRs "
            f"(`{_fmt_price(atr)}`/day) beyond your {stop_label} (`{_fmt_price(ref_stop)}`){lev_note}."
        )

        # 2. Safe stop price before liquidation (at current leverage)
        if a.liquidation_price is not None:
            lp = float(a.liquidation_price)
            buffered_stop = lp + k_buf * atr if a.side == "long" else lp - k_buf * atr
            protective_side = (
                (a.side == "long" and buffered_stop < a.mark_price)
                or (a.side == "short" and buffered_stop > a.mark_price)
            )
            if protective_side and buffered_stop > 0:
                lines.append(
                    f"2. **Safe Stop Price Before Liquidation (at {a.leverage:.1f}x):** Place stop at "
                    f"**`{_fmt_price(buffered_stop)}`** (keeps `{k_buf:.2f}` ATR buffer inside liquidation `{_fmt_price(lp)}`)."
                )
            else:
                side_cmp = "above" if a.side == "long" else "below"
                tight_mid = (lp + a.mark_price) / 2.0
                lines.append(
                    f"2. **Safe Stop Price Before Liquidation (at {a.leverage:.1f}x):** Current liquidation `{_fmt_price(lp)}` "
                    f"is only `{a.distance_to_liq_atr or 0.0:.2f}` ATR from mark (`{_fmt_price(a.mark_price)}`), so a `{k_buf:.2f}` ATR "
                    f"buffer (`{_fmt_price(buffered_stop)}`) lies past mark. Either cut leverage to **`<= {safe_lev:.2f}x`** "
                    f"for your `{_fmt_price(ref_stop)}` stop, or place an immediate tight stop strictly {side_cmp} `{_fmt_price(lp)}` "
                    f"(e.g. `{_fmt_price(tight_mid)}` between LP and mark)."
                )
        else:
            lines.append(
                f"2. **Stop Placement:** Liquidation is `{lp_txt}`; maintain a protective stop at **`{_fmt_price(ref_stop)}`**."
            )

        # 3. Quantity within max_risk_per_trade_pct
        stop_dist_mark = max(1e-9, abs(a.mark_price - ref_stop))
        max_qty_rule = max_risk_usd / stop_dist_mark
        max_notional_rule = max_qty_rule * a.mark_price
        if a.quantity > max_qty_rule:
            trim_qty = a.quantity - max_qty_rule
            trim_pct = (trim_qty / a.quantity) * 100.0
            lines.append(
                f"3. **Risk-Compliant Quantity (Max {max_risk_pct:.2f}% / ${max_risk_usd:,.2f} risk):** "
                f"Reduce size to **`{max_qty_rule:,.4f}` {a.symbol}** (`${max_notional_rule:,.2f}` notional) — "
                f"trim `{trim_qty:,.4f}` units (`{trim_pct:.1f}%`) at {stop_label} `{_fmt_price(ref_stop)}`."
            )
        else:
            lines.append(
                f"3. **Risk-Compliant Quantity (Max {max_risk_pct:.2f}% / ${max_risk_usd:,.2f} risk):** "
                f"Current size `{a.quantity:,.4f}` is within the max-risk limit at {stop_label} `{_fmt_price(ref_stop)}` "
                f"(cap is `{max_qty_rule:,.4f}` units / `${max_notional_rule:,.2f}`)."
            )

    lines.append(f"\n*RiskGuard Offline Report — {DISCLAIMER}*")
    return "\n".join(lines)


def _format_btc_shock_report(
    ctx: CoachContext,
    shock_frac: float,
    *,
    notice: str | None = None,
) -> str:
    """Intent (b): Custom BTC shock through portfolio stress engine with beta propagation & liquidations."""
    lines: list[str] = []
    _append_notice(lines, notice)
    shock_pct = shock_frac * 100.0
    label = f"BTC {shock_pct:+.1f}%"
    lines.append(f"### Custom Stress Scenario — {label} (Deterministic Offline Coach)")

    if not ctx.analyses:
        lines.append(f"- **Portfolio PnL:** `$0.00` (`0.00%` of `${ctx.account.equity:,.2f}` equity)")
        lines.append("- **Liquidated Symbols:** None (no open positions).")
        lines.append(f"\n*RiskGuard Offline Report — {DISCLAIMER}*")
        return "\n".join(lines)

    betas = ctx.portfolio.betas_vs_btc
    shocks = {a.symbol: betas.get(a.symbol, 1.0) * shock_frac for a in ctx.analyses}
    res = _evaluate_shock_scenario(
        name=label,
        description=f"Custom BTC {shock_pct:+.1f}% shock with 90d hourly beta propagation.",
        analyses=ctx.analyses,
        account=ctx.account,
        shocks_by_symbol=shocks,
    )

    liq_text = ", ".join(res.liquidated_symbols) if res.liquidated_symbols else "None"
    lines.append(
        f"- **Portfolio PnL:** **`${res.portfolio_pnl_usd:+,.2f}` (`{res.portfolio_pnl_pct:+.2f}%` equity)**"
    )
    lines.append(
        f"- **Shocked Account Equity:** `${res.shocked_equity_usd:,.2f}` (from `${ctx.account.equity:,.2f}`)"
    )
    lines.append(f"- **Liquidated Symbols ({res.liquidation_count}):** **{liq_text}**")

    lines.append("\n#### Per-Position Beta Propagation & Liquidation Check")
    for a in ctx.analyses:
        beta = betas.get(a.symbol, 1.0)
        pos_shock = shocks.get(a.symbol, shock_frac)
        shocked_px = max(1e-9, a.mark_price * (1.0 + pos_shock))
        sign = 1.0 if a.side == "long" else -1.0
        raw_pnl = sign * a.quantity * (shocked_px - a.mark_price)
        if a.margin_mode == "isolated":
            pos_pnl = max(raw_pnl, -max(0.0, a.margin_used + a.unrealized_pnl))
            cap_note = " (capped at isolated equity)" if pos_pnl > raw_pnl + 1e-6 else ""
        else:
            pos_pnl = raw_pnl
            cap_note = " (cross pool)"
        is_liq = a.symbol in res.liquidated_symbols
        liq_flag = " **LIQUIDATED**" if is_liq else ""
        lines.append(
            f"- **{a.symbol}** ({a.side.upper()} {a.leverage:.0f}x {a.margin_mode}): "
            f"90d beta `{beta:.2f}` -> shocked move `{pos_shock * 100:+.2f}%` "
            f"(`{_fmt_price(a.mark_price)}` -> `{_fmt_price(shocked_px)}`) | "
            f"PnL `${pos_pnl:+,.2f}`{cap_note}{liq_flag}"
        )

    lines.append(f"\n*RiskGuard Offline Report — {DISCLAIMER}*")
    return "\n".join(lines)


def _format_pre_trade_report(
    ctx: CoachContext,
    side: Side,
    raw_coin: str,
    requested_leverage: float | None,
    *,
    notice: str | None = None,
) -> str:
    """Intent (c): Pre-trade check (entry=mark, stop=2 ATR, lev=N or max, fixed-fractional notional)."""
    sym, snap, market_arg = _resolve_symbol_snapshot(raw_coin, ctx)
    if snap is None or snap.mark_price <= 0:
        return _format_unheld_unknown_report(ctx, sym, notice=notice, for_pre_trade=True)

    entry = float(snap.mark_price)
    atr = float(snap.atr_1d) if snap.atr_1d > 0 else entry * 0.03
    stop_dist = 2.0 * atr
    stop = (entry - stop_dist) if side == "long" else (entry + stop_dist)
    lev = float(requested_leverage) if requested_leverage is not None else float(ctx.rules.max_leverage)

    lines: list[str] = []
    _append_notice(lines, notice)
    lines.append(f"### Pre-Trade Check — {side.upper()} {sym} @ {lev:.1f}x (Deterministic Offline Coach)")

    if stop <= 0:
        lines.append("- **Verdict:** **NO-GO**")
        lines.append(
            f"- **Reason:** 2 daily ATR (`{_fmt_price(stop_dist)}`) >= entry mark (`{_fmt_price(entry)}`), "
            "so a 2-ATR long stop cannot sit at a positive price."
        )
        lines.append(f"\n*RiskGuard Offline Report — {DISCLAIMER}*")
        return "\n".join(lines)

    if lev < 1.0 or lev > 150.0:
        lines.append("- **Verdict:** **NO-GO**")
        lines.append(
            f"- **Reason:** Requested leverage `{lev:.1f}x` is outside supported perpetual bounds [1x, 150x] "
            f"(rule max is `{ctx.rules.max_leverage:.1f}x`)."
        )
        lines.append(f"\n*RiskGuard Offline Report — {DISCLAIMER}*")
        return "\n".join(lines)

    risk_pct = min(1.0, float(ctx.rules.max_risk_per_trade_pct))
    recs = recommend_sizes(
        symbol=sym,
        side=side,
        entry_price=entry,
        stop_price=stop,
        account=ctx.account,
        atr_1d=atr,
        risk_pct=risk_pct,
        leverage=lev,
        margin_mode="isolated",
        k_atr_buffer=float(ctx.k_atr_buffer),
    )
    ff = recs[0]
    proposed = PositionInput(
        symbol=sym,
        side=side,
        entry_price=entry,
        notional=ff.notional_usd,
        leverage=lev,
        margin_mode="isolated",
        stop_price=stop,
    )
    res = pre_trade_check(
        proposed=proposed,
        existing_positions=ctx.positions,
        account=ctx.account,
        market=market_arg,
        rules=ctx.rules,
        journal=ctx.journal,
        k_atr_buffer=float(ctx.k_atr_buffer),
    )

    lines.append(f"- **Verdict:** **`{res.verdict}`**")
    lines.append(
        f"- **Simulated Trade:** {side.upper()} `{ff.quantity:,.6g}` {sym} | "
        f"Entry (mark) `{_fmt_price(entry)}` | Stop (2 daily ATR) `{_fmt_price(stop)}` | "
        f"Leverage `{lev:.1f}x` isolated | Fixed-Fractional Notional `${ff.notional_usd:,.2f}` "
        f"({risk_pct:.2f}% risk = `${ff.risk_usd:,.2f}`)"
    )
    lines.append("- **Reasons:**")
    for reason in res.reasons:
        lines.append(f"  - {reason}")

    lines.append(
        f"- **Portfolio Impact:** 1d 95% Hist VaR `${res.before_var_95_usd:,.2f}` -> `${res.after_var_95_usd:,.2f}` | "
        f"Total Open Risk `{res.before_open_risk_pct:.2f}%` -> `{res.after_open_risk_pct:.2f}%` | "
        f"Proposed LP `{liquidation_label(res.proposed_analysis)}`"
    )
    if res.recommended_quantity is not None or res.recommended_leverage is not None:
        rec_parts: list[str] = []
        if res.recommended_quantity is not None:
            rec_parts.append(
                f"Max Recommended Qty `{res.recommended_quantity:,.6g}` (`${res.recommended_quantity * entry:,.2f}`)"
            )
        if res.recommended_leverage is not None:
            rec_parts.append(f"Max Safe Leverage `{res.recommended_leverage:.2f}x`")
        lines.append(f"- **Adjustments:** {' | '.join(rec_parts)}")

    lines.append(f"\n*RiskGuard Offline Report — {DISCLAIMER}*")
    return "\n".join(lines)


def _format_unheld_unknown_report(
    ctx: CoachContext,
    symbol: str,
    *,
    notice: str | None = None,
    for_pre_trade: bool = False,
) -> str:
    """Intent (d): Clear response when a coin is not held and/or unknown in the active feed."""
    lines: list[str] = []
    _append_notice(lines, notice)
    lines.append(f"### Unknown / Unheld Coin — {symbol} (Deterministic Offline Coach)")

    held_syms = [a.symbol for a in ctx.analyses]
    snap = ctx.market.snapshots.get(symbol) if ctx.market is not None else None
    avail_feed = sorted(ctx.market.snapshots.keys()) if ctx.market is not None else []

    if snap is not None and snap.mark_price > 0:
        lines.append(
            f"- **{symbol}** is **not currently held** in your open positions "
            f"(open positions: `{', '.join(held_syms) if held_syms else 'none'}`)."
        )
        lines.append(
            f"- Market feed data is loaded for **{symbol}** (mark `{_fmt_price(snap.mark_price)}`, "
            f"daily ATR `{_fmt_price(snap.atr_1d)}`). Ask e.g. `Can I long {symbol_to_hyperliquid_coin(symbol)} at 5x?` "
            "to run a pre-trade check."
        )
    else:
        reason = "cannot run pre-trade sizing without a mark price" if for_pre_trade else "is neither in your open positions nor in the active market feed"
        lines.append(
            f"- **{symbol}** is **unknown / unavailable** in the current offline/session data ({reason})."
        )
        lines.append(f"- **Open Positions:** `{', '.join(held_syms) if held_syms else 'none'}`")
        lines.append(
            f"- **Loaded Market Symbols:** `{', '.join(avail_feed) if avail_feed else 'none'}`"
        )
        lines.append(
            "- Switch **Market Data Mode** to **Live auto** in the sidebar to query live Binance/Hyperliquid prices, "
            "or enter a manual entry price on the **Pre-Trade & Sizing** page."
        )

    lines.append(f"\n*RiskGuard Offline Report — {DISCLAIMER}*")
    return "\n".join(lines)


def _build_full_briefing_report(
    ctx: CoachContext,
    user_question: str | None = None,
    notice: str | None = None,
) -> str:
    """Intent (e): Full prioritized portfolio, stress, rule, and discipline briefing."""
    p = ctx.portfolio
    cb = ctx.circuit_breaker
    lines: list[str] = []

    _append_notice(lines, notice)

    lines.append("### RiskGuard Risk & Discipline Briefing (Deterministic Offline Coach)")
    if user_question:
        lines.append(f"**Focus:** {user_question.strip()}\n")

    # Priority 1 & 2: Critical position dangers
    past_lp = [a for a in ctx.analyses if a.at_or_past_liquidation or a.liquidated]
    stop_beyond = [
        a for a in ctx.analyses if a.stop_beyond_liquidation and not a.at_or_past_liquidation and not a.liquidated
    ]
    critical_stop_lp = past_lp + stop_beyond
    crossed_stops = [a for a in ctx.analyses if a.invalid_stop_direction and a not in critical_stop_lp]
    missing_stops = [a for a in ctx.analyses if a.missing_stop and a not in critical_stop_lp]
    tight_lp = [
        a
        for a in ctx.analyses
        if a.distance_to_liq_atr is not None
        and 0 < a.distance_to_liq_atr < 1.5
        and a not in critical_stop_lp
    ]
    breaches = [r for r in ctx.rule_evals if r.status == "BREACH"]

    lines.append("#### 1. Prioritized Actions (Most Dangerous First)")
    action_idx = 1
    for a in past_lp:
        lines.append(
            f"{action_idx}. **CRITICAL — {a.symbol} ({a.side.upper()} {a.leverage:.0f}x {a.margin_mode}):** "
            f"Mark (${a.mark_price:,.4f}) is **at or past the liquidation price ({liquidation_label(a)})**. "
            "The position is being liquidated; close or add margin now."
        )
        action_idx += 1

    for a in stop_beyond:
        lines.append(
            f"{action_idx}. **CRITICAL — {a.symbol} ({a.side.upper()} {a.leverage:.0f}x {a.margin_mode}):** "
            f"Your stop (${a.stop_price:,.4f}) sits **beyond your liquidation price ({liquidation_label(a)})**. "
            f"Liquidation is only {a.distance_to_liq_pct:.2f}% away ({a.distance_to_liq_atr or 0:.2f} ATR), "
            "so you are liquidated before your stop order ever executes. Cut leverage or tighten the stop "
            "inside liquidation immediately."
        )
        action_idx += 1

    for a in crossed_stops:
        lines.append(
            f"{action_idx}. **CRITICAL — {a.symbol} ({a.side.upper()} {a.leverage:.0f}x {a.margin_mode}):** "
            f"Mark (${a.mark_price:,.4f}) has already crossed your stop (${a.stop_price:,.4f}), so the stop "
            f"is not protecting this ${a.notional:,.0f} position (liquidation {liquidation_label(a)}). "
            "Check that a live stop order exists on the exchange or close the position."
        )
        action_idx += 1

    for a in missing_stops:
        lines.append(
            f"{action_idx}. **HIGH — {a.symbol} ({a.side.upper()} {a.leverage:.0f}x):** "
            f"Missing stop-loss on ${a.notional:,.0f} notional exposure (liquidation {liquidation_label(a)}). "
            "Place a hard stop-loss sized to your max risk-per-trade rule immediately."
        )
        action_idx += 1

    for a in tight_lp:
        lines.append(
            f"{action_idx}. **HIGH — {a.symbol} Tight Liquidation Buffer:** "
            f"Liquidation ({liquidation_label(a)}) is only {a.distance_to_liq_atr:.2f} daily ATRs away "
            f"({a.distance_to_liq_pct:.2f}%). Reduce position size or add buffer."
        )
        action_idx += 1

    for r in breaches:
        if r.rule_id in ("liquidation_safety", "mandatory_stop"):
            continue
        lines.append(
            f"{action_idx}. **RULE BREACH — {r.name}:** {r.explanation} "
            f"(Current: {r.current_value} vs Limit: {r.limit_value})."
        )
        action_idx += 1

    if action_idx == 1:
        lines.append("1. No critical position or rule breaches detected. Maintain current stops and position limits.")

    # Portfolio & Stress summary
    lines.append("\n#### 2. Portfolio Exposure, VaR & Stress Reality")
    lines.append(
        f"- **Account Equity:** ${p.equity:,.2f} | **Gross:** ${p.gross_exposure_usd:,.0f} ({p.gross_leverage:.2f}x) | "
        f"**Net:** ${p.net_exposure_usd:+,.0f} ({p.net_leverage:+.2f}x) | "
        f"**BTC Beta-Adjusted Net:** ${p.btc_beta_adj_net_usd:+,.0f} ({p.btc_beta_adj_net_pct:+.1f}% equity)"
    )
    lines.append(
        f"- **1-Day 95% / 99% Historical VaR:** ${p.var_95_hist_usd:,.0f} / ${p.var_99_hist_usd:,.0f} "
        f"(Cornish-Fisher: ${p.var_95_cf_usd:,.0f} / ${p.var_99_cf_usd:,.0f}) | "
        f"**95% CVaR:** ${p.cvar_95_hist_usd:,.0f}"
    )
    lines.append(
        f"- **Worst-Case Known Stops:** ${p.known_stop_risk_usd:,.0f} ({p.known_stop_risk_pct:.2f}% equity) "
        f"| **Max 30d Pairwise Correlation:** {p.max_pairwise_corr_30d:.2f} | **HHI:** {p.hhi:.2f}"
    )
    if p.stress_scenarios:
        worst_all = min(p.stress_scenarios, key=lambda s: s.portfolio_pnl_usd)
        lines.append(
            f"- **Worst Stress Scenario:** {worst_all.scenario} -> ${worst_all.portfolio_pnl_usd:+,.0f} "
            f"({worst_all.portfolio_pnl_pct:+.1f}% of equity)."
        )
    liq_scenarios = [s for s in p.stress_scenarios if s.liquidation_count > 0]
    if liq_scenarios:
        worst_liq = max(liq_scenarios, key=lambda s: (s.liquidation_count, -s.portfolio_pnl_usd))
        first_liq = min(liq_scenarios, key=lambda s: -s.portfolio_pnl_usd)
        lines.append(
            f"- **Stress Liquidation Alert:** {len(liq_scenarios)} of {len(p.stress_scenarios)} stress scenarios "
            f"trigger forced liquidations. Mildest one: **{first_liq.scenario}** "
            f"(${first_liq.portfolio_pnl_usd:+,.0f}) liquidates {', '.join(first_liq.liquidated_symbols)}; "
            f"most liquidations: **{worst_liq.scenario}** ({', '.join(worst_liq.liquidated_symbols)})."
        )

    # Circuit breaker & Journal discipline
    lines.append("\n#### 3. Circuit Breaker & Discipline Diagnosis")
    lines.append(
        f"- **Trading Allowed Today:** {'YES' if cb.trading_allowed else 'NO — HALTED'} "
        f"(Status: **{cb.status}**, Session PnL: ${cb.daily_pnl_usd:+,.2f} [{cb.daily_pnl_pct:+.2f}%], "
        f"Drawdown: {cb.current_drawdown_pct:.2f}%"
        + (f", evaluated for {cb.reference_date}" if cb.reference_date else "")
        + ")"
    )
    if ctx.journal and ctx.journal.total_trades > 0:
        j = ctx.journal
        lines.append(
            f"- **Journal ({j.total_trades} trades):** Win Rate {j.win_rate * 100:.1f}%, "
            f"Profit Factor {j.profit_factor:.2f}, Expectancy ${j.expectancy_usd:+,.2f}/trade, "
            f"**Discipline Score: {j.discipline_score:.1f}/100 ({j.discipline_grade})**."
        )
        if j.episode_counts:
            flags = ", ".join(f"{k.replace('_', ' ')}: {v}" for k, v in sorted(j.episode_counts.items()))
            lines.append(f"- **Behavioural Red Flags:** {flags}.")
            if j.episode_counts.get("revenge_trading", 0) > 0 or j.episode_counts.get("size_escalation", 0) > 0:
                lines.append(
                    "  - *Rule of thumb:* Never size up after a loss to win back capital. "
                    "Step risk down on drawdowns rather than escalating."
                )

    if ctx.drawdown_plan is not None:
        dp = ctx.drawdown_plan
        lines.append(
            f"- **Drawdown Schedule:** Current DD {dp.current_dd_pct:.2f}% -> "
            f"Recommended risk per trade **{dp.recommended_risk_pct:.2f}%** "
            f"(Step-down {dp.step_down_multiplier:.2f}x, CPPI {dp.cppi_multiplier:.2f}x)."
        )

    lines.append(f"\n*RiskGuard Offline Report — {DISCLAIMER}*")
    return "\n".join(lines)


def build_offline_coach_report(
    ctx: CoachContext,
    user_question: str | None = None,
    notice: str | None = None,
) -> str:
    """Route user questions deterministically across (a) position fix, (b) BTC shock, (c) pre-trade, (d) unknown coin, (e) briefing."""
    if not user_question or not user_question.strip():
        return _build_full_briefing_report(ctx, user_question=user_question, notice=notice)

    q_low = user_question.strip().lower()

    # (b) Custom BTC shock ("what if BTC drops 15%", "btc crash 20%", "btc -10")
    btc_shock = _parse_btc_shock(q_low)
    if btc_shock is not None:
        return _format_btc_shock_report(ctx, btc_shock, notice=notice)

    # (c) Pre-trade check ("can I open/buy/long/short <coin> [at N x]")
    pre_trade = _parse_pre_trade_query(q_low)
    if pre_trade is not None:
        side, coin_raw, lev = pre_trade
        return _format_pre_trade_report(ctx, side, coin_raw, lev, notice=notice)

    # (a) Mentions one of the user's open position symbols (full symbol or base coin)
    matched_positions = _match_held_positions(q_low, ctx.analyses)
    if matched_positions:
        return _format_position_fix_report(ctx, matched_positions, notice=notice)

    # (d) Coin not held / unknown
    unheld_sym = _extract_unheld_coin_query(user_question.strip(), q_low, ctx)
    if unheld_sym is not None:
        return _format_unheld_unknown_report(ctx, unheld_sym, notice=notice)

    # (e) Everything else -> full briefing
    return _build_full_briefing_report(ctx, user_question=user_question, notice=notice)


def run_coach_turn(
    ctx: CoachContext,
    history: list[dict[str, Any]],
    user_message: str,
    *,
    api_key: str | None = None,
    model: str | None = None,
    client_factory: Callable[..., anthropic.Anthropic] | None = None,
) -> CoachTurnResult:
    """Execute one conversational turn with Claude tool_runner or offline fallback.

    Preserves full assistant content on `history` for multi-turn continuity.
    """
    resolved_key = (api_key or os.getenv("ANTHROPIC_API_KEY") or "").strip()
    user_entry = {"role": "user", "content": user_message}
    history.append(user_entry)

    if not resolved_key:
        notice = "No Anthropic API key provided — using deterministic offline coach."
        report = build_offline_coach_report(ctx, user_question=user_message, notice=notice)
        assistant_content = [{"type": "text", "text": report}]
        history.append({"role": "assistant", "content": assistant_content})
        return CoachTurnResult(
            reply_text=report,
            assistant_content=assistant_content,
            tool_calls=[],
            used_fallback_offline=True,
            notice=notice,
            stop_reason="offline_no_key",
            model=None,
        )

    chosen_model = model or os.getenv("RG_MODEL", DEFAULT_MODEL)
    tools = build_coach_tools(ctx)
    tool_calls_log: list[dict[str, Any]] = []

    try:
        if client_factory is not None:
            client = client_factory(api_key=resolved_key)
        else:
            client = anthropic.Anthropic(api_key=resolved_key)

        runner = client.beta.messages.tool_runner(
            model=chosen_model,
            max_tokens=16000,
            max_iterations=16,
            system=COACH_PROMPT,
            output_config={"effort": "high"},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            tools=tools,
            messages=list(history),
        )

        last_message: Any = None
        for message in runner:
            last_message = message
            content = getattr(message, "content", []) or []
            for block in content:
                b_type = getattr(block, "type", None)
                if b_type == "tool_use":
                    tool_calls_log.append(
                        {
                            "tool": getattr(block, "name", "unknown"),
                            "input": getattr(block, "input", {}),
                            "id": getattr(block, "id", ""),
                        }
                    )
            if getattr(message, "stop_reason", None) == "refusal":
                notice = "Claude returned stop_reason='refusal'; switched to offline risk coach report."
                report = build_offline_coach_report(ctx, user_question=user_message, notice=notice)
                assistant_content = [{"type": "text", "text": report}]
                history.append({"role": "assistant", "content": assistant_content})
                return CoachTurnResult(
                    reply_text=report,
                    assistant_content=assistant_content,
                    tool_calls=tool_calls_log,
                    used_fallback_offline=True,
                    notice=notice,
                    stop_reason="refusal",
                    model=chosen_model,
                )

        if last_message is None:
            notice = "Claude tool runner returned no message; switched to offline risk coach report."
            report = build_offline_coach_report(ctx, user_question=user_message, notice=notice)
            assistant_content = [{"type": "text", "text": report}]
            history.append({"role": "assistant", "content": assistant_content})
            return CoachTurnResult(
                reply_text=report,
                assistant_content=assistant_content,
                tool_calls=tool_calls_log,
                used_fallback_offline=True,
                notice=notice,
                stop_reason="empty_runner",
                model=chosen_model,
            )

        raw_content = list(getattr(last_message, "content", []) or [])
        stop_reason = getattr(last_message, "stop_reason", None)
        reply_text = _extract_text_from_content(raw_content)
        if not reply_text:
            reply_text = build_offline_coach_report(
                ctx,
                user_question=user_message,
                notice="Model returned empty text content; rendered offline summary.",
            )
        elif "not financial advice" not in reply_text.lower():
            reply_text = f"{reply_text}\n\n*{DISCLAIMER}*"

        stored_content = _serialize_content_for_history(
            raw_content,
            strip_tool_use=(stop_reason == "tool_use"),
        )
        if not stored_content:
            stored_content = [{"type": "text", "text": reply_text}]
        history.append({"role": "assistant", "content": stored_content})

        return CoachTurnResult(
            reply_text=reply_text,
            assistant_content=stored_content,
            tool_calls=tool_calls_log,
            used_fallback_offline=False,
            notice=None,
            stop_reason=stop_reason,
            model=getattr(last_message, "model", chosen_model),
        )

    except anthropic.AnthropicError as exc:
        # Base class of every SDK error: connection/timeout, rate limit, auth, other HTTP status errors,
        # response validation and credential errors.
        logger.warning("Anthropic API error (%s); falling back to offline coach.", exc)
        notice = f"Anthropic API unavailable ({type(exc).__name__}: {exc}) — switched to deterministic offline coach."
        report = build_offline_coach_report(ctx, user_question=user_message, notice=notice)
        assistant_content = [{"type": "text", "text": report}]
        history.append({"role": "assistant", "content": assistant_content})
        return CoachTurnResult(
            reply_text=report,
            assistant_content=assistant_content,
            tool_calls=tool_calls_log,
            used_fallback_offline=True,
            notice=notice,
            stop_reason="api_error_fallback",
            model=chosen_model,
        )
