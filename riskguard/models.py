"""Core Pydantic domain models and validation for RiskGuard."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

MarginMode = Literal["isolated", "cross"]
Side = Literal["long", "short"]
RuleStatus = Literal["OK", "WARN", "BREACH"]
Verdict = Literal["GO", "REDUCE SIZE", "NO-GO"]
DataSource = Literal["binance", "hyperliquid", "sample"]


def normalize_symbol(raw: str) -> str:
    """Normalize perpetual symbol names like 'btc', 'BTC-USDT', or 'ETH/USDT' -> 'BTCUSDT'."""
    cleaned = str(raw).strip().upper().replace("/", "").replace("-", "").replace("_", "")
    for suffix in ("PERP", "SWAP"):
        if cleaned.endswith(suffix):
            cleaned = cleaned[: -len(suffix)]
    if not cleaned:
        raise ValueError("Symbol cannot be empty.")
    if not cleaned.endswith(("USDT", "USDC", "BUSD", "USD")):
        cleaned = f"{cleaned}USDT"
    return cleaned


def normalize_side(raw: str) -> Side:
    """Normalize trade side aliases ('buy'/'long' -> 'long', 'sell'/'short' -> 'short')."""
    val = str(raw).strip().lower()
    if val in ("long", "buy", "l", "1", "+1"):
        return "long"
    if val in ("short", "sell", "s", "-1"):
        return "short"
    raise ValueError(f"Unsupported side '{raw}'. Expected 'long' or 'short'.")


def ensure_utc_datetime(val: Any) -> datetime:
    """Parse or coerce a timestamp into a timezone-aware UTC datetime."""
    if isinstance(val, datetime):
        if val.tzinfo is None:
            return val.replace(tzinfo=timezone.utc)
        return val.astimezone(timezone.utc)
    if isinstance(val, (int, float)):
        # Detect milliseconds vs seconds epoch
        ts = float(val)
        if ts > 1e12:
            ts /= 1000.0
        return datetime.fromtimestamp(ts, tz=timezone.utc)
    if isinstance(val, str):
        text = val.strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        dt = datetime.fromisoformat(text)
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    raise ValueError(f"Cannot parse datetime from {val!r}")


class MarginTier(BaseModel):
    """Single maintenance-margin bracket for USDT-M linear perpetuals."""

    max_notional: float = Field(..., gt=0, description="Upper notional bound in USDT")
    mmr: float = Field(..., ge=0, lt=1, description="Maintenance margin rate (decimal, e.g. 0.005)")
    cum_b: float = Field(0.0, ge=0, description="Continuity deduction amount (cumB) in USDT")


class AccountState(BaseModel):
    """Trader account equity and margin parameters."""

    equity: float = Field(10_000.0, gt=0, description="Total account equity in account currency")
    currency: str = Field("USDT", min_length=2, max_length=12)
    default_mmr: float = Field(0.005, ge=0.0, lt=0.5)


class PositionInput(BaseModel):
    """Normalized open perpetual position."""

    symbol: str
    side: Side
    entry_price: float = Field(..., gt=0)
    quantity: float = Field(..., gt=0)
    leverage: float = Field(1.0, ge=1.0, le=150.0)
    margin_mode: MarginMode = "isolated"
    stop_price: float | None = None
    take_profit: float | None = None
    opened_at: datetime | None = None
    notional_input: float | None = Field(default=None, exclude=True)

    @field_validator("symbol", mode="before")
    @classmethod
    def _clean_symbol(cls, v: Any) -> str:
        return normalize_symbol(str(v))

    @field_validator("side", mode="before")
    @classmethod
    def _clean_side(cls, v: Any) -> Side:
        return normalize_side(str(v))

    @field_validator("margin_mode", mode="before")
    @classmethod
    def _clean_margin_mode(cls, v: Any) -> MarginMode:
        val = str(v or "isolated").strip().lower()
        if val not in ("isolated", "cross"):
            raise ValueError(f"Unsupported margin_mode '{v}'. Expected 'isolated' or 'cross'.")
        return val  # type: ignore[return-value]

    @field_validator("stop_price", "take_profit", mode="before")
    @classmethod
    def _clean_optional_price(cls, v: Any) -> float | None:
        if v is None:
            return None
        if isinstance(v, str) and v.strip() == "":
            return None
        val = float(v)
        if val != val or val <= 0:  # catches NaN and non-positive
            return None
        return val

    @field_validator("opened_at", mode="before")
    @classmethod
    def _clean_opened_at(cls, v: Any) -> datetime | None:
        if v is None or (isinstance(v, str) and v.strip() == ""):
            return None
        try:
            return ensure_utc_datetime(v)
        except Exception:
            return None

    @model_validator(mode="before")
    @classmethod
    def _resolve_quantity_or_notional(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        payload = dict(data)
        qty = payload.get("quantity")
        notional = payload.get("notional")
        entry = payload.get("entry_price")
        # Missing = None, blank, NaN or exactly zero. A negative quantity (some exchanges export shorts
        # as negative size) is treated as its absolute value; the side column carries the direction.
        qty_missing = (
            qty is None
            or (isinstance(qty, str) and qty.strip() == "")
            or (isinstance(qty, float) and (qty != qty or qty == 0.0))
        )
        if qty_missing and notional is not None and entry is not None:
            n_val = float(notional)
            e_val = float(entry)
            if n_val > 0 and e_val > 0:
                payload["quantity"] = n_val / e_val
        elif not qty_missing:
            payload["quantity"] = abs(float(qty))
        return payload

    @property
    def side_sign(self) -> int:
        return 1 if self.side == "long" else -1


class JournalTrade(BaseModel):
    """Normalized closed trade record from trading journal."""

    opened_at: datetime
    closed_at: datetime
    symbol: str
    side: Side
    quantity: float = Field(..., gt=0)
    entry_price: float = Field(..., gt=0)
    exit_price: float = Field(..., gt=0)
    planned_stop: float | None = None
    fees: float = Field(0.0, ge=0.0)
    pnl: float = 0.0
    source: str | None = None

    @field_validator("symbol", mode="before")
    @classmethod
    def _clean_symbol(cls, v: Any) -> str:
        return normalize_symbol(str(v))

    @field_validator("side", mode="before")
    @classmethod
    def _clean_side(cls, v: Any) -> Side:
        return normalize_side(str(v))

    @field_validator("opened_at", "closed_at", mode="before")
    @classmethod
    def _clean_dt(cls, v: Any) -> datetime:
        return ensure_utc_datetime(v)

    @field_validator("planned_stop", mode="before")
    @classmethod
    def _clean_stop(cls, v: Any) -> float | None:
        if v is None or (isinstance(v, str) and v.strip() == ""):
            return None
        val = float(v)
        if val != val or val <= 0:
            return None
        return val

    @model_validator(mode="before")
    @classmethod
    def _compute_missing_pnl(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        payload = dict(data)
        fees = payload.get("fees", 0.0)
        if fees is None or (isinstance(fees, float) and fees != fees) or (isinstance(fees, str) and fees.strip() == ""):
            payload["fees"] = 0.0
        else:
            payload["fees"] = abs(float(fees))

        pnl = payload.get("pnl")
        pnl_missing = pnl is None or (isinstance(pnl, str) and pnl.strip() == "") or (isinstance(pnl, float) and pnl != pnl)
        if pnl_missing:
            side = normalize_side(str(payload["side"]))
            sign = 1 if side == "long" else -1
            qty = abs(float(payload["quantity"]))
            ep = float(payload["entry_price"])
            xp = float(payload["exit_price"])
            payload["pnl"] = sign * qty * (xp - ep) - payload["fees"]
        return payload

    @property
    def side_sign(self) -> int:
        return 1 if self.side == "long" else -1

    @property
    def entry_notional(self) -> float:
        return self.quantity * self.entry_price

    @property
    def hold_minutes(self) -> float:
        return max(0.0, (self.closed_at - self.opened_at).total_seconds() / 60.0)


class RuleConfig(BaseModel):
    """Configurable portfolio & behavioural risk rules."""

    max_risk_per_trade_pct: float = Field(2.0, gt=0, le=100)
    max_total_open_risk_pct: float = Field(6.0, gt=0, le=100)
    max_leverage: float = Field(10.0, ge=1.0, le=150)
    daily_loss_limit_pct: float = Field(3.0, gt=0, le=100)
    weekly_loss_limit_pct: float = Field(7.0, gt=0, le=100)
    max_drawdown_pct: float = Field(15.0, gt=0, le=100)
    max_positions: int = Field(5, ge=1, le=50)
    max_correlated_exposure_pct: float = Field(150.0, gt=0, le=2000)
    correlation_threshold: float = Field(0.70, ge=0.1, le=0.99)
    mandatory_stop: bool = True
    warn_fraction: float = Field(0.80, gt=0.1, lt=1.0)


class PositionAnalysis(BaseModel):
    """Full per-position risk & liquidation breakdown."""

    symbol: str
    side: Side
    margin_mode: MarginMode
    quantity: float
    entry_price: float
    mark_price: float
    notional: float
    margin_used: float
    leverage: float
    effective_leverage: float
    account_leverage_share: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    stop_price: float | None
    take_profit: float | None
    risk_to_stop_usd: float | None
    risk_to_stop_pct: float | None
    reward_to_risk: float | None
    funding_rate_8h: float
    funding_cost_daily_usd: float
    mmr: float
    cum_b: float
    liquidation_price: float | None
    liquidation_note: str | None = None
    distance_to_liq_pct: float | None
    distance_to_liq_atr: float | None
    atr_1d: float
    stop_beyond_liquidation: bool
    missing_stop: bool
    invalid_stop_direction: bool
    at_or_past_liquidation: bool
    liquidated: bool = False
    used_entry_proxy: bool = False
    margin_tier_source: str = ""
    warnings: list[str] = Field(default_factory=list)
    danger_level: Literal["OK", "WARN", "DANGER"] = "OK"


class StressScenarioResult(BaseModel):
    """Single stress scenario portfolio impact and liquidation check."""

    scenario: str
    description: str
    portfolio_pnl_usd: float
    portfolio_pnl_pct: float
    shocked_equity_usd: float
    liquidated_symbols: list[str] = Field(default_factory=list)
    liquidation_count: int = 0
    symbol_shocks_pct: dict[str, float] = Field(default_factory=dict)


class PortfolioSummary(BaseModel):
    """Aggregate portfolio exposure, VaR/CVaR, correlation, and stress outputs."""

    equity: float
    position_count: int
    gross_exposure_usd: float
    net_exposure_usd: float
    long_exposure_usd: float
    short_exposure_usd: float
    gross_leverage: float
    net_leverage: float
    liquidated_positions: int = 0
    max_effective_leverage: float = 0.0
    avg_effective_leverage: float = 0.0
    btc_beta_adj_net_usd: float
    btc_beta_adj_gross_usd: float
    btc_beta_adj_net_pct: float
    total_unrealized_pnl_usd: float
    total_daily_funding_cost_usd: float
    known_stop_risk_usd: float
    known_stop_risk_pct: float
    unstopped_positions: int
    unstopped_notional_usd: float
    worst_case_stops_usd: float
    worst_case_stops_pct: float
    hhi: float
    hhi_10000: float
    effective_num_positions: float
    max_pairwise_corr_30d: float
    max_pairwise_corr_90d: float
    var_95_hist_usd: float
    var_99_hist_usd: float
    cvar_95_hist_usd: float
    cvar_99_hist_usd: float
    var_95_cf_usd: float
    var_99_cf_usd: float
    cvar_95_cf_usd: float
    cvar_99_cf_usd: float
    var_lookback_days: int
    per_asset_exposure: dict[str, dict[str, float]] = Field(default_factory=dict)
    betas_vs_btc: dict[str, float] = Field(default_factory=dict)
    corr_30d: dict[str, dict[str, float]] = Field(default_factory=dict)
    corr_90d: dict[str, dict[str, float]] = Field(default_factory=dict)
    stress_scenarios: list[StressScenarioResult] = Field(default_factory=list)


class RuleEvaluation(BaseModel):
    """Result of evaluating one trading risk rule."""

    rule_id: str
    name: str
    status: RuleStatus
    current_value: str
    limit_value: str
    explanation: str
    affected_symbols: list[str] = Field(default_factory=list)


class CircuitBreakerStatus(BaseModel):
    """Daily/weekly/drawdown trading permission status."""

    trading_allowed: bool
    status: Literal["OPEN", "CAUTION", "TRIPPED"]
    daily_pnl_usd: float
    daily_pnl_pct: float
    weekly_pnl_usd: float
    weekly_pnl_pct: float
    current_drawdown_pct: float
    reference_date: str | None = None
    reasons: list[str] = Field(default_factory=list)


class SizingRecommendation(BaseModel):
    """Recommended position sizing for a specific sizing methodology."""

    method: str
    quantity: float
    notional_usd: float
    risk_usd: float
    risk_pct: float
    max_safe_leverage: float | None = None
    quantity_cap: float | None = None  # liquidation-safety size cap before any risk budget (Max-Leverage-Safe only)
    notes: str = ""


class PreTradeCheckResult(BaseModel):
    """Before/after risk evaluation of a proposed new trade."""

    verdict: Verdict
    reasons: list[str]
    proposed_analysis: PositionAnalysis
    before_var_95_usd: float
    after_var_95_usd: float
    before_var_99_usd: float
    after_var_99_usd: float
    before_open_risk_pct: float
    after_open_risk_pct: float
    before_gross_exposure_usd: float
    after_gross_exposure_usd: float
    before_net_exposure_usd: float
    after_net_exposure_usd: float
    before_btc_beta_exposure_usd: float
    after_btc_beta_exposure_usd: float
    before_breaches: list[str]
    after_breaches: list[str]
    recommended_quantity: float | None = None
    recommended_leverage: float | None = None
    sizing_options: list[SizingRecommendation] = Field(default_factory=list)


class BehaviourEpisode(BaseModel):
    """Detected behavioural discipline red flag in the trade journal."""

    episode_type: Literal[
        "revenge_trading",
        "overtrading",
        "size_escalation",
        "disposition_effect",
        "stop_violation",
        "tilt_session",
    ]
    timestamp: datetime
    symbol: str
    severity: Literal["LOW", "MEDIUM", "HIGH"]
    description: str
    metrics: dict[str, float | str] = Field(default_factory=dict)


class JournalAnalytics(BaseModel):
    """Complete trade journal performance and behavioural analysis."""

    total_trades: int
    win_count: int
    loss_count: int
    win_rate: float
    total_pnl_usd: float
    total_fees_usd: float
    avg_win_usd: float
    avg_loss_usd: float
    payoff_ratio: float
    expectancy_usd: float
    profit_factor: float
    avg_r: float | None
    expectancy_r: float | None
    r_count: int
    max_drawdown_pct: float
    max_drawdown_usd: float
    current_drawdown_pct: float
    max_drawdown_duration_hours: float
    current_drawdown_duration_hours: float
    avg_win_hold_minutes: float
    avg_loss_hold_minutes: float
    disposition_ratio: float
    today_pnl_usd: float
    week_pnl_usd: float
    reference_date: str
    discipline_score: float
    discipline_grade: str
    discipline_components: dict[str, dict[str, Any]]
    episodes: list[BehaviourEpisode] = Field(default_factory=list)
    episode_counts: dict[str, int] = Field(default_factory=dict)
    r_multiples: list[float] = Field(default_factory=list)
    by_symbol: list[dict[str, Any]] = Field(default_factory=list)
    by_weekday: list[dict[str, Any]] = Field(default_factory=list)
    by_hour: list[dict[str, Any]] = Field(default_factory=list)
    equity_curve: list[dict[str, Any]] = Field(default_factory=list)


class DrawdownScheduleTier(BaseModel):
    """Single tier in a drawdown-based risk step-down schedule."""

    min_dd_pct: float
    max_dd_pct: float
    risk_multiplier: float
    effective_risk_pct: float
    status: str


class DrawdownPlan(BaseModel):
    """Step-down and CPPI-style drawdown risk control plan."""

    base_risk_pct: float
    current_dd_pct: float
    max_dd_limit_pct: float
    step_down_multiplier: float
    step_down_risk_pct: float
    cppi_multiplier: float
    cppi_risk_pct: float
    cppi_cushion_usd: float
    recommended_risk_pct: float
    trading_halted: bool
    schedule_table: list[DrawdownScheduleTier]


class RiskOfRuinResult(BaseModel):
    """Analytical and Monte Carlo risk-of-ruin evaluation."""

    ruin_drawdown_pct: float
    risk_per_trade_pct: float
    win_rate: float
    payoff_ratio: float
    expectancy_r: float
    analytical_ruin_prob: float
    mc_ruin_prob_flat: float
    mc_ruin_prob_scheduled: float
    mc_median_terminal_equity: float
    mc_p05_terminal_equity: float
    mc_p95_max_dd_pct: float
    simulations: int
    horizon_trades: int
