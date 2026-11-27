"""Market data providers: Binance USDT-M REST -> Hyperliquid REST -> Bundled Sample."""

from __future__ import annotations

from datetime import datetime, timezone
import re
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Sequence

import pandas as pd
import requests
from pydantic import BaseModel, Field

from riskguard.data.cache import TTLCache
from riskguard.data.indicators import compute_daily_atr
from riskguard.logging_config import get_logger
from riskguard.models import DataSource, normalize_symbol

logger = get_logger("riskguard.providers")

BINANCE_BASE_URL = "https://fapi.binance.com"
HYPERLIQUID_INFO_URL = "https://api.hyperliquid.xyz/info"

DEFAULT_DEMO_SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "XRPUSDT",
    "DOGEUSDT",
    "BNBUSDT",
]

CANDLE_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_volume",
    "trades",
    "taker_buy_base",
]

DEFAULT_DEMO_FUNDING_8H = 0.0001  # 0.01% per 8h standard baseline
USDT_M_SYMBOL_PATTERN = re.compile(r"^[A-Z0-9]{2,20}USDT$")
BINANCE_MULTIPLIER_PREFIXES = ("1000", "1000000", "1M", "10000", "100", "10")


class SymbolResolution(BaseModel):
    """Resolved and validated USDT-M perpetual symbol."""

    symbol: str
    input_raw: str
    validated_against_live_list: bool = False
    used_fallback_format_only: bool = False
    note: str | None = None


def validate_symbol_format(raw: str) -> str:
    """Normalize user input and validate ^[A-Z0-9]{2,20}USDT$ format for offline/demo mode."""
    norm = normalize_symbol(raw)
    if not USDT_M_SYMBOL_PATTERN.fullmatch(norm):
        raise ValueError(
            f"Invalid USDT-M symbol '{raw}' (normalized '{norm}'). "
            "Expected 2-20 alphanumeric characters followed by USDT (e.g. 'LINK', 'pepe', 'LINKUSDT')."
        )
    return norm


def match_binance_usdtm_symbol(raw: str, valid_symbols: Iterable[str]) -> str:
    """Resolve user input (e.g. 'pepe' -> '1000PEPEUSDT' or 'PEPEUSDT') against active Binance symbols."""
    norm = normalize_symbol(raw)
    valid_set = {str(s).strip().upper() for s in valid_symbols if str(s).strip()}
    if norm in valid_set:
        return norm
    if norm.endswith("USDT"):
        base = norm[:-4]
        if not re.fullmatch(r"[A-Z0-9]{1,20}", base):
            raise ValueError(f"Invalid symbol '{raw}'.")
        for prefix in BINANCE_MULTIPLIER_PREFIXES:
            candidate = f"{prefix}{base}USDT"
            if candidate in valid_set:
                return candidate
    raise ValueError(f"Symbol '{norm}' was not found among active Binance USDT-M perpetual contracts.")


class ProviderError(RuntimeError):
    """Raised when a remote market data provider cannot serve a request."""


# Errors that mean "this feed cannot be used right now": transport/HTTP errors (ProviderError) and
# malformed payloads (missing keys, None/garbage numbers, wrong shapes) raised while parsing.
_FEED_ERRORS: tuple[type[Exception], ...] = (ProviderError, KeyError, TypeError, ValueError, IndexError, AttributeError)


class SymbolMarketData(BaseModel):
    """Snapshot market metrics for one perpetual symbol."""

    symbol: str
    mark_price: float
    index_price: float | None = None
    funding_rate_8h: float = DEFAULT_DEMO_FUNDING_8H
    funding_rate_daily: float = DEFAULT_DEMO_FUNDING_8H * 3.0
    price_change_24h_pct: float = 0.0
    atr_1d: float = 0.0
    source: DataSource = "sample"
    as_of: datetime


class MarketDataBundle(BaseModel):
    """Complete fetched market data bundle with provenance metadata."""

    model_config = {"arbitrary_types_allowed": True}

    source: DataSource
    badge_label: str
    reason: str
    is_live: bool
    fetched_at: datetime
    snapshots: dict[str, SymbolMarketData] = Field(default_factory=dict)
    candles: dict[str, pd.DataFrame] = Field(default_factory=dict)
    missing_symbols: list[str] = Field(default_factory=list)


def symbol_to_hyperliquid_coin(symbol: str) -> str:
    """Convert Binance-style symbol ('BTCUSDT') to Hyperliquid coin ('BTC')."""
    norm = normalize_symbol(symbol)
    for quote in ("USDT", "USDC", "BUSD", "USD"):
        if norm.endswith(quote) and len(norm) > len(quote):
            return norm[: -len(quote)]
    return norm


def parse_binance_klines(raw_klines: list[list[Any]]) -> pd.DataFrame:
    """Parse Binance /fapi/v1/klines JSON array into standard OHLCV DataFrame."""
    if not raw_klines:
        return pd.DataFrame(columns=CANDLE_COLUMNS)
    rows = []
    for k in raw_klines:
        rows.append(
            {
                "open_time": pd.to_datetime(int(k[0]), unit="ms", utc=True),
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": float(k[5]),
                "quote_volume": float(k[7]) if len(k) > 7 else 0.0,
                "trades": int(k[8]) if len(k) > 8 else 0,
                "taker_buy_base": float(k[9]) if len(k) > 9 else 0.0,
            }
        )
    df = pd.DataFrame(rows, columns=CANDLE_COLUMNS)
    df = df.drop_duplicates(subset=["open_time"]).sort_values("open_time").reset_index(drop=True)
    return df


def parse_hyperliquid_candles(raw_candles: list[dict[str, Any]]) -> pd.DataFrame:
    """Parse Hyperliquid candleSnapshot JSON into standard OHLCV DataFrame."""
    if not raw_candles:
        return pd.DataFrame(columns=CANDLE_COLUMNS)
    rows = []
    for c in raw_candles:
        close_px = float(c["c"])
        vol = float(c.get("v", 0.0))
        rows.append(
            {
                "open_time": pd.to_datetime(int(c["t"]), unit="ms", utc=True),
                "open": float(c["o"]),
                "high": float(c["h"]),
                "low": float(c["l"]),
                "close": close_px,
                "volume": vol,
                "quote_volume": vol * close_px,
                "trades": int(c.get("n", 0)),
                "taker_buy_base": 0.0,
            }
        )
    df = pd.DataFrame(rows, columns=CANDLE_COLUMNS)
    df = df.drop_duplicates(subset=["open_time"]).sort_values("open_time").reset_index(drop=True)
    return df


def load_sample_candles(symbol: str, sample_dir: Path) -> pd.DataFrame:
    """Load bundled {SYMBOL}_1h.csv.gz sample data safely without modifying source files."""
    norm = normalize_symbol(symbol)
    path = sample_dir / f"{norm}_1h.csv.gz"
    if not path.exists():
        raise FileNotFoundError(f"Bundled sample file not found for {norm}: {path}")
    df = pd.read_csv(path, compression="gzip")
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    for col in ("open", "high", "low", "close", "volume", "quote_volume", "taker_buy_base"):
        if col in df.columns:
            df[col] = df[col].astype(float)
    if "trades" in df.columns:
        df["trades"] = df["trades"].astype(int)
    df = df.sort_values("open_time").reset_index(drop=True)
    return df


class MarketDataProvider:
    """Resilient market data fetcher with Binance -> Hyperliquid -> Demo sample fallback."""

    def __init__(
        self,
        sample_dir: Path | str | None = None,
        session: requests.Session | None = None,
        timeout: float = 5.0,
        max_retries: int = 2,
        backoff_base: float = 0.4,
        cache: TTLCache | None = None,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        default_dir = Path(__file__).resolve().parents[2] / "data" / "sample"
        self.sample_dir = Path(sample_dir) if sample_dir is not None else default_dir
        self.session = session or requests.Session()
        self.timeout = float(timeout)
        self.max_retries = int(max_retries)
        self.backoff_base = float(backoff_base)
        self.cache = cache if cache is not None else TTLCache(default_ttl_seconds=60.0)
        self.sleep_fn = sleep_fn

    def _request_with_retry(
        self,
        method: str,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
    ) -> Any:
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = self.session.request(
                    method=method,
                    url=url,
                    params=params,
                    json=json_body,
                    timeout=self.timeout,
                )
            except requests.RequestException as exc:
                last_exc = exc
                raise ProviderError(f"Network error calling {url}: {exc}") from exc

            if resp.status_code == 429:
                if attempt >= self.max_retries:
                    raise ProviderError(f"HTTP 429 rate limited after {self.max_retries + 1} attempts at {url}")
                retry_after_raw = resp.headers.get("Retry-After")
                delay = self.backoff_base * (2**attempt)
                if retry_after_raw is not None:
                    try:
                        delay = min(10.0, max(0.01, float(retry_after_raw)))
                    except ValueError:
                        pass
                logger.warning("HTTP 429 from %s; retrying in %.2fs", url, delay)
                self.sleep_fn(delay)
                continue

            if resp.status_code == 451:
                raise ProviderError(f"HTTP 451 geo-restricted by {url}")
            if resp.status_code >= 400:
                raise ProviderError(f"HTTP {resp.status_code} from {url}: {resp.text[:200]}")

            try:
                return resp.json()
            except ValueError as exc:
                raise ProviderError(f"Invalid JSON from {url}: {exc}") from exc

        raise ProviderError(f"Request failed for {url}: {last_exc}")

    def _fetch_binance_usdtm_symbols(self) -> tuple[str, ...]:
        raw = self._request_with_retry("GET", f"{BINANCE_BASE_URL}/fapi/v1/exchangeInfo")
        if not isinstance(raw, dict) or not isinstance(raw.get("symbols"), list):
            raise ProviderError("Malformed Binance exchangeInfo response: missing 'symbols' list.")
        symbols: list[str] = []
        seen: set[str] = set()
        for item in raw["symbols"]:
            if not isinstance(item, dict):
                continue
            sym = str(item.get("symbol", "")).strip().upper()
            if not sym or not sym.endswith("USDT"):
                continue
            quote = item.get("quoteAsset")
            if quote is not None and str(quote).strip().upper() != "USDT":
                continue
            contract_type = item.get("contractType")
            if contract_type is not None and str(contract_type).strip().upper() != "PERPETUAL":
                continue
            status = item.get("status")
            if status is not None and str(status).strip().upper() != "TRADING":
                continue
            if sym not in seen:
                seen.add(sym)
                symbols.append(sym)
        if not symbols:
            raise ProviderError("Binance exchangeInfo returned no active USDT-M perpetual symbols.")
        return tuple(sorted(symbols))

    def get_binance_usdtm_symbols(self) -> tuple[str, ...]:
        """Fetch and cache active Binance USDT-M perpetual symbols from /fapi/v1/exchangeInfo."""
        cache_key = "binance:exchange_info:usdtm_symbols"
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached
        symbols = self._fetch_binance_usdtm_symbols()
        ttl = 3600.0 if self.cache.default_ttl_seconds > 0 else 0.0
        self.cache.set(cache_key, symbols, ttl_seconds=ttl)
        return symbols

    def resolve_symbol(
        self,
        raw: str,
        mode: Literal["auto", "demo"] = "auto",
    ) -> SymbolResolution:
        """Normalize and validate any Binance USDT-M perpetual symbol for Live or Demo mode."""
        if mode == "demo":
            sym = validate_symbol_format(raw)
            return SymbolResolution(
                symbol=sym,
                input_raw=raw,
                validated_against_live_list=False,
                used_fallback_format_only=False,
            )

        norm = normalize_symbol(raw)
        if not norm.endswith("USDT") or not re.fullmatch(r"[A-Z0-9]{1,20}USDT", norm):
            raise ValueError(
                f"Invalid USDT-M symbol '{raw}' (normalized '{norm}'). "
                "Expected alphanumeric coin/ticker ending in USDT (e.g. 'LINK', 'pepe', 'LINKUSDT')."
            )

        try:
            valid_symbols = self.get_binance_usdtm_symbols()
        except _FEED_ERRORS as exc:
            sym = validate_symbol_format(raw)
            return SymbolResolution(
                symbol=sym,
                input_raw=raw,
                validated_against_live_list=False,
                used_fallback_format_only=True,
                note=f"Live Binance symbol list unavailable ({self._describe_error('Binance', exc)}); validated format only.",
            )

        sym = match_binance_usdtm_symbol(raw, valid_symbols)
        return SymbolResolution(
            symbol=sym,
            input_raw=raw,
            validated_against_live_list=True,
            used_fallback_format_only=False,
        )

    def _fetch_binance_bundle(
        self,
        symbols: list[str],
        lookback_days: int,
    ) -> MarketDataBundle:
        premium_raw = self._request_with_retry("GET", f"{BINANCE_BASE_URL}/fapi/v1/premiumIndex")
        ticker_raw = self._request_with_retry("GET", f"{BINANCE_BASE_URL}/fapi/v1/ticker/24hr")

        premium_list = premium_raw if isinstance(premium_raw, list) else [premium_raw]
        ticker_list = ticker_raw if isinstance(ticker_raw, list) else [ticker_raw]

        premium_map = {str(item["symbol"]).upper(): item for item in premium_list if "symbol" in item}
        ticker_map = {str(item["symbol"]).upper(): item for item in ticker_list if "symbol" in item}

        now = datetime.now(timezone.utc)
        target_hours = max(24, lookback_days * 24)
        end_ms = int(now.timestamp() * 1000)

        snapshots: dict[str, SymbolMarketData] = {}
        candles: dict[str, pd.DataFrame] = {}
        missing: list[str] = []

        for sym in symbols:
            prem = premium_map.get(sym)
            tick = ticker_map.get(sym)
            if prem is None:
                missing.append(sym)
                continue

            frames: list[pd.DataFrame] = []
            cursor_end = end_ms
            remaining = target_hours
            prev_oldest: int | None = None

            while remaining > 0:
                limit = min(1500, remaining)
                raw_k = self._request_with_retry(
                    "GET",
                    f"{BINANCE_BASE_URL}/fapi/v1/klines",
                    params={
                        "symbol": sym,
                        "interval": "1h",
                        "limit": limit,
                        "endTime": cursor_end,
                    },
                )
                if not isinstance(raw_k, list) or not raw_k:
                    break
                df_page = parse_binance_klines(raw_k)
                frames.append(df_page)
                oldest_ms = int(raw_k[0][0])
                if len(raw_k) < limit or (prev_oldest is not None and oldest_ms >= prev_oldest):
                    break
                prev_oldest = oldest_ms
                cursor_end = oldest_ms - 1
                remaining -= len(raw_k)

            df_candles = (
                pd.concat(frames, ignore_index=True)
                .drop_duplicates(subset=["open_time"])
                .sort_values("open_time")
                .reset_index(drop=True)
                if frames
                else pd.DataFrame(columns=CANDLE_COLUMNS)
            )
            candles[sym] = df_candles
            mark_px = float(prem["markPrice"])
            idx_px = float(prem["indexPrice"]) if prem.get("indexPrice") else None
            fund_8h = float(prem.get("lastFundingRate", 0.0))
            chg_24h = float(tick.get("priceChangePercent", 0.0)) if tick else 0.0
            atr_1d = compute_daily_atr(df_candles) if not df_candles.empty else mark_px * 0.03
            as_of = (
                df_candles["open_time"].iloc[-1].to_pydatetime()
                if not df_candles.empty
                else now
            )
            snapshots[sym] = SymbolMarketData(
                symbol=sym,
                mark_price=mark_px,
                index_price=idx_px,
                funding_rate_8h=fund_8h,
                funding_rate_daily=fund_8h * 3.0,
                price_change_24h_pct=chg_24h,
                atr_1d=atr_1d,
                source="binance",
                as_of=as_of,
            )

        if not snapshots:
            raise ProviderError("Binance returned no matching symbols.")

        return MarketDataBundle(
            source="binance",
            badge_label="LIVE · Binance USDT-M",
            reason="Connected to Binance USDT-M Futures public REST",
            is_live=True,
            fetched_at=now,
            snapshots=snapshots,
            candles=candles,
            missing_symbols=missing,
        )

    def _fetch_hyperliquid_bundle(
        self,
        symbols: list[str],
        lookback_days: int,
        fallback_reason: str,
    ) -> MarketDataBundle:
        raw_meta = self._request_with_retry(
            "POST",
            HYPERLIQUID_INFO_URL,
            json_body={"type": "metaAndAssetCtxs"},
        )
        if not isinstance(raw_meta, list) or len(raw_meta) < 2:
            raise ProviderError("Unexpected Hyperliquid metaAndAssetCtxs response shape.")

        universe = raw_meta[0].get("universe", [])
        ctxs = raw_meta[1]
        coin_ctx: dict[str, dict[str, Any]] = {}
        coin_raw_name: dict[str, str] = {}
        for meta_item, ctx in zip(universe, ctxs):
            raw_name = str(meta_item.get("name", "")).strip()
            coin_name = raw_name.upper()
            if coin_name:
                coin_ctx[coin_name] = ctx
                coin_raw_name[coin_name] = raw_name

        now = datetime.now(timezone.utc)
        end_ms = int(now.timestamp() * 1000)
        start_ms = end_ms - max(24, lookback_days * 24) * 3600 * 1000

        snapshots: dict[str, SymbolMarketData] = {}
        candles: dict[str, pd.DataFrame] = {}
        missing: list[str] = []

        for sym in symbols:
            coin = symbol_to_hyperliquid_coin(sym)
            lookup_key = coin
            if lookup_key not in coin_ctx and coin.startswith("1000") and len(coin) > 4:
                alt_key = f"K{coin[4:]}"
                if alt_key in coin_ctx:
                    lookup_key = alt_key
            ctx = coin_ctx.get(lookup_key)
            if ctx is None:
                missing.append(sym)
                continue
            hl_coin = coin_raw_name.get(lookup_key, coin)

            raw_candles = self._request_with_retry(
                "POST",
                HYPERLIQUID_INFO_URL,
                json_body={
                    "type": "candleSnapshot",
                    "req": {
                        "coin": hl_coin,
                        "interval": "1h",
                        "startTime": start_ms,
                        "endTime": end_ms,
                    },
                },
            )
            df_candles = parse_hyperliquid_candles(raw_candles if isinstance(raw_candles, list) else [])
            candles[sym] = df_candles

            mark_px = float(ctx.get("markPx") or ctx.get("midPx") or ctx.get("oraclePx"))
            oracle_px = float(ctx["oraclePx"]) if ctx.get("oraclePx") else None
            prev_px = float(ctx["prevDayPx"]) if ctx.get("prevDayPx") else 0.0
            chg_24h = ((mark_px - prev_px) / prev_px * 100.0) if prev_px > 0 else 0.0
            # Hyperliquid funding is hourly
            funding_1h = float(ctx.get("funding", 0.0))
            atr_1d = compute_daily_atr(df_candles) if not df_candles.empty else mark_px * 0.03
            as_of = (
                df_candles["open_time"].iloc[-1].to_pydatetime()
                if not df_candles.empty
                else now
            )

            snapshots[sym] = SymbolMarketData(
                symbol=sym,
                mark_price=mark_px,
                index_price=oracle_px,
                funding_rate_8h=funding_1h * 8.0,
                funding_rate_daily=funding_1h * 24.0,
                price_change_24h_pct=chg_24h,
                atr_1d=atr_1d,
                source="hyperliquid",
                as_of=as_of,
            )

        if not snapshots:
            raise ProviderError("Hyperliquid returned no matching symbols.")

        return MarketDataBundle(
            source="hyperliquid",
            badge_label="LIVE · Hyperliquid Fallback",
            reason=f"{fallback_reason} -> fell back to Hyperliquid",
            is_live=True,
            fetched_at=now,
            snapshots=snapshots,
            candles=candles,
            missing_symbols=missing,
        )

    def _load_sample_bundle(
        self,
        symbols: list[str],
        reason: str = "Demo mode (offline bundled sample)",
    ) -> MarketDataBundle:
        now = datetime.now(timezone.utc)
        snapshots: dict[str, SymbolMarketData] = {}
        candles: dict[str, pd.DataFrame] = {}
        missing: list[str] = []

        for sym in symbols:
            try:
                df = self.cache.get_or_set(
                    f"sample_csv:{self.sample_dir}:{sym}",
                    lambda s=sym: load_sample_candles(s, self.sample_dir),
                    ttl_seconds=3600.0,
                )
            except FileNotFoundError:
                missing.append(sym)
                continue

            candles[sym] = df
            latest_close = float(df["close"].iloc[-1])
            prev_24h_close = float(df["close"].iloc[-25]) if len(df) >= 25 else float(df["close"].iloc[0])
            chg_24h = ((latest_close - prev_24h_close) / prev_24h_close * 100.0) if prev_24h_close > 0 else 0.0
            atr_1d = compute_daily_atr(df)
            as_of = df["open_time"].iloc[-1].to_pydatetime()

            snapshots[sym] = SymbolMarketData(
                symbol=sym,
                mark_price=latest_close,
                index_price=latest_close,
                funding_rate_8h=DEFAULT_DEMO_FUNDING_8H,
                funding_rate_daily=DEFAULT_DEMO_FUNDING_8H * 3.0,
                price_change_24h_pct=chg_24h,
                atr_1d=atr_1d,
                source="sample",
                as_of=as_of,
            )

        return MarketDataBundle(
            source="sample",
            badge_label="DEMO · Offline Sample (2023–2025)",
            reason=reason,
            is_live=False,
            fetched_at=now,
            snapshots=snapshots,
            candles=candles,
            missing_symbols=missing,
        )

    def get_market_data(
        self,
        symbols: Sequence[str],
        mode: Literal["auto", "demo"] = "auto",
        lookback_days: int = 120,
    ) -> MarketDataBundle:
        """Fetch market snapshots & hourly candles with caching and deterministic fallback."""
        norm_symbols = []
        seen = set()
        for raw in [*symbols, "BTCUSDT"]:
            sym = normalize_symbol(raw)
            if sym not in seen:
                seen.add(sym)
                norm_symbols.append(sym)

        cache_key = f"bundle:{mode}:{lookback_days}:{','.join(sorted(norm_symbols))}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached

        if mode == "demo":
            bundle = self._load_sample_bundle(norm_symbols, reason="Demo mode selected (offline sample)")
            self.cache.set(cache_key, bundle, ttl_seconds=300.0)
            return bundle

        binance_err = ""
        try:
            bundle = self._fetch_binance_bundle(norm_symbols, lookback_days=lookback_days)
            self._note_missing_symbols(bundle)
            self.cache.set(cache_key, bundle, ttl_seconds=60.0)
            return bundle
        except _FEED_ERRORS as exc:
            binance_err = self._describe_error("Binance", exc)
            logger.warning("Binance unavailable: %s", binance_err)

        try:
            bundle = self._fetch_hyperliquid_bundle(
                norm_symbols,
                lookback_days=lookback_days,
                fallback_reason=binance_err,
            )
            self._note_missing_symbols(bundle)
            self.cache.set(cache_key, bundle, ttl_seconds=60.0)
            return bundle
        except _FEED_ERRORS as exc:
            hl_err = self._describe_error("Hyperliquid", exc)
            logger.warning("Hyperliquid unavailable: %s", hl_err)
            reason = f"{binance_err}; {hl_err} -> fell back to bundled sample"
            bundle = self._load_sample_bundle(norm_symbols, reason=reason)
            self.cache.set(cache_key, bundle, ttl_seconds=60.0)
            return bundle

    @staticmethod
    def _describe_error(feed: str, exc: Exception) -> str:
        """ProviderError messages are already descriptive; malformed payloads are labelled as such."""
        if isinstance(exc, ProviderError):
            return str(exc)
        return f"Malformed {feed} response ({type(exc).__name__}: {exc})"

    @staticmethod
    def _note_missing_symbols(bundle: MarketDataBundle) -> None:
        """Record live-feed symbols that could not be served.

        They are deliberately NOT back-filled from the bundled sample: sample prices are historical
        (2023-2025) and would silently mix stale marks and non-overlapping history into a live bundle.
        Positions in these symbols fall back to entry price as mark and are flagged in their warnings.
        """
        if bundle.missing_symbols:
            bundle.reason = (
                f"{bundle.reason}; not on live feed: {', '.join(bundle.missing_symbols)} "
                "(entry price used as mark)"
            )
