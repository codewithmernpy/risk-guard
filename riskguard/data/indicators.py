"""Candle resampling, ATR, returns, correlation, and BTC-beta calculations."""

from __future__ import annotations

import math
from typing import Mapping

import numpy as np
import pandas as pd


def resample_ohlcv(df_1h: pd.DataFrame, rule: str = "1D") -> pd.DataFrame:
    """Resample 1h OHLCV candles (with UTC open_time column) into a coarser timeframe."""
    if df_1h is None or df_1h.empty:
        return pd.DataFrame(columns=["open_time", "open", "high", "low", "close", "volume"])
    work = df_1h.copy()
    work["open_time"] = pd.to_datetime(work["open_time"], utc=True)
    work = work.sort_values("open_time").set_index("open_time")
    agg: dict[str, str] = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    if "quote_volume" in work.columns:
        agg["quote_volume"] = "sum"
    resampled = work.resample(rule, label="left", closed="left").agg(agg).dropna(subset=["close"])
    return resampled.reset_index()


def complete_daily_bars(df_1h: pd.DataFrame) -> pd.DataFrame:
    """Resample 1h candles to UTC daily bars, dropping a trailing day that has fewer than 24 hourly bars.

    A live feed's last day is still forming; including it would add a partial-day true range to ATR
    and a partial-day return to the VaR sample.
    """
    daily = resample_ohlcv(df_1h, rule="1D")
    if daily.empty:
        return daily
    times = pd.to_datetime(df_1h["open_time"], utc=True)
    last_day_start = pd.Timestamp(daily["open_time"].iloc[-1])
    if last_day_start.tzinfo is None:
        last_day_start = last_day_start.tz_localize("UTC")
    bars_in_last_day = int(((times >= last_day_start) & (times < last_day_start + pd.Timedelta(days=1))).sum())
    if bars_in_last_day < 24 and len(daily) > 1:
        daily = daily.iloc[:-1].reset_index(drop=True)
    return daily


def compute_atr(df: pd.DataFrame, period: int = 14) -> float:
    """Compute latest Wilder-smoothed Average True Range (ATR) on an OHLCV DataFrame."""
    if df is None or df.empty:
        return 0.0
    work = df.sort_values("open_time") if "open_time" in df.columns else df
    high = work["high"].astype(float).to_numpy()
    low = work["low"].astype(float).to_numpy()
    close = work["close"].astype(float).to_numpy()
    if len(close) == 1:
        return float(max(0.0, high[0] - low[0]))

    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum(
        high - low,
        np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)),
    )
    if len(tr) <= period:
        return float(np.mean(tr))

    # Wilder's smoothing seeded by first `period` simple mean
    atr = float(np.mean(tr[:period]))
    alpha = 1.0 / period
    for val in tr[period:]:
        atr = (1.0 - alpha) * atr + alpha * float(val)
    return float(max(0.0, atr))


def compute_daily_atr(df_1h: pd.DataFrame, period: int = 14) -> float:
    """Compute 1D ATR(period) from 1h candles, scaling 1h ATR when history is short."""
    if df_1h is None or df_1h.empty:
        return 0.0
    daily = complete_daily_bars(df_1h)
    if len(daily) >= period + 1:
        return compute_atr(daily, period=period)
    if len(daily) >= 3:
        return compute_atr(daily, period=min(period, len(daily) - 1))
    atr_1h = compute_atr(df_1h, period=min(period, max(1, len(df_1h) - 1)))
    return float(atr_1h * math.sqrt(24.0))


def compute_hourly_returns(
    candles_by_symbol: Mapping[str, pd.DataFrame],
    window_hours: int | None = 2160,
    log_returns: bool = False,
) -> pd.DataFrame:
    """Align 1h closes across symbols and compute trailing hourly return matrix."""
    series_map: dict[str, pd.Series] = {}
    for sym, df in candles_by_symbol.items():
        if df is None or df.empty:
            continue
        ts = pd.to_datetime(df["open_time"], utc=True)
        s = pd.Series(df["close"].astype(float).to_numpy(), index=ts, name=sym)
        s = s[~s.index.duplicated(keep="last")].sort_index()
        series_map[sym] = s

    if not series_map:
        return pd.DataFrame()

    prices = pd.DataFrame(series_map).sort_index()
    if window_hours is not None and len(prices) > window_hours + 1:
        prices = prices.iloc[-(window_hours + 1) :]
    prices = prices.ffill(limit=2)
    if len(prices) < 2:
        return pd.DataFrame(columns=prices.columns)
    if log_returns:
        rets = np.log(prices / prices.shift(1))
    else:
        rets = prices.pct_change()
    # First row is the undefined return of the first price; drop it (and any all-NaN rows) only
    return rets.iloc[1:].dropna(how="all")


def compute_daily_returns(
    candles_by_symbol: Mapping[str, pd.DataFrame],
    lookback_days: int = 365,
) -> pd.DataFrame:
    """Resample 1h candles to daily UTC closes and compute aligned daily simple returns."""
    series_map: dict[str, pd.Series] = {}
    for sym, df in candles_by_symbol.items():
        if df is None or df.empty:
            continue
        daily = complete_daily_bars(df)
        if daily.empty:
            continue
        ts = pd.to_datetime(daily["open_time"], utc=True)
        s = pd.Series(daily["close"].astype(float).to_numpy(), index=ts, name=sym)
        s = s[~s.index.duplicated(keep="last")].sort_index()
        series_map[sym] = s

    if not series_map:
        return pd.DataFrame()

    prices = pd.DataFrame(series_map).sort_index()
    if lookback_days > 0 and len(prices) > lookback_days + 1:
        prices = prices.iloc[-(lookback_days + 1) :]
    rets = prices.pct_change().iloc[1:].dropna(how="all")
    return rets


def compute_btc_betas(
    hourly_returns_df: pd.DataFrame,
    benchmark: str = "BTCUSDT",
    min_obs: int = 24,
) -> dict[str, float]:
    """Compute OLS beta of each symbol against BTCUSDT from hourly returns."""
    if hourly_returns_df is None or hourly_returns_df.empty:
        return {}
    betas: dict[str, float] = {}
    has_bench = benchmark in hourly_returns_df.columns
    for sym in hourly_returns_df.columns:
        if sym == benchmark:
            betas[sym] = 1.0
            continue
        if not has_bench:
            betas[sym] = 1.0
            continue
        pair = hourly_returns_df[[sym, benchmark]].dropna()
        if len(pair) < min_obs:
            betas[sym] = 1.0
            continue
        bench_var = float(pair[benchmark].var(ddof=1))
        if bench_var <= 1e-16:
            betas[sym] = 1.0
            continue
        cov = float(pair[sym].cov(pair[benchmark]))
        beta = cov / bench_var
        betas[sym] = float(beta) if np.isfinite(beta) else 1.0
    return betas


def compute_correlation_matrix(
    hourly_returns_df: pd.DataFrame,
    window_hours: int = 720,
) -> pd.DataFrame:
    """Compute Pearson correlation matrix over the trailing window_hours observations."""
    if hourly_returns_df is None or hourly_returns_df.empty:
        return pd.DataFrame()
    sub = hourly_returns_df.iloc[-window_hours:] if len(hourly_returns_df) > window_hours else hourly_returns_df
    if len(sub) < 3:
        cols = list(sub.columns)
        return pd.DataFrame(np.eye(len(cols)), index=cols, columns=cols)
    corr = sub.corr().fillna(0.0)
    arr = corr.to_numpy(dtype=float, copy=True)
    np.fill_diagonal(arr, 1.0)
    return pd.DataFrame(arr, index=corr.index, columns=corr.columns)
