"""Tests for market data providers using saved JSON fixtures and mocked HTTP errors (no network calls)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import requests

from riskguard.data.cache import TTLCache
from riskguard.data.providers import MarketDataProvider

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
SAMPLE_DIR = Path(__file__).resolve().parents[1] / "data" / "sample"


class _FakeResponse:
    def __init__(self, status_code: int, payload: Any = None, headers: dict[str, str] | None = None, text: str = "") -> None:
        self.status_code = status_code
        self._payload = payload
        self.headers = headers or {}
        self.text = text or (json.dumps(payload) if payload is not None else "")

    def json(self) -> Any:
        if self._payload is None:
            raise ValueError("No JSON payload")
        return self._payload


class _FakeSession:
    def __init__(self, handler) -> None:
        self._handler = handler
        self.calls: list[tuple[str, str]] = []

    def request(self, method: str, url: str, params=None, json=None, timeout=None):
        self.calls.append((method, url))
        return self._handler(method, url, params=params, json_body=json, timeout=timeout)


def _load_fixture(name: str) -> Any:
    return json.loads((FIXTURES_DIR / name).read_text(encoding="utf-8"))


def test_binance_fixture_parsing_and_429_retry() -> None:
    prem = _load_fixture("binance_premium_index.json")
    tick = _load_fixture("binance_ticker_24hr.json")
    klines = _load_fixture("binance_klines.json")
    slept: list[float] = []
    attempts = {"prem": 0}

    def handler(method, url, **kwargs):
        if "premiumIndex" in url:
            attempts["prem"] += 1
            if attempts["prem"] == 1:
                return _FakeResponse(429, headers={"Retry-After": "0.1"})
            return _FakeResponse(200, prem)
        if "ticker/24hr" in url:
            return _FakeResponse(200, tick)
        if "klines" in url:
            return _FakeResponse(200, klines)
        raise AssertionError(f"Unexpected URL: {url}")

    provider = MarketDataProvider(
        sample_dir=SAMPLE_DIR,
        session=_FakeSession(handler),  # type: ignore[arg-type]
        cache=TTLCache(default_ttl_seconds=0.0),
        sleep_fn=lambda d: slept.append(d),
    )
    bundle = provider.get_market_data(["BTCUSDT", "ETHUSDT"], mode="auto", lookback_days=5)

    assert bundle.source == "binance"
    assert bundle.is_live is True
    assert slept == [0.1]
    assert bundle.snapshots["BTCUSDT"].mark_price == pytest.approx(109500.0)
    assert bundle.snapshots["ETHUSDT"].funding_rate_8h == pytest.approx(0.00015)
    assert len(bundle.candles["BTCUSDT"]) == 3


def test_fallback_binance_451_to_hyperliquid_fixtures() -> None:
    hl_meta = _load_fixture("hyperliquid_meta.json")
    hl_candles = _load_fixture("hyperliquid_candles.json")

    def handler(method, url, json_body=None, **kwargs):
        if "fapi.binance.com" in url:
            return _FakeResponse(451, text="Service unavailable from restricted location")
        if "api.hyperliquid.xyz" in url:
            req_type = (json_body or {}).get("type")
            if req_type == "metaAndAssetCtxs":
                return _FakeResponse(200, hl_meta)
            if req_type == "candleSnapshot":
                return _FakeResponse(200, hl_candles)
        raise AssertionError(f"Unexpected call: {url}")

    provider = MarketDataProvider(
        sample_dir=SAMPLE_DIR,
        session=_FakeSession(handler),  # type: ignore[arg-type]
        cache=TTLCache(default_ttl_seconds=0.0),
    )
    bundle = provider.get_market_data(["BTCUSDT", "ETHUSDT"], mode="auto", lookback_days=5)

    assert bundle.source == "hyperliquid"
    assert bundle.is_live is True
    assert "451" in bundle.reason
    assert bundle.snapshots["BTCUSDT"].mark_price == pytest.approx(109480.0)
    # Hyperliquid hourly funding 0.0000125 -> 8h rate = 0.0001
    assert bundle.snapshots["BTCUSDT"].funding_rate_8h == pytest.approx(0.0001)
    assert len(bundle.candles["BTCUSDT"]) == 3


def test_fallback_timeout_to_bundled_sample_offline() -> None:
    def handler(method, url, **kwargs):
        raise requests.Timeout("Simulated timeout")

    provider = MarketDataProvider(
        sample_dir=SAMPLE_DIR,
        session=_FakeSession(handler),  # type: ignore[arg-type]
        cache=TTLCache(default_ttl_seconds=0.0),
    )
    bundle = provider.get_market_data(["BTCUSDT", "SOLUSDT"], mode="auto", lookback_days=30)

    assert bundle.source == "sample"
    assert bundle.is_live is False
    assert "SOLUSDT" in bundle.snapshots
    assert len(bundle.candles["BTCUSDT"]) > 20_000
