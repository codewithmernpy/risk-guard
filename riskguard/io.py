"""Input/output loaders and serializers for positions, journals, and YAML rules."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from riskguard.models import JournalTrade, PositionInput, RuleConfig

POSITION_COLUMNS = [
    "symbol",
    "side",
    "entry_price",
    "quantity",
    "leverage",
    "margin_mode",
    "stop_price",
    "take_profit",
    "opened_at",
]


def _read_text_or_path(source: str | Path | io.IOBase | bytes) -> str:
    if isinstance(source, (str, Path)):
        path = Path(source)
        if path.exists() and path.is_file():
            return path.read_text(encoding="utf-8")
        if isinstance(source, Path):
            raise FileNotFoundError(f"File not found: {path}")
        return str(source)
    if isinstance(source, bytes):
        return source.decode("utf-8")
    content = source.read()
    if isinstance(content, bytes):
        return content.decode("utf-8")
    return str(content)


def _clean_record(row: dict[str, Any]) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    for k, v in row.items():
        key = str(k).strip()
        if pd.isna(v):
            cleaned[key] = None
        else:
            cleaned[key] = v
    return cleaned


def positions_from_dataframe(df: pd.DataFrame) -> list[PositionInput]:
    """Validate and convert a DataFrame into a list of PositionInput models."""
    if df is None or df.empty:
        return []
    positions: list[PositionInput] = []
    for raw_row in df.to_dict(orient="records"):
        row = _clean_record(raw_row)
        if not row.get("symbol"):
            continue
        positions.append(PositionInput.model_validate(row))
    return positions


def positions_to_dataframe(positions: list[PositionInput]) -> pd.DataFrame:
    """Convert PositionInput models into a clean DataFrame suitable for st.data_editor."""
    if not positions:
        return pd.DataFrame(columns=POSITION_COLUMNS)
    rows = []
    for p in positions:
        rows.append(
            {
                "symbol": p.symbol,
                "side": p.side,
                "entry_price": float(p.entry_price),
                "quantity": float(p.quantity),
                "leverage": float(p.leverage),
                "margin_mode": p.margin_mode,
                "stop_price": float(p.stop_price) if p.stop_price is not None else None,
                "take_profit": float(p.take_profit) if p.take_profit is not None else None,
                "opened_at": p.opened_at.isoformat() if p.opened_at is not None else "",
            }
        )
    df = pd.DataFrame(rows, columns=POSITION_COLUMNS)
    return df


def load_positions_csv(source: str | Path | io.IOBase | bytes) -> list[PositionInput]:
    """Load positions from a CSV file path, text buffer, or raw CSV string."""
    text = _read_text_or_path(source)
    df = pd.read_csv(io.StringIO(text), comment="#")
    return positions_from_dataframe(df)


def load_positions_json(source: str | Path | io.IOBase | bytes) -> list[PositionInput]:
    """Load positions from a JSON file path, buffer, or JSON string."""
    text = _read_text_or_path(source)
    parsed = json.loads(text)
    if isinstance(parsed, dict):
        if "positions" in parsed and isinstance(parsed["positions"], list):
            parsed = parsed["positions"]
        else:
            parsed = [parsed]
    if not isinstance(parsed, list):
        raise ValueError("Positions JSON must be a list of position objects or {'positions': [...]}.")
    return [PositionInput.model_validate(_clean_record(item)) for item in parsed]


def journal_from_dataframe(df: pd.DataFrame) -> list[JournalTrade]:
    """Validate and convert a journal DataFrame into sorted JournalTrade records."""
    if df is None or df.empty:
        return []
    trades: list[JournalTrade] = []
    for raw_row in df.to_dict(orient="records"):
        row = _clean_record(raw_row)
        if not row.get("symbol"):
            continue
        trades.append(JournalTrade.model_validate(row))
    trades.sort(key=lambda t: t.closed_at)
    return trades


def load_journal_csv(source: str | Path | io.IOBase | bytes) -> list[JournalTrade]:
    """Load closed trades from a CSV file path, text buffer, or CSV string."""
    text = _read_text_or_path(source)
    df = pd.read_csv(io.StringIO(text), comment="#")
    return journal_from_dataframe(df)


def load_rules_yaml(source: str | Path | io.IOBase | bytes) -> RuleConfig:
    """Load RuleConfig from YAML file path, buffer, or YAML string."""
    text = _read_text_or_path(source)
    loaded = yaml.safe_load(text) or {}
    if not isinstance(loaded, dict):
        raise ValueError("Rules YAML must decode to a mapping/dictionary.")
    if "rules" in loaded and isinstance(loaded["rules"], dict):
        loaded = loaded["rules"]
    return RuleConfig.model_validate(loaded)


def dump_rules_yaml(rules: RuleConfig) -> str:
    """Serialize RuleConfig to YAML."""
    return yaml.safe_dump(rules.model_dump(), sort_keys=False)
