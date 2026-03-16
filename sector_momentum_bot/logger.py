"""Structured logging and execution monitoring."""

import json
import logging
import os
from datetime import date, datetime
from pathlib import Path
from typing import Any, Optional

from sector_momentum_bot.rebalancer import RebalanceResult

logger = logging.getLogger(__name__)

DEFAULT_LOG_DIR = Path("logs")


def setup_logging(level: str = "INFO", log_dir: Optional[Path] = None) -> None:
    """Configure console + file logging."""
    log_dir = log_dir or DEFAULT_LOG_DIR
    log_dir.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger("sector_momentum_bot")
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Console handler
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    root.addHandler(console)

    # File handler
    log_file = log_dir / f"sector_rotation_{date.today().isoformat()}.log"
    fh = logging.FileHandler(log_file)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    ))
    root.addHandler(fh)


def build_execution_record(
    rebalance: RebalanceResult,
    sector_rankings: list[tuple[str, float]],
    selected_sectors: list[str],
    execution_etfs: list[str],
    weights: dict[str, float],
    regime: str,
    spy_return: float,
    bil_return: float,
) -> dict[str, Any]:
    """Build a structured JSON record of a single execution."""
    return {
        "date": rebalance.timestamp.isoformat(),
        "spy_12m_return": round(spy_return, 6),
        "bil_12m_return": round(bil_return, 6),
        "regime": regime,
        "sector_rankings": [
            {"etf": s, "momentum": round(m, 6)} for s, m in sector_rankings
        ],
        "selected_sectors": selected_sectors,
        "execution_etfs": execution_etfs,
        "weights": {k: round(v, 6) for k, v in weights.items()},
        "previous_holdings": rebalance.previous_holdings,
        "new_holdings": rebalance.new_holdings,
        "trades_executed": [
            {
                "action": t.side.value.upper(),
                "symbol": t.symbol,
                "qty": t.qty,
                "reason": t.reason,
            }
            for t in rebalance.trades
        ],
        "account_value_before": round(rebalance.account_value_before, 2),
        "account_value_after": round(rebalance.account_value_after, 2),
        "errors": rebalance.errors,
    }


def save_execution_record(
    record: dict[str, Any],
    log_dir: Optional[Path] = None,
) -> Path:
    """Save execution record to a JSON file."""
    log_dir = log_dir or DEFAULT_LOG_DIR
    log_dir.mkdir(parents=True, exist_ok=True)

    filename = f"execution_{record['date'][:10]}.json"
    path = log_dir / filename

    # Append to a JSONL file for history
    history_path = log_dir / "execution_history.jsonl"
    with open(history_path, "a") as f:
        f.write(json.dumps(record) + "\n")

    # Also write the latest as standalone JSON
    with open(path, "w") as f:
        json.dump(record, f, indent=2)

    logger.info("Execution record saved to %s", path)
    return path
