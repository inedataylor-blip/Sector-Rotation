"""Backtesting engine for the sector momentum strategy."""

import logging
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from sector_momentum_bot.brokers.backtest_broker import BacktestBroker
from sector_momentum_bot.config import DEFAULT_SECTORS, StrategyConfig
from sector_momentum_bot.rebalancer import execute_rebalance
from sector_momentum_bot.scheduler import get_monthly_rebalance_dates
from sector_momentum_bot.strategies.sector_rotation import generate_target_portfolio

logger = logging.getLogger(__name__)


@dataclass
class BacktestMetrics:
    """Performance metrics from a backtest."""
    start_date: date
    end_date: date
    initial_value: float
    final_value: float
    cagr: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    monthly_win_rate: float
    total_trades: int
    annual_turnover: float
    equity_curve: pd.Series


@dataclass
class BacktestResult:
    """Full backtest output."""
    metrics: BacktestMetrics
    monthly_returns: pd.Series
    holdings_history: list[dict]
    trade_log: list[dict]


def load_price_data_from_csv(
    data_dir: Path,
    symbols: Optional[list[str]] = None,
) -> dict[str, pd.DataFrame]:
    """
    Load price data from CSV files in a directory.

    Expects files named {SYMBOL}.csv with columns: date, open, high, low, close, volume.
    """
    data = {}
    symbols = symbols or []
    for csv_file in data_dir.glob("*.csv"):
        symbol = csv_file.stem.upper()
        if symbols and symbol not in symbols:
            continue
        df = pd.read_csv(csv_file, parse_dates=["date"])
        df["date"] = df["date"].dt.date
        df = df.set_index("date").sort_index()
        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        data[symbol] = df[["open", "high", "low", "close", "volume"]]
    return data


def download_price_data(
    symbols: list[str],
    start: date,
    end: date,
) -> dict[str, pd.DataFrame]:
    """
    Download price data using yfinance.

    Requires yfinance: pip install yfinance
    """
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("yfinance is required for downloading data: pip install yfinance")

    data = {}
    for symbol in symbols:
        logger.info("Downloading %s...", symbol)
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start.isoformat(), end=end.isoformat(), auto_adjust=True)
        if df.empty:
            logger.warning("No data for %s", symbol)
            continue
        df.index = df.index.date
        df.columns = [c.lower() for c in df.columns]
        data[symbol] = df[["open", "high", "low", "close", "volume"]]
    return data


def run_backtest(
    price_data: dict[str, pd.DataFrame],
    config: StrategyConfig,
    start: date,
    end: date,
    initial_cash: float = 100_000.0,
) -> BacktestResult:
    """
    Run a full backtest of the sector momentum strategy.

    Args:
        price_data: Dict of {symbol: DataFrame} with OHLCV data.
        config: Strategy configuration.
        start: Backtest start date.
        end: Backtest end date.
        initial_cash: Starting capital.

    Returns:
        BacktestResult with metrics, equity curve, and trade log.
    """
    broker = BacktestBroker(price_data, initial_cash=initial_cash)
    rebalance_dates = get_monthly_rebalance_dates(start, end)

    equity_values = []
    monthly_values = []
    holdings_history = []

    logger.info(
        "Backtest: %s to %s, %d rebalance dates",
        start, end, len(rebalance_dates),
    )

    for rebal_date in rebalance_dates:
        broker.set_current_date(rebal_date)

        try:
            target = generate_target_portfolio(broker, config, as_of=rebal_date)
            result = execute_rebalance(
                broker, target,
                rebalance_threshold=config.rebalance_threshold,
            )

            value = broker.get_account_value()
            equity_values.append((rebal_date, value))
            monthly_values.append(value)

            holdings = {
                "date": rebal_date.isoformat(),
                "value": round(value, 2),
                "positions": {s: q for s, q in broker._positions.items() if q > 0},
                "target": target,
            }
            holdings_history.append(holdings)

            logger.info(
                "%s: value=$%.2f, holdings=%s",
                rebal_date, value,
                list(target.keys()),
            )
        except Exception as e:
            logger.error("Error on %s: %s", rebal_date, e)
            value = broker.get_account_value()
            equity_values.append((rebal_date, value))
            monthly_values.append(value)

    # Build equity curve
    equity_curve = pd.Series(
        {d: v for d, v in equity_values},
        name="equity",
    )

    # Compute metrics
    metrics = _compute_metrics(
        equity_curve=equity_curve,
        initial_cash=initial_cash,
        start=start,
        end=end,
        trade_count=len(broker.get_trade_log()),
    )

    # Monthly returns
    values = pd.Series(monthly_values)
    monthly_returns = values.pct_change().dropna()

    # Trade log
    trade_log = [
        {
            "date": o.filled_at.date().isoformat() if o.filled_at else "",
            "symbol": o.symbol,
            "side": o.side.value,
            "qty": o.qty,
            "price": o.filled_price,
        }
        for o in broker.get_trade_log()
    ]

    return BacktestResult(
        metrics=metrics,
        monthly_returns=monthly_returns,
        holdings_history=holdings_history,
        trade_log=trade_log,
    )


def _compute_metrics(
    equity_curve: pd.Series,
    initial_cash: float,
    start: date,
    end: date,
    trade_count: int,
) -> BacktestMetrics:
    """Calculate performance metrics from an equity curve."""
    if equity_curve.empty:
        return BacktestMetrics(
            start_date=start, end_date=end,
            initial_value=initial_cash, final_value=initial_cash,
            cagr=0, max_drawdown=0, sharpe_ratio=0, sortino_ratio=0,
            monthly_win_rate=0, total_trades=0, annual_turnover=0,
            equity_curve=equity_curve,
        )

    final = equity_curve.iloc[-1]
    years = max((end - start).days / 365.25, 0.01)

    # CAGR
    cagr = (final / initial_cash) ** (1 / years) - 1

    # Max drawdown
    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max
    max_dd = float(drawdown.min())

    # Monthly returns
    returns = equity_curve.pct_change().dropna()

    # Sharpe (annualized, assuming monthly)
    if len(returns) > 1 and returns.std() > 0:
        sharpe = float(returns.mean() / returns.std() * np.sqrt(12))
    else:
        sharpe = 0.0

    # Sortino
    downside = returns[returns < 0]
    if len(downside) > 1 and downside.std() > 0:
        sortino = float(returns.mean() / downside.std() * np.sqrt(12))
    else:
        sortino = 0.0

    # Win rate
    win_rate = float((returns > 0).sum() / max(len(returns), 1))

    # Turnover (rough estimate)
    annual_turnover = trade_count / max(years, 0.01)

    return BacktestMetrics(
        start_date=start,
        end_date=end,
        initial_value=initial_cash,
        final_value=float(final),
        cagr=float(cagr),
        max_drawdown=float(max_dd),
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        monthly_win_rate=win_rate,
        total_trades=trade_count,
        annual_turnover=annual_turnover,
        equity_curve=equity_curve,
    )


def print_backtest_report(result: BacktestResult) -> None:
    """Print a formatted backtest summary."""
    m = result.metrics
    print("\n" + "=" * 60)
    print("SECTOR MOMENTUM STRATEGY - BACKTEST RESULTS")
    print("=" * 60)
    print(f"Period:          {m.start_date} to {m.end_date}")
    print(f"Initial Value:   ${m.initial_value:>12,.2f}")
    print(f"Final Value:     ${m.final_value:>12,.2f}")
    print(f"CAGR:            {m.cagr:>11.2%}")
    print(f"Max Drawdown:    {m.max_drawdown:>11.2%}")
    print(f"Sharpe Ratio:    {m.sharpe_ratio:>11.2f}")
    print(f"Sortino Ratio:   {m.sortino_ratio:>11.2f}")
    print(f"Monthly Win Rate:{m.monthly_win_rate:>11.2%}")
    print(f"Total Trades:    {m.total_trades:>11d}")
    print(f"Annual Turnover: {m.annual_turnover:>11.0f} trades/yr")
    print("=" * 60)
