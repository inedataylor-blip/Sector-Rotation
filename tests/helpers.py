"""Shared test helpers."""

from datetime import date, timedelta

import numpy as np
import pandas as pd

from sector_momentum_bot.brokers.backtest_broker import BacktestBroker


def make_price_series(
    start_price: float = 100.0,
    daily_return: float = 0.0005,
    days: int = 600,
    start_date: date = date(2023, 1, 2),
    volatility: float = 0.01,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic OHLCV data."""
    rng = np.random.RandomState(seed)
    dates = []
    d = start_date
    for _ in range(days):
        if d.weekday() < 5:  # skip weekends
            dates.append(d)
        d += timedelta(days=1)
    dates = dates[:days]

    prices = [start_price]
    for _ in range(len(dates) - 1):
        ret = daily_return + rng.normal(0, volatility)
        prices.append(prices[-1] * (1 + ret))

    prices = np.array(prices[:len(dates)])
    df = pd.DataFrame({
        "open": prices * (1 + rng.normal(0, 0.002, len(dates))),
        "high": prices * (1 + abs(rng.normal(0.005, 0.003, len(dates)))),
        "low": prices * (1 - abs(rng.normal(0.005, 0.003, len(dates)))),
        "close": prices,
        "volume": rng.randint(100000, 1000000, len(dates)),
    }, index=dates)
    return df


def make_test_broker(
    symbols: list[str] | None = None,
    initial_cash: float = 100_000.0,
    daily_returns: dict[str, float] | None = None,
    days: int = 600,
) -> BacktestBroker:
    """Create a BacktestBroker with synthetic data for all needed symbols."""
    if symbols is None:
        symbols = [
            "XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB",
            "XLU", "XLRE", "XLC", "SPY", "BIL", "BND",
            "TECL", "FAS", "ERX", "CURE", "DRN", "TMF",
        ]

    daily_returns = daily_returns or {}
    price_data = {}
    for i, sym in enumerate(symbols):
        ret = daily_returns.get(sym, 0.0003 + i * 0.00005)
        price_data[sym] = make_price_series(
            daily_return=ret,
            days=days,
            seed=42 + i,
        )

    broker = BacktestBroker(price_data, initial_cash=initial_cash)
    # Set current date to last available date
    last_date = max(df.index[-1] for df in price_data.values())
    broker.set_current_date(last_date)
    broker.as_of_date = last_date  # convenience for tests
    return broker
