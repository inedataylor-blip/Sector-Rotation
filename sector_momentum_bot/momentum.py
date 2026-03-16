"""Momentum calculation methods."""

from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from sector_momentum_bot.brokers.base import BaseBroker
from sector_momentum_bot.config import MomentumMethod


def _get_close_series(
    broker: BaseBroker,
    symbol: str,
    lookback_days: int,
    as_of: Optional[date] = None,
) -> pd.Series:
    """Fetch a close-price series for the given lookback window."""
    end = as_of or date.today()
    # Fetch extra days to account for weekends/holidays
    start = end - timedelta(days=int(lookback_days * 1.6) + 30)
    df = broker.get_historical_prices(symbol, start, end)
    if df.empty:
        return pd.Series(dtype=float)
    return df["close"]


def simple_12m_return(
    broker: BaseBroker,
    symbol: str,
    lookback_days: int = 252,
    as_of: Optional[date] = None,
) -> float:
    """Simple trailing return over lookback_days."""
    closes = _get_close_series(broker, symbol, lookback_days, as_of)
    if len(closes) < 2:
        return 0.0
    # Use the actual lookback or all available data
    n = min(lookback_days, len(closes) - 1)
    return float(closes.iloc[-1] / closes.iloc[-1 - n] - 1)


def composite_momentum(
    broker: BaseBroker,
    symbol: str,
    lookback_days: int = 252,
    as_of: Optional[date] = None,
) -> float:
    """
    Composite multi-period momentum.
    Score = 0.33 * 12m_return + 0.33 * 6m_return + 0.34 * 3m_return
    """
    closes = _get_close_series(broker, symbol, lookback_days, as_of)
    if len(closes) < 63:  # need at least ~3 months
        return 0.0

    current = closes.iloc[-1]

    def _ret(n_days: int) -> float:
        idx = min(n_days, len(closes) - 1)
        return float(current / closes.iloc[-1 - idx] - 1)

    ret_12m = _ret(min(252, len(closes) - 1))
    ret_6m = _ret(min(126, len(closes) - 1))
    ret_3m = _ret(min(63, len(closes) - 1))

    return 0.33 * ret_12m + 0.33 * ret_6m + 0.34 * ret_3m


def sma_10m_momentum(
    broker: BaseBroker,
    symbol: str,
    lookback_days: int = 252,
    as_of: Optional[date] = None,
) -> float:
    """
    Faber's 10-month SMA method.
    Returns price/sma - 1 if above SMA, else a large negative value.
    """
    closes = _get_close_series(broker, symbol, lookback_days, as_of)
    if len(closes) < 200:
        return -999.0
    sma_200 = closes.iloc[-200:].mean()
    current = closes.iloc[-1]
    if current > sma_200:
        return float(current / sma_200 - 1)
    return float(current / sma_200 - 1)  # negative when below SMA


def risk_adjusted_momentum(
    broker: BaseBroker,
    symbol: str,
    lookback_days: int = 252,
    as_of: Optional[date] = None,
) -> float:
    """
    Risk-adjusted (Sharpe-like) momentum.
    annualized_return / annualized_vol
    """
    closes = _get_close_series(broker, symbol, lookback_days, as_of)
    n = min(lookback_days, len(closes))
    if n < 20:
        return 0.0
    returns = closes.pct_change().dropna().iloc[-n:]
    if len(returns) == 0 or returns.std() == 0:
        return 0.0
    return float(returns.mean() / returns.std() * np.sqrt(252))


# Dispatch table
_METHODS = {
    MomentumMethod.SIMPLE_12M: simple_12m_return,
    MomentumMethod.COMPOSITE: composite_momentum,
    MomentumMethod.SMA_10M: sma_10m_momentum,
    MomentumMethod.RISK_ADJUSTED: risk_adjusted_momentum,
}


def calculate_momentum(
    broker: BaseBroker,
    symbol: str,
    method: MomentumMethod = MomentumMethod.COMPOSITE,
    lookback_days: int = 252,
    as_of: Optional[date] = None,
) -> float:
    """Calculate momentum for a symbol using the specified method."""
    fn = _METHODS[method]
    return fn(broker, symbol, lookback_days, as_of)


def calculate_sector_volatility(
    broker: BaseBroker,
    symbol: str,
    lookback_days: int = 60,
    as_of: Optional[date] = None,
) -> float:
    """Calculate annualized volatility for a symbol."""
    closes = _get_close_series(broker, symbol, lookback_days, as_of)
    if len(closes) < 10:
        return 999.0  # very high vol as fallback
    returns = closes.pct_change().dropna()
    n = min(lookback_days, len(returns))
    return float(returns.iloc[-n:].std() * np.sqrt(252))
