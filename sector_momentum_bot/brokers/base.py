"""Abstract base broker interface for broker-agnostic strategy execution."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
from typing import Optional

import pandas as pd


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"


class OrderStatus(Enum):
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class Position:
    """A current holding in the portfolio."""
    symbol: str
    qty: float
    market_value: float
    avg_entry_price: float
    current_price: float
    unrealized_pnl: float


@dataclass
class Order:
    """An executed or pending order."""
    id: str
    symbol: str
    side: OrderSide
    qty: float
    order_type: OrderType
    status: OrderStatus
    filled_price: Optional[float] = None
    filled_at: Optional[datetime] = None


class BaseBroker(ABC):
    """
    Abstract broker interface.

    All broker implementations must implement these methods. The strategy
    engine interacts exclusively through this interface, making it
    trivial to swap brokers.
    """

    @abstractmethod
    def get_account_value(self) -> float:
        """Return total portfolio value (cash + positions)."""

    @abstractmethod
    def get_cash(self) -> float:
        """Return available cash balance."""

    @abstractmethod
    def get_positions(self) -> dict[str, Position]:
        """Return current positions keyed by symbol."""

    @abstractmethod
    def get_historical_prices(
        self,
        symbol: str,
        start: date,
        end: date,
    ) -> pd.DataFrame:
        """
        Return daily OHLCV data for a symbol.

        Returns a DataFrame with columns: open, high, low, close, volume
        indexed by date. Prices should be adjusted for splits/dividends.
        """

    @abstractmethod
    def get_current_price(self, symbol: str) -> float:
        """Return the latest price for a symbol."""

    @abstractmethod
    def submit_market_order(
        self,
        symbol: str,
        qty: float,
        side: OrderSide,
    ) -> Order:
        """Submit a market order. Returns the Order object."""

    @abstractmethod
    def close_position(self, symbol: str) -> Optional[Order]:
        """Close an entire position in a symbol. Returns None if no position."""

    @abstractmethod
    def is_market_open(self) -> bool:
        """Check if the market is currently open for trading."""

    @abstractmethod
    def get_next_market_close(self) -> datetime:
        """Return the datetime of the next market close."""

    def is_tradeable(self, symbol: str) -> bool:
        """Check if a symbol is tradeable. Default returns True."""
        return True

    def get_position_qty(self, symbol: str) -> float:
        """Convenience: return quantity held for a symbol, or 0."""
        positions = self.get_positions()
        if symbol in positions:
            return positions[symbol].qty
        return 0.0
