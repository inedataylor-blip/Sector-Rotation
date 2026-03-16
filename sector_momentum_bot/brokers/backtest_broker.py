"""Backtesting broker implementation using local/downloaded price data."""

from datetime import date, datetime, timedelta
from typing import Optional

import pandas as pd

from sector_momentum_bot.brokers.base import (
    BaseBroker,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
)


class BacktestBroker(BaseBroker):
    """
    Simulated broker for backtesting.

    Takes a dict of DataFrames {symbol: df} with OHLCV data and simulates
    trading at the close price of each day.
    """

    def __init__(
        self,
        price_data: dict[str, pd.DataFrame],
        initial_cash: float = 100_000.0,
        commission_per_trade: float = 0.0,
    ):
        self._price_data = price_data
        self._cash = initial_cash
        self._initial_cash = initial_cash
        self._commission = commission_per_trade
        self._positions: dict[str, float] = {}  # symbol -> qty
        self._avg_prices: dict[str, float] = {}  # symbol -> avg_entry
        self._current_date: date = date.today()
        self._orders: list[Order] = []
        self._order_counter = 0

    def set_current_date(self, d: date) -> None:
        """Advance the simulation clock to a given date."""
        self._current_date = d

    def get_account_value(self) -> float:
        total = self._cash
        for symbol, qty in self._positions.items():
            price = self.get_current_price(symbol)
            total += qty * price
        return total

    def get_cash(self) -> float:
        return self._cash

    def get_positions(self) -> dict[str, Position]:
        result = {}
        for symbol, qty in self._positions.items():
            if qty == 0:
                continue
            price = self.get_current_price(symbol)
            avg = self._avg_prices.get(symbol, price)
            result[symbol] = Position(
                symbol=symbol,
                qty=qty,
                market_value=qty * price,
                avg_entry_price=avg,
                current_price=price,
                unrealized_pnl=qty * (price - avg),
            )
        return result

    def get_historical_prices(
        self,
        symbol: str,
        start: date,
        end: date,
    ) -> pd.DataFrame:
        if symbol not in self._price_data:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        df = self._price_data[symbol].copy()
        # Ensure index is date
        if not isinstance(df.index[0], date):
            df.index = pd.to_datetime(df.index).date
        mask = (df.index >= start) & (df.index <= end)
        return df.loc[mask, ["open", "high", "low", "close", "volume"]]

    def get_current_price(self, symbol: str) -> float:
        if symbol not in self._price_data:
            raise ValueError(f"No price data for {symbol}")
        df = self._price_data[symbol]
        if not isinstance(df.index[0], date):
            df.index = pd.to_datetime(df.index).date
        # Find the most recent price on or before current_date
        valid = df.loc[df.index <= self._current_date]
        if valid.empty:
            raise ValueError(f"No price data for {symbol} on or before {self._current_date}")
        return float(valid["close"].iloc[-1])

    def submit_market_order(
        self,
        symbol: str,
        qty: float,
        side: OrderSide,
    ) -> Order:
        price = self.get_current_price(symbol)
        self._order_counter += 1

        if side == OrderSide.BUY:
            cost = qty * price + self._commission
            self._cash -= cost
            prev_qty = self._positions.get(symbol, 0)
            prev_avg = self._avg_prices.get(symbol, 0)
            new_qty = prev_qty + qty
            if new_qty > 0:
                self._avg_prices[symbol] = (
                    (prev_avg * prev_qty + price * qty) / new_qty
                )
            self._positions[symbol] = new_qty
        else:
            proceeds = qty * price - self._commission
            self._cash += proceeds
            self._positions[symbol] = self._positions.get(symbol, 0) - qty
            if self._positions[symbol] <= 0:
                self._positions.pop(symbol, None)
                self._avg_prices.pop(symbol, None)

        order = Order(
            id=str(self._order_counter),
            symbol=symbol,
            side=side,
            qty=qty,
            order_type=OrderType.MARKET,
            status=OrderStatus.FILLED,
            filled_price=price,
            filled_at=datetime.combine(self._current_date, datetime.min.time()),
        )
        self._orders.append(order)
        return order

    def close_position(self, symbol: str) -> Optional[Order]:
        qty = self._positions.get(symbol, 0)
        if qty == 0:
            return None
        side = OrderSide.SELL if qty > 0 else OrderSide.BUY
        return self.submit_market_order(symbol, abs(qty), side)

    def is_market_open(self) -> bool:
        return True

    def get_next_market_close(self) -> datetime:
        return datetime.combine(self._current_date, datetime.min.time()).replace(hour=16)

    def get_trade_log(self) -> list[Order]:
        """Return all orders executed during the backtest."""
        return list(self._orders)
