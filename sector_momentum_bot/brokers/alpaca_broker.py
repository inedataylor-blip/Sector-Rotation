"""Alpaca broker implementation."""

import os
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


class AlpacaBroker(BaseBroker):
    """
    Alpaca Markets broker implementation.

    Uses the alpaca-py SDK. Requires ALPACA_API_KEY and ALPACA_SECRET_KEY
    environment variables. Set ALPACA_PAPER=true for paper trading.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        paper: bool = True,
    ):
        try:
            from alpaca.trading.client import TradingClient
            from alpaca.data.historical import StockHistoricalDataClient
        except ImportError:
            raise ImportError(
                "alpaca-py is required for AlpacaBroker. "
                "Install with: pip install alpaca-py"
            )

        self._api_key = api_key or os.environ["ALPACA_API_KEY"]
        self._secret_key = secret_key or os.environ["ALPACA_SECRET_KEY"]
        self._paper = paper or os.environ.get("ALPACA_PAPER", "true").lower() == "true"

        self._trading = TradingClient(
            self._api_key, self._secret_key, paper=self._paper
        )
        self._data = StockHistoricalDataClient(self._api_key, self._secret_key)

    def get_account_value(self) -> float:
        account = self._trading.get_account()
        return float(account.equity)

    def get_cash(self) -> float:
        account = self._trading.get_account()
        return float(account.cash)

    def get_positions(self) -> dict[str, Position]:
        from alpaca.trading.requests import GetAssetsRequest

        raw = self._trading.get_all_positions()
        result = {}
        for p in raw:
            result[p.symbol] = Position(
                symbol=p.symbol,
                qty=float(p.qty),
                market_value=float(p.market_value),
                avg_entry_price=float(p.avg_entry_price),
                current_price=float(p.current_price),
                unrealized_pnl=float(p.unrealized_pl),
            )
        return result

    def get_historical_prices(
        self,
        symbol: str,
        start: date,
        end: date,
    ) -> pd.DataFrame:
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame

        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Day,
            start=datetime.combine(start, datetime.min.time()),
            end=datetime.combine(end, datetime.min.time()),
        )
        bars = self._data.get_stock_bars(request)
        df = bars.df
        if isinstance(df.index, pd.MultiIndex):
            df = df.droplevel(0)
        df.index = pd.to_datetime(df.index).date
        df = df.rename(columns={
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "volume": "volume",
        })
        return df[["open", "high", "low", "close", "volume"]]

    def get_current_price(self, symbol: str) -> float:
        from alpaca.data.requests import StockLatestQuoteRequest

        request = StockLatestQuoteRequest(symbol_or_symbols=symbol)
        quote = self._data.get_stock_latest_quote(request)
        if isinstance(quote, dict):
            q = quote[symbol]
        else:
            q = quote
        return float(q.ask_price + q.bid_price) / 2

    def submit_market_order(
        self,
        symbol: str,
        qty: float,
        side: OrderSide,
    ) -> Order:
        from alpaca.trading.requests import MarketOrderRequest
        from alpaca.trading.enums import OrderSide as AlpacaSide, TimeInForce

        alpaca_side = AlpacaSide.BUY if side == OrderSide.BUY else AlpacaSide.SELL

        request = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=alpaca_side,
            time_in_force=TimeInForce.DAY,
        )
        order = self._trading.submit_order(request)
        return Order(
            id=str(order.id),
            symbol=order.symbol,
            side=side,
            qty=float(order.qty),
            order_type=OrderType.MARKET,
            status=OrderStatus.PENDING,
        )

    def close_position(self, symbol: str) -> Optional[Order]:
        positions = self.get_positions()
        if symbol not in positions:
            return None
        order = self._trading.close_position(symbol)
        return Order(
            id=str(order.id),
            symbol=symbol,
            side=OrderSide.SELL,
            qty=positions[symbol].qty,
            order_type=OrderType.MARKET,
            status=OrderStatus.PENDING,
        )

    def is_market_open(self) -> bool:
        clock = self._trading.get_clock()
        return clock.is_open

    def get_next_market_close(self) -> datetime:
        clock = self._trading.get_clock()
        return clock.next_close

    def is_tradeable(self, symbol: str) -> bool:
        try:
            asset = self._trading.get_asset(symbol)
            return asset.tradable
        except Exception:
            return False
