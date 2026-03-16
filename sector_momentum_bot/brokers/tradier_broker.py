"""Tradier broker implementation."""

import os
from datetime import date, datetime
from typing import Optional

import pandas as pd
import requests

from sector_momentum_bot.brokers.base import (
    BaseBroker,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
)

_SANDBOX_URL = "https://sandbox.tradier.com/v1"
_PROD_URL = "https://api.tradier.com/v1"


class TradierBroker(BaseBroker):
    """
    Tradier broker implementation.

    Uses the Tradier REST API directly. Requires TRADIER_ACCESS_TOKEN
    environment variable. Set TRADIER_SANDBOX=true for sandbox mode.
    """

    def __init__(
        self,
        access_token: Optional[str] = None,
        account_id: Optional[str] = None,
        sandbox: bool = True,
    ):
        self._token = access_token or os.environ["TRADIER_ACCESS_TOKEN"]
        self._account_id = account_id or os.environ.get("TRADIER_ACCOUNT_ID", "")
        self._sandbox = sandbox or os.environ.get("TRADIER_SANDBOX", "true").lower() == "true"
        self._base_url = _SANDBOX_URL if self._sandbox else _PROD_URL
        self._headers = {
            "Authorization": f"Bearer {self._token}",
            "Accept": "application/json",
        }
        if not self._account_id:
            self._account_id = self._fetch_account_id()

    def _get(self, path: str, params: Optional[dict] = None) -> dict:
        resp = requests.get(
            f"{self._base_url}{path}",
            headers=self._headers,
            params=params,
        )
        resp.raise_for_status()
        return resp.json()

    def _post(self, path: str, data: Optional[dict] = None) -> dict:
        resp = requests.post(
            f"{self._base_url}{path}",
            headers=self._headers,
            data=data,
        )
        resp.raise_for_status()
        return resp.json()

    def _fetch_account_id(self) -> str:
        data = self._get("/user/profile")
        account = data["profile"]["account"]
        if isinstance(account, list):
            return account[0]["account_number"]
        return account["account_number"]

    def get_account_value(self) -> float:
        data = self._get(f"/accounts/{self._account_id}/balances")
        return float(data["balances"]["total_equity"])

    def get_cash(self) -> float:
        data = self._get(f"/accounts/{self._account_id}/balances")
        balances = data["balances"]
        return float(balances.get("total_cash", balances.get("cash", {}).get("cash_available", 0)))

    def get_positions(self) -> dict[str, Position]:
        data = self._get(f"/accounts/{self._account_id}/positions")
        positions_data = data.get("positions", {})
        if positions_data == "null" or not positions_data:
            return {}
        raw = positions_data.get("position", [])
        if isinstance(raw, dict):
            raw = [raw]
        result = {}
        for p in raw:
            symbol = p["symbol"]
            qty = float(p["quantity"])
            cost_basis = float(p["cost_basis"])
            avg_price = cost_basis / qty if qty != 0 else 0
            current_price = float(p.get("last_price", 0))
            market_value = qty * current_price
            result[symbol] = Position(
                symbol=symbol,
                qty=qty,
                market_value=market_value,
                avg_entry_price=avg_price,
                current_price=current_price,
                unrealized_pnl=market_value - cost_basis,
            )
        return result

    def get_historical_prices(
        self,
        symbol: str,
        start: date,
        end: date,
    ) -> pd.DataFrame:
        data = self._get(
            "/markets/history",
            params={
                "symbol": symbol,
                "interval": "daily",
                "start": start.isoformat(),
                "end": end.isoformat(),
            },
        )
        history = data.get("history", {})
        if not history:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        days = history.get("day", [])
        if isinstance(days, dict):
            days = [days]
        df = pd.DataFrame(days)
        df["date"] = pd.to_datetime(df["date"]).dt.date
        df = df.set_index("date")
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col])
        return df[["open", "high", "low", "close", "volume"]]

    def get_current_price(self, symbol: str) -> float:
        data = self._get("/markets/quotes", params={"symbols": symbol})
        quote = data["quotes"]["quote"]
        if isinstance(quote, list):
            quote = quote[0]
        return float(quote["last"])

    def submit_market_order(
        self,
        symbol: str,
        qty: float,
        side: OrderSide,
    ) -> Order:
        data = self._post(
            f"/accounts/{self._account_id}/orders",
            data={
                "class": "equity",
                "symbol": symbol,
                "side": side.value,
                "quantity": str(int(qty)),
                "type": "market",
                "duration": "day",
            },
        )
        order_data = data.get("order", {})
        return Order(
            id=str(order_data.get("id", "")),
            symbol=symbol,
            side=side,
            qty=qty,
            order_type=OrderType.MARKET,
            status=OrderStatus.PENDING,
        )

    def close_position(self, symbol: str) -> Optional[Order]:
        positions = self.get_positions()
        if symbol not in positions:
            return None
        pos = positions[symbol]
        side = OrderSide.SELL if pos.qty > 0 else OrderSide.BUY
        return self.submit_market_order(symbol, abs(pos.qty), side)

    def is_market_open(self) -> bool:
        data = self._get("/markets/clock")
        return data["clock"]["state"] == "open"

    def get_next_market_close(self) -> datetime:
        data = self._get("/markets/clock")
        desc = data["clock"].get("description", "")
        # Tradier doesn't give exact close time in a standard way;
        # assume 4pm ET on the current/next trading day.
        from datetime import timezone, timedelta
        et = timezone(timedelta(hours=-4))
        now = datetime.now(et)
        close_today = now.replace(hour=16, minute=0, second=0, microsecond=0)
        if now < close_today:
            return close_today
        return close_today + timedelta(days=1)
