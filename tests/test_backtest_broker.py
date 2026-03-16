"""Tests for the backtest broker."""

from datetime import date

from sector_momentum_bot.brokers.base import OrderSide

from tests.helpers import make_price_series, make_test_broker


def test_initial_state():
    broker = make_test_broker(initial_cash=50_000)
    assert broker.get_cash() == 50_000
    assert broker.get_account_value() == 50_000
    assert broker.get_positions() == {}


def test_buy_and_sell():
    broker = make_test_broker(symbols=["XLK"])
    price = broker.get_current_price("XLK")
    order = broker.submit_market_order("XLK", 10, OrderSide.BUY)
    assert order.filled_price == price
    positions = broker.get_positions()
    assert "XLK" in positions
    assert positions["XLK"].qty == 10

    broker.submit_market_order("XLK", 10, OrderSide.SELL)
    assert broker.get_positions() == {}


def test_close_position():
    broker = make_test_broker(symbols=["XLF"])
    broker.submit_market_order("XLF", 20, OrderSide.BUY)
    assert "XLF" in broker.get_positions()
    broker.close_position("XLF")
    assert broker.get_positions() == {}


def test_historical_prices():
    broker = make_test_broker(symbols=["SPY"])
    df = broker.get_historical_prices("SPY", date(2024, 6, 1), date(2024, 9, 1))
    assert not df.empty
    assert "close" in df.columns


def test_account_value_includes_positions():
    broker = make_test_broker(symbols=["XLK"], initial_cash=100_000)
    price = broker.get_current_price("XLK")
    broker.submit_market_order("XLK", 100, OrderSide.BUY)
    # Account value should be close to initial (minus commission, rounding)
    assert abs(broker.get_account_value() - 100_000) < 1
