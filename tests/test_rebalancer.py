"""Tests for the rebalancer."""

from sector_momentum_bot.brokers.base import OrderSide
from sector_momentum_bot.rebalancer import (
    compute_target_shares,
    compute_trades,
    execute_rebalance,
)

from tests.helpers import make_test_broker


def test_compute_target_shares():
    broker = make_test_broker(symbols=["XLK", "XLF"])
    targets = compute_target_shares({"XLK": 0.5, "XLF": 0.5}, 100_000, broker)
    assert "XLK" in targets
    assert "XLF" in targets
    assert all(isinstance(v, int) for v in targets.values())


def test_compute_trades_from_empty():
    current = {}
    target = {"XLK": 100, "XLF": 50}
    trades = compute_trades(current, target)
    assert len(trades) == 2
    assert all(t.side == OrderSide.BUY for t in trades)


def test_compute_trades_full_rotation():
    current = {"XLK": 100, "XLF": 50}
    target = {"XLE": 80, "XLV": 40}
    trades = compute_trades(current, target)
    sells = [t for t in trades if t.side == OrderSide.SELL]
    buys = [t for t in trades if t.side == OrderSide.BUY]
    assert len(sells) == 2
    assert len(buys) == 2
    # Sells should come first
    sell_indices = [i for i, t in enumerate(trades) if t.side == OrderSide.SELL]
    buy_indices = [i for i, t in enumerate(trades) if t.side == OrderSide.BUY]
    assert max(sell_indices) < min(buy_indices)


def test_compute_trades_no_change():
    current = {"XLK": 100}
    target = {"XLK": 100}
    trades = compute_trades(current, target)
    assert len(trades) == 0


def test_execute_rebalance_dry_run():
    broker = make_test_broker()
    result = execute_rebalance(
        broker, {"XLK": 0.5, "XLF": 0.5}, dry_run=True
    )
    assert len(result.trades) > 0
    # Dry run: no actual orders
    assert all(t.order is None for t in result.trades)


def test_execute_rebalance_live():
    broker = make_test_broker()
    result = execute_rebalance(broker, {"XLK": 0.5, "XLF": 0.5})
    assert len(result.trades) > 0
    positions = broker.get_positions()
    assert "XLK" in positions or "XLF" in positions
