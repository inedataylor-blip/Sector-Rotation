"""Tests for momentum calculations."""

from sector_momentum_bot.config import MomentumMethod
from sector_momentum_bot.momentum import (
    calculate_momentum,
    calculate_sector_volatility,
    composite_momentum,
    risk_adjusted_momentum,
    simple_12m_return,
    sma_10m_momentum,
)

from tests.helpers import make_test_broker


def test_simple_12m_return_positive_drift():
    broker = make_test_broker(
        symbols=["XLK"],
        daily_returns={"XLK": 0.001},
    )
    ret = simple_12m_return(broker, "XLK", as_of=broker.as_of_date)
    assert ret > 0


def test_simple_12m_return_negative_drift():
    broker = make_test_broker(
        symbols=["XLE"],
        daily_returns={"XLE": -0.001},
    )
    ret = simple_12m_return(broker, "XLE", as_of=broker.as_of_date)
    assert ret < 0


def test_composite_momentum():
    broker = make_test_broker(symbols=["XLF"], daily_returns={"XLF": 0.001})
    score = composite_momentum(broker, "XLF", as_of=broker.as_of_date)
    assert score > 0


def test_sma_10m_momentum():
    broker = make_test_broker(symbols=["XLV"], daily_returns={"XLV": 0.001})
    score = sma_10m_momentum(broker, "XLV", as_of=broker.as_of_date)
    assert score > 0


def test_risk_adjusted_momentum():
    broker = make_test_broker(symbols=["XLI"], daily_returns={"XLI": 0.001})
    score = risk_adjusted_momentum(broker, "XLI", as_of=broker.as_of_date)
    assert score > 0


def test_calculate_momentum_dispatch():
    broker = make_test_broker(symbols=["XLK"], daily_returns={"XLK": 0.001})
    for method in MomentumMethod:
        score = calculate_momentum(broker, "XLK", method=method, as_of=broker.as_of_date)
        assert isinstance(score, float)


def test_sector_volatility():
    broker = make_test_broker(symbols=["XLK"])
    vol = calculate_sector_volatility(broker, "XLK", as_of=broker.as_of_date)
    assert vol > 0
    assert vol < 5.0  # reasonable annualized vol range
