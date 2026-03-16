"""Tests for the scheduler module."""

from datetime import date

from sector_momentum_bot.scheduler import (
    get_monthly_rebalance_dates,
    is_last_trading_day_of_month,
)


def test_last_trading_day_jan_2024():
    # Jan 31, 2024 is a Wednesday → should be last trading day
    assert is_last_trading_day_of_month(date(2024, 1, 31)) is True


def test_not_last_trading_day():
    assert is_last_trading_day_of_month(date(2024, 1, 15)) is False


def test_last_trading_day_when_month_ends_saturday():
    # Nov 30, 2024 is a Saturday → last trading day is Nov 29 (Friday)
    assert is_last_trading_day_of_month(date(2024, 11, 29)) is True
    assert is_last_trading_day_of_month(date(2024, 11, 30)) is False


def test_monthly_rebalance_dates():
    dates = get_monthly_rebalance_dates(date(2024, 1, 1), date(2024, 6, 30))
    assert len(dates) == 6
    # Each date should be in a different month
    months = [d.month for d in dates]
    assert months == [1, 2, 3, 4, 5, 6]
    # Each date should be a weekday
    for d in dates:
        assert d.weekday() < 5
