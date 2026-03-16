"""Scheduling utilities: detect last trading day of month."""

from datetime import date, timedelta
from typing import Optional

import pandas as pd


def is_last_trading_day_of_month(
    today: Optional[date] = None,
) -> bool:
    """
    Check if today is the last trading day of the month.

    Uses pandas market calendar logic: a trading day is Mon-Fri, excluding
    US market holidays. The last trading day is the final such day in the month.
    """
    today = today or date.today()
    last_day = _last_trading_day_of_month(today.year, today.month)
    return today == last_day


def _last_trading_day_of_month(year: int, month: int) -> date:
    """Find the last business day of a given month."""
    # Find last calendar day of month
    if month == 12:
        last_cal = date(year + 1, 1, 1) - timedelta(days=1)
    else:
        last_cal = date(year, month + 1, 1) - timedelta(days=1)

    # Walk backwards to find a weekday (Mon-Fri)
    d = last_cal
    while d.weekday() >= 5:  # Saturday=5, Sunday=6
        d -= timedelta(days=1)
    return d


def get_monthly_rebalance_dates(
    start: date,
    end: date,
) -> list[date]:
    """
    Generate a list of last-trading-day-of-month dates between start and end.

    Useful for backtesting.
    """
    dates = []
    current = date(start.year, start.month, 1)
    while current <= end:
        ltd = _last_trading_day_of_month(current.year, current.month)
        if start <= ltd <= end:
            dates.append(ltd)
        # Move to next month
        if current.month == 12:
            current = date(current.year + 1, 1, 1)
        else:
            current = date(current.year, current.month + 1, 1)
    return dates
