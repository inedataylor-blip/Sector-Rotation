"""Portfolio rebalancing and trade execution logic."""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from sector_momentum_bot.brokers.base import BaseBroker, Order, OrderSide

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """A planned or executed trade."""
    symbol: str
    side: OrderSide
    qty: int
    reason: str
    order: Optional[Order] = None


@dataclass
class RebalanceResult:
    """Summary of a rebalance operation."""
    timestamp: datetime
    account_value_before: float
    account_value_after: float
    target_weights: dict[str, float]
    previous_holdings: list[str]
    new_holdings: list[str]
    trades: list[Trade] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def compute_target_shares(
    target_weights: dict[str, float],
    account_value: float,
    broker: BaseBroker,
) -> dict[str, int]:
    """
    Convert target weights to target share counts.

    Uses current prices to determine how many whole shares to buy.
    """
    targets: dict[str, int] = {}
    for symbol, weight in target_weights.items():
        target_value = account_value * weight
        try:
            price = broker.get_current_price(symbol)
            if price > 0:
                targets[symbol] = int(target_value / price)
        except Exception as e:
            logger.error("Could not get price for %s: %s", symbol, e)
    return targets


def compute_trades(
    current_positions: dict[str, float],
    target_shares: dict[str, int],
    rebalance_threshold: float = 0.05,
    account_value: float = 0.0,
    broker: Optional[BaseBroker] = None,
) -> list[Trade]:
    """
    Determine what trades are needed to move from current to target portfolio.

    Sells are generated first, then buys (to free up cash).
    """
    trades: list[Trade] = []
    all_symbols = set(current_positions.keys()) | set(target_shares.keys())

    sells: list[Trade] = []
    buys: list[Trade] = []

    for symbol in all_symbols:
        current_qty = current_positions.get(symbol, 0)
        target_qty = target_shares.get(symbol, 0)
        diff = target_qty - current_qty

        if diff == 0:
            continue

        # Check if the change exceeds the rebalance threshold
        if account_value > 0 and broker is not None and abs(diff) > 0:
            try:
                price = broker.get_current_price(symbol)
                drift = abs(diff * price) / account_value
                if drift < rebalance_threshold and current_qty > 0 and target_qty > 0:
                    logger.debug(
                        "Skipping %s: drift %.2%% below threshold", symbol, drift
                    )
                    continue
            except Exception:
                pass

        if diff < 0:
            sells.append(Trade(
                symbol=symbol,
                side=OrderSide.SELL,
                qty=abs(int(diff)),
                reason="close" if target_qty == 0 else "reduce",
            ))
        else:
            buys.append(Trade(
                symbol=symbol,
                side=OrderSide.BUY,
                qty=int(diff),
                reason="open" if current_qty == 0 else "increase",
            ))

    # Sells first, then buys
    return sells + buys


def execute_rebalance(
    broker: BaseBroker,
    target_weights: dict[str, float],
    rebalance_threshold: float = 0.05,
    dry_run: bool = False,
) -> RebalanceResult:
    """
    Execute a full portfolio rebalance to match target weights.

    1. Compute target share counts from weights
    2. Diff against current holdings
    3. Execute sells first, then buys
    4. Return a summary of all actions taken
    """
    account_value = broker.get_account_value()
    positions = broker.get_positions()
    current_qty = {sym: pos.qty for sym, pos in positions.items()}
    previous_holdings = list(current_qty.keys())

    logger.info(
        "Rebalancing: account=$%.2f, %d current positions, %d targets",
        account_value,
        len(current_qty),
        len(target_weights),
    )

    # Compute target shares
    target_shares = compute_target_shares(target_weights, account_value, broker)

    # Determine trades
    trades = compute_trades(
        current_qty,
        target_shares,
        rebalance_threshold,
        account_value,
        broker,
    )

    result = RebalanceResult(
        timestamp=datetime.utcnow(),
        account_value_before=account_value,
        account_value_after=account_value,
        target_weights=target_weights,
        previous_holdings=previous_holdings,
        new_holdings=list(target_shares.keys()),
        trades=[],
    )

    if dry_run:
        logger.info("DRY RUN - %d trades planned:", len(trades))
        for t in trades:
            logger.info("  %s %d %s (%s)", t.side.value.upper(), t.qty, t.symbol, t.reason)
            result.trades.append(t)
        return result

    # Execute trades
    for trade in trades:
        try:
            if trade.qty <= 0:
                continue
            logger.info(
                "Executing: %s %d %s (%s)",
                trade.side.value.upper(), trade.qty, trade.symbol, trade.reason,
            )
            order = broker.submit_market_order(trade.symbol, trade.qty, trade.side)
            trade.order = order
            result.trades.append(trade)
        except Exception as e:
            err = f"Failed to {trade.side.value} {trade.qty} {trade.symbol}: {e}"
            logger.error(err)
            result.errors.append(err)

    result.account_value_after = broker.get_account_value()
    logger.info(
        "Rebalance complete: %d trades, value $%.2f → $%.2f",
        len(result.trades),
        result.account_value_before,
        result.account_value_after,
    )
    return result
