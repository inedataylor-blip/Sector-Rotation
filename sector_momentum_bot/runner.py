"""Main execution runner: ties strategy, rebalancer, and logging together."""

import logging
from datetime import date
from pathlib import Path
from typing import Optional

from sector_momentum_bot.brokers.base import BaseBroker
from sector_momentum_bot.config import StrategyConfig
from sector_momentum_bot.logger import (
    build_execution_record,
    save_execution_record,
)
from sector_momentum_bot.momentum import simple_12m_return
from sector_momentum_bot.rebalancer import RebalanceResult, execute_rebalance
from sector_momentum_bot.scheduler import is_last_trading_day_of_month
from sector_momentum_bot.strategies.sector_rotation import (
    check_absolute_momentum,
    compute_sector_scores,
    generate_target_portfolio,
    map_to_execution_etfs,
    select_top_sectors,
    compute_weights,
)

logger = logging.getLogger(__name__)


def run_strategy(
    broker: BaseBroker,
    config: StrategyConfig,
    dry_run: bool = False,
    force: bool = False,
    as_of: Optional[date] = None,
    log_dir: Optional[Path] = None,
) -> Optional[RebalanceResult]:
    """
    Execute one cycle of the sector rotation strategy.

    Args:
        broker: Broker implementation to use.
        config: Strategy configuration.
        dry_run: If True, compute trades but don't execute.
        force: If True, run even if not last trading day.
        as_of: Override date for momentum calculations.
        log_dir: Directory for execution logs.

    Returns:
        RebalanceResult if rebalance was executed, None if skipped.
    """
    today = as_of or date.today()

    # Check if we should run today
    if not force and not is_last_trading_day_of_month(today):
        logger.info("Not last trading day of month (%s). Use --force to override.", today)
        return None

    logger.info("=" * 60)
    logger.info("SECTOR ROTATION STRATEGY - %s", today)
    logger.info("Variant: %s | Momentum: %s", config.variant.value, config.momentum_method.value)
    logger.info("=" * 60)

    # Compute sector rankings
    ranked = compute_sector_scores(broker, config, as_of)

    # Absolute momentum check
    risk_on = check_absolute_momentum(broker, config, as_of)
    spy_ret = simple_12m_return(broker, config.market_benchmark, as_of=as_of)
    bil_ret = simple_12m_return(broker, config.riskfree, as_of=as_of)
    regime = "risk_on" if risk_on else "risk_off"

    # Generate target portfolio
    target_weights = generate_target_portfolio(broker, config, as_of)

    # Determine what sectors/etfs were selected (for logging)
    if risk_on:
        selected = select_top_sectors(ranked, config)
        selected_sectors = [s for s, _ in selected]
        signal_weights = compute_weights(selected, config, broker, as_of)
        execution_etfs = list(map_to_execution_etfs(signal_weights, config).keys())
    else:
        selected_sectors = []
        execution_etfs = [config.effective_bonds()]

    # Execute rebalance
    result = execute_rebalance(
        broker,
        target_weights,
        rebalance_threshold=config.rebalance_threshold,
        dry_run=dry_run,
    )

    # Build and save execution record
    record = build_execution_record(
        rebalance=result,
        sector_rankings=ranked,
        selected_sectors=selected_sectors,
        execution_etfs=execution_etfs,
        weights=target_weights,
        regime=regime,
        spy_return=spy_ret,
        bil_return=bil_ret,
    )
    save_execution_record(record, log_dir)

    if result.errors:
        logger.warning("Rebalance completed with %d errors", len(result.errors))
    else:
        logger.info("Rebalance completed successfully")

    return result
