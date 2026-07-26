"""
Core Sector Rotation Strategy Engine.

Implements all four variants (A-D) of the Leveraged Sector Momentum Strategy.
"""

import logging
from datetime import date
from typing import Optional

from sector_momentum_bot.brokers.base import BaseBroker
from sector_momentum_bot.config import (
    SECTOR_MAP,
    SUBSECTOR_MAP,
    StrategyConfig,
    Variant,
    WeightingScheme,
    signal_to_sector,
)
from sector_momentum_bot.momentum import (
    _get_close_series,
    calculate_momentum,
    calculate_sector_volatility,
    simple_12m_return,
)

logger = logging.getLogger(__name__)


def check_fast_riskoff(
    broker: BaseBroker,
    config: StrategyConfig,
    as_of: Optional[date] = None,
    days_below: int = 0,
) -> tuple[bool, int]:
    """
    Fast daily risk-off check: is SPY below its N-day SMA?

    Returns (is_riskoff, consecutive_days_below).
    The caller tracks days_below across iterations and passes it in.
    Risk-off triggers when days_below >= config.fast_riskoff_confirmation_days.
    """
    if not config.fast_riskoff_enabled:
        return False, 0

    sma_days = config.fast_riskoff_sma_days
    closes = _get_close_series(broker, config.market_benchmark, sma_days, as_of)

    if len(closes) < sma_days:
        return False, 0

    current_price = closes.iloc[-1]
    sma = closes.iloc[-sma_days:].mean()

    if current_price < sma:
        days_below += 1
    else:
        days_below = 0

    is_riskoff = days_below >= config.fast_riskoff_confirmation_days
    if is_riskoff:
        logger.info(
            "FAST RISK-OFF: %s=%.2f < SMA(%d)=%.2f (%d days)",
            config.market_benchmark, current_price, sma_days, sma, days_below,
        )
    return is_riskoff, days_below


def compute_sector_scores(
    broker: BaseBroker,
    config: StrategyConfig,
    as_of: Optional[date] = None,
) -> list[tuple[str, float]]:
    """
    Calculate momentum scores for all sectors in the universe.

    Returns list of (signal_etf, momentum_score) sorted descending.
    """
    universe = list(config.sectors)
    if config.include_subsectors:
        universe += [s.signal_etf for s in SUBSECTOR_MAP.values()]

    scores: list[tuple[str, float]] = []
    for etf in universe:
        try:
            score = calculate_momentum(
                broker,
                etf,
                method=config.momentum_method,
                lookback_days=config.momentum_lookback_days,
                as_of=as_of,
            )
            scores.append((etf, score))
            logger.debug("Momentum %s: %.4f", etf, score)
        except Exception as e:
            logger.warning("Failed to calculate momentum for %s: %s", etf, e)

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores


def check_absolute_momentum(
    broker: BaseBroker,
    config: StrategyConfig,
    as_of: Optional[date] = None,
) -> bool:
    """
    Absolute momentum test: is the broad market outperforming risk-free?

    Returns True if risk-on (SPY > BIL), False if risk-off.
    """
    spy_ret = simple_12m_return(
        broker, config.market_benchmark,
        lookback_days=config.momentum_lookback_days,
        as_of=as_of,
    )
    bil_ret = simple_12m_return(
        broker, config.riskfree,
        lookback_days=config.momentum_lookback_days,
        as_of=as_of,
    )
    logger.info(
        "Absolute momentum: %s=%.4f vs %s=%.4f → %s",
        config.market_benchmark, spy_ret,
        config.riskfree, bil_ret,
        "RISK-ON" if spy_ret > bil_ret else "RISK-OFF",
    )
    return spy_ret > bil_ret


def select_top_sectors(
    ranked_sectors: list[tuple[str, float]],
    config: StrategyConfig,
) -> list[tuple[str, float]]:
    """
    Select the top N sectors based on variant and config.

    Filters for positive momentum if required, and respects concentration limits.
    """
    candidates = list(ranked_sectors)

    # Filter positive momentum only
    if config.require_positive_momentum:
        candidates = [(s, m) for s, m in candidates if m > 0]

    n = config.effective_num_sectors()

    # Variant D: prefer sectors with leveraged ETFs
    if config.variant == Variant.TOP_1_LEVERAGED:
        leveraged = [(s, m) for s, m in candidates if _has_leverage(s)]
        if leveraged:
            return leveraged[:1]
        return candidates[:1] if candidates else []

    # Clamp between min and max sectors
    n = max(config.min_sectors, min(n, config.max_sectors, len(candidates)))
    return candidates[:n]


def _has_leverage(signal_etf: str) -> bool:
    """Check if a signal ETF has a leveraged execution counterpart."""
    sec = signal_to_sector(signal_etf)
    if sec is None:
        return False
    return sec.leverage_factor > 1


def compute_weights(
    selected: list[tuple[str, float]],
    config: StrategyConfig,
    broker: Optional[BaseBroker] = None,
    as_of: Optional[date] = None,
) -> dict[str, float]:
    """
    Compute allocation weights for selected sectors.

    Returns dict of {signal_etf: weight} summing to ~1.0.
    """
    if not selected:
        return {}

    if config.weighting == WeightingScheme.EQUAL:
        w = 1.0 / len(selected)
        return {s: w for s, _ in selected}

    elif config.weighting == WeightingScheme.MOMENTUM:
        total = sum(max(0, m) for _, m in selected)
        if total == 0:
            w = 1.0 / len(selected)
            return {s: w for s, _ in selected}
        weights = {s: max(0, m) / total for s, m in selected}
        # Apply concentration cap
        if config.max_single_sector < 1.0:
            weights = _cap_weights(weights, config.max_single_sector)
        return weights

    elif config.weighting == WeightingScheme.INVERSE_VOL:
        if broker is None:
            w = 1.0 / len(selected)
            return {s: w for s, _ in selected}
        inv_vols = {}
        for s, _ in selected:
            vol = calculate_sector_volatility(
                broker, s, config.vol_lookback_days, as_of
            )
            inv_vols[s] = 1.0 / max(vol, 0.01)
        total = sum(inv_vols.values())
        weights = {s: iv / total for s, iv in inv_vols.items()}
        if config.max_single_sector < 1.0:
            weights = _cap_weights(weights, config.max_single_sector)
        return weights

    # fallback
    w = 1.0 / len(selected)
    return {s: w for s, _ in selected}


def _cap_weights(weights: dict[str, float], cap: float) -> dict[str, float]:
    """Redistribute excess weight from capped positions."""
    capped = {}
    excess = 0.0
    uncapped_count = 0
    for s, w in weights.items():
        if w > cap:
            capped[s] = cap
            excess += w - cap
        else:
            capped[s] = w
            uncapped_count += 1
    if excess > 0 and uncapped_count > 0:
        redistribution = excess / uncapped_count
        for s in capped:
            if capped[s] < cap:
                capped[s] += redistribution
    return capped


def map_to_execution_etfs(
    weights: dict[str, float],
    config: StrategyConfig,
) -> dict[str, float]:
    """
    Map signal ETFs to execution ETFs (leveraged if preferred and available).

    Returns dict of {execution_ticker: weight}.
    """
    exec_weights: dict[str, float] = {}
    for signal_etf, weight in weights.items():
        sec = signal_to_sector(signal_etf)
        if sec is None:
            exec_etf = signal_etf
        elif config.prefer_leveraged and sec.leverage_factor > 1:
            exec_etf = sec.leveraged_etf
        else:
            exec_etf = sec.signal_etf
        # Merge in case multiple signals map to same execution ETF
        exec_weights[exec_etf] = exec_weights.get(exec_etf, 0) + weight
    return exec_weights


def generate_target_portfolio(
    broker: BaseBroker,
    config: StrategyConfig,
    as_of: Optional[date] = None,
) -> dict[str, float]:
    """
    Run the full strategy logic and return target portfolio weights.

    Returns dict of {ticker: weight} where weights sum to ~1.0.
    This is the main entry point for the strategy engine.
    """
    # Step 1: Absolute momentum check
    risk_on = check_absolute_momentum(broker, config, as_of)

    if not risk_on:
        bonds = config.effective_bonds()
        logger.info("RISK-OFF regime → 100%% %s", bonds)
        return {bonds: 1.0}

    # Step 2: Calculate sector momentum scores
    ranked = compute_sector_scores(broker, config, as_of)
    logger.info(
        "Sector rankings: %s",
        ", ".join(f"{s}={m:.4f}" for s, m in ranked[:5]),
    )

    # Step 3: Select top sectors
    selected = select_top_sectors(ranked, config)

    if not selected:
        bonds = config.effective_bonds()
        logger.info("No sectors with positive momentum → 100%% %s", bonds)
        return {bonds: 1.0}

    logger.info(
        "Selected sectors: %s",
        ", ".join(f"{s}({m:.4f})" for s, m in selected),
    )

    # Step 4: Compute weights
    weights = compute_weights(selected, config, broker, as_of)

    # Step 5: Map to execution ETFs
    exec_weights = map_to_execution_etfs(weights, config)

    # Apply allocation percentage (cash buffer)
    exec_weights = {
        etf: w * config.allocation_pct for etf, w in exec_weights.items()
    }

    logger.info(
        "Target portfolio: %s",
        ", ".join(f"{t}={w:.2%}" for t, w in exec_weights.items()),
    )

    return exec_weights
