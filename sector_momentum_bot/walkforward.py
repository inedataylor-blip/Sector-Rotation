"""
Walk-Forward Backtest Engine.

Divides historical data into rolling in-sample (IS) and out-of-sample (OOS)
windows. For each window, optimizes strategy parameters on the IS period,
then evaluates on the OOS period. The stitched OOS results give a realistic
estimate of live performance.

Two modes:
- Rolling: IS window slides forward each step (fixed-length training).
- Anchored: IS window always starts from the beginning (expanding training).
"""

import itertools
import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Optional

import numpy as np
import pandas as pd

from sector_momentum_bot.backtest import BacktestMetrics, run_backtest, _compute_metrics
from sector_momentum_bot.brokers.backtest_broker import BacktestBroker
from sector_momentum_bot.config import (
    MomentumMethod,
    StrategyConfig,
    Variant,
    WeightingScheme,
)

logger = logging.getLogger(__name__)


# ── Parameter grid ──────────────────────────────────────────────────────────

# Every field that can be optimised, with the values to try.
DEFAULT_PARAM_GRID: dict[str, list[Any]] = {
    "variant": [Variant.TOP_1, Variant.TOP_3_EQUAL, Variant.TOP_3_WEIGHTED],
    "momentum_method": [MomentumMethod.SIMPLE_12M, MomentumMethod.COMPOSITE],
    "num_top_sectors": [1, 3, 5],
    "weighting": [WeightingScheme.EQUAL, WeightingScheme.MOMENTUM],
    "prefer_leveraged": [True, False],
}


@dataclass
class OptimizationMetric:
    """Which metric to optimise and in which direction."""
    name: str = "sharpe_ratio"  # attribute name on BacktestMetrics
    higher_is_better: bool = True


# ── Per-window result ───────────────────────────────────────────────────────

@dataclass
class WindowResult:
    """Result from a single walk-forward window."""
    window_index: int
    is_start: date
    is_end: date
    oos_start: date
    oos_end: date

    # Best config found during in-sample optimisation
    best_config: StrategyConfig
    best_is_metric: float

    # Out-of-sample performance with that config
    oos_metrics: BacktestMetrics
    oos_equity_curve: pd.Series
    oos_holdings: list[dict]


# ── Aggregate result ────────────────────────────────────────────────────────

@dataclass
class WalkForwardResult:
    """Aggregate walk-forward output."""
    windows: list[WindowResult]
    combined_equity_curve: pd.Series
    combined_metrics: BacktestMetrics
    window_summary: pd.DataFrame  # one row per window


# ── Date helpers ────────────────────────────────────────────────────────────

def _add_months(d: date, months: int) -> date:
    """Add *months* calendar months to *d*, clamping to valid day."""
    month = d.month + months
    year = d.year + (month - 1) // 12
    month = (month - 1) % 12 + 1
    # clamp day (e.g. Jan-31 + 1 month → Feb-28)
    import calendar
    max_day = calendar.monthrange(year, month)[1]
    return date(year, month, min(d.day, max_day))


def generate_windows(
    data_start: date,
    data_end: date,
    in_sample_months: int,
    out_of_sample_months: int,
    step_months: int,
    anchored: bool = False,
) -> list[tuple[date, date, date, date]]:
    """
    Generate (is_start, is_end, oos_start, oos_end) tuples.

    In rolling mode the IS window slides by *step_months* each iteration.
    In anchored mode the IS window always starts at *data_start*.
    """
    windows: list[tuple[date, date, date, date]] = []
    idx = 0
    while True:
        if anchored:
            is_start = data_start
            # In anchored mode the IS window expands: its end advances each step
            is_end = _add_months(data_start, in_sample_months + idx * step_months) - timedelta(days=1)
        else:
            is_start = _add_months(data_start, idx * step_months)
            is_end = _add_months(is_start, in_sample_months) - timedelta(days=1)

        oos_start = is_end + timedelta(days=1)
        oos_end = _add_months(oos_start, out_of_sample_months) - timedelta(days=1)

        if oos_end > data_end:
            break

        windows.append((is_start, is_end, oos_start, oos_end))
        idx += 1

    return windows


# ── Grid helpers ────────────────────────────────────────────────────────────

def _build_configs(
    base_config: StrategyConfig,
    param_grid: dict[str, list[Any]],
) -> list[StrategyConfig]:
    """
    Expand *param_grid* into the full Cartesian product of StrategyConfigs.
    """
    if not param_grid:
        return [base_config]

    keys = list(param_grid.keys())
    value_lists = [param_grid[k] for k in keys]
    configs: list[StrategyConfig] = []

    for combo in itertools.product(*value_lists):
        from dataclasses import asdict
        cfg_dict = asdict(base_config)
        for k, v in zip(keys, combo):
            cfg_dict[k] = v
        # StrategyConfig uses enums/optionals—reconstruct properly
        cfg = StrategyConfig(**cfg_dict)
        configs.append(cfg)

    return configs


def _metric_value(metrics: BacktestMetrics, name: str) -> float:
    """Extract a named metric as a float."""
    return float(getattr(metrics, name, 0.0))


# ── Core walk-forward engine ───────────────────────────────────────────────

def _optimise_in_sample(
    price_data: dict[str, pd.DataFrame],
    is_start: date,
    is_end: date,
    base_config: StrategyConfig,
    param_grid: dict[str, list[Any]],
    opt_metric: OptimizationMetric,
    initial_cash: float,
) -> tuple[StrategyConfig, float]:
    """
    Run every config in the grid on the IS window, return the best one.
    """
    configs = _build_configs(base_config, param_grid)
    best_cfg = base_config
    best_val = float("-inf") if opt_metric.higher_is_better else float("inf")

    for cfg in configs:
        try:
            result = run_backtest(
                price_data, cfg, is_start, is_end, initial_cash,
            )
            val = _metric_value(result.metrics, opt_metric.name)
        except Exception as e:
            logger.debug("IS grid error for %s: %s", cfg.variant.value, e)
            continue

        better = (
            val > best_val if opt_metric.higher_is_better else val < best_val
        )
        if better:
            best_val = val
            best_cfg = cfg

    return best_cfg, best_val


def run_walk_forward(
    price_data: dict[str, pd.DataFrame],
    base_config: StrategyConfig,
    data_start: date,
    data_end: date,
    in_sample_months: int = 36,
    out_of_sample_months: int = 6,
    step_months: int = 6,
    anchored: bool = False,
    param_grid: Optional[dict[str, list[Any]]] = None,
    opt_metric: Optional[OptimizationMetric] = None,
    initial_cash: float = 100_000.0,
) -> WalkForwardResult:
    """
    Run a full walk-forward backtest.

    Args:
        price_data: {symbol: DataFrame} with OHLCV data.
        base_config: Default strategy config (fields not in param_grid stay fixed).
        data_start: Earliest date in the dataset to use.
        data_end: Latest date in the dataset to use.
        in_sample_months: Length of the training window.
        out_of_sample_months: Length of the evaluation window.
        step_months: How far to slide the window each iteration.
        anchored: If True, IS window always starts at data_start.
        param_grid: Parameters to optimize. None → use base_config as-is.
        opt_metric: Metric to optimize on. Defaults to Sharpe ratio.
        initial_cash: Starting capital for each window.

    Returns:
        WalkForwardResult with per-window details and combined OOS metrics.
    """
    if opt_metric is None:
        opt_metric = OptimizationMetric()
    if param_grid is None:
        param_grid = {}

    windows = generate_windows(
        data_start, data_end,
        in_sample_months, out_of_sample_months, step_months,
        anchored,
    )

    if not windows:
        raise ValueError(
            f"No valid walk-forward windows for {data_start}–{data_end} "
            f"with IS={in_sample_months}m, OOS={out_of_sample_months}m"
        )

    logger.info(
        "Walk-forward: %d windows, IS=%dm, OOS=%dm, step=%dm, %s",
        len(windows), in_sample_months, out_of_sample_months,
        step_months, "anchored" if anchored else "rolling",
    )

    window_results: list[WindowResult] = []
    all_oos_curves: list[pd.Series] = []

    for i, (is_start, is_end, oos_start, oos_end) in enumerate(windows):
        logger.info(
            "Window %d/%d: IS %s→%s | OOS %s→%s",
            i + 1, len(windows), is_start, is_end, oos_start, oos_end,
        )

        # 1. Optimise on IS period
        if param_grid:
            best_cfg, best_is_val = _optimise_in_sample(
                price_data, is_start, is_end,
                base_config, param_grid, opt_metric, initial_cash,
            )
            logger.info(
                "  IS best: %s=%.4f | variant=%s, momentum=%s, weight=%s",
                opt_metric.name, best_is_val,
                best_cfg.variant.value,
                best_cfg.momentum_method.value,
                best_cfg.weighting.value,
            )
        else:
            best_cfg = base_config
            is_result = run_backtest(
                price_data, best_cfg, is_start, is_end, initial_cash,
            )
            best_is_val = _metric_value(is_result.metrics, opt_metric.name)

        # 2. Evaluate on OOS period with the chosen config
        oos_result = run_backtest(
            price_data, best_cfg, oos_start, oos_end, initial_cash,
        )

        logger.info(
            "  OOS: CAGR=%.2f%%, Sharpe=%.2f, MaxDD=%.2f%%",
            oos_result.metrics.cagr * 100,
            oos_result.metrics.sharpe_ratio,
            oos_result.metrics.max_drawdown * 100,
        )

        wr = WindowResult(
            window_index=i,
            is_start=is_start,
            is_end=is_end,
            oos_start=oos_start,
            oos_end=oos_end,
            best_config=best_cfg,
            best_is_metric=best_is_val,
            oos_metrics=oos_result.metrics,
            oos_equity_curve=oos_result.metrics.equity_curve,
            oos_holdings=oos_result.holdings_history,
        )
        window_results.append(wr)
        all_oos_curves.append(oos_result.metrics.equity_curve)

    # 3. Stitch OOS equity curves into one continuous series
    combined_curve = _stitch_equity_curves(all_oos_curves, initial_cash)

    # 4. Compute combined metrics
    total_trades = sum(w.oos_metrics.total_trades for w in window_results)
    combined_metrics = _compute_metrics(
        combined_curve, initial_cash,
        window_results[0].oos_start,
        window_results[-1].oos_end,
        total_trades,
    )

    # 5. Build summary DataFrame
    summary = _build_summary(window_results)

    return WalkForwardResult(
        windows=window_results,
        combined_equity_curve=combined_curve,
        combined_metrics=combined_metrics,
        window_summary=summary,
    )


# ── Post-processing ────────────────────────────────────────────────────────

def _stitch_equity_curves(
    curves: list[pd.Series],
    initial_cash: float,
) -> pd.Series:
    """
    Chain OOS equity curves so each starts where the previous ended.

    The first curve runs from initial_cash. Each subsequent curve is rescaled
    so its starting value equals the ending value of the previous one, giving
    a continuous compounded return stream.
    """
    if not curves:
        return pd.Series(dtype=float)

    segments: list[pd.Series] = []
    running_value = initial_cash

    for curve in curves:
        if curve.empty:
            continue
        # Normalise so the curve starts at running_value
        scale = running_value / curve.iloc[0] if curve.iloc[0] != 0 else 1.0
        scaled = curve * scale
        segments.append(scaled)
        running_value = scaled.iloc[-1]

    if not segments:
        return pd.Series(dtype=float)

    combined = pd.concat(segments)
    # Drop duplicate index entries (window boundaries might overlap)
    combined = combined[~combined.index.duplicated(keep="last")]
    return combined.sort_index()


def _build_summary(windows: list[WindowResult]) -> pd.DataFrame:
    """Build a DataFrame with one row per walk-forward window."""
    rows = []
    for w in windows:
        rows.append({
            "window": w.window_index,
            "is_start": w.is_start,
            "is_end": w.is_end,
            "oos_start": w.oos_start,
            "oos_end": w.oos_end,
            "variant": w.best_config.variant.value,
            "momentum": w.best_config.momentum_method.value,
            "weighting": w.best_config.weighting.value,
            "num_sectors": w.best_config.effective_num_sectors(),
            "leveraged": w.best_config.prefer_leveraged,
            "is_metric": round(w.best_is_metric, 4),
            "oos_cagr": round(w.oos_metrics.cagr, 4),
            "oos_sharpe": round(w.oos_metrics.sharpe_ratio, 4),
            "oos_max_dd": round(w.oos_metrics.max_drawdown, 4),
            "oos_trades": w.oos_metrics.total_trades,
        })
    return pd.DataFrame(rows)


# ── Walk-forward efficiency ratio ──────────────────────────────────────────

def walk_forward_efficiency(result: WalkForwardResult) -> float:
    """
    Walk-Forward Efficiency Ratio (WFER).

    WFER = mean(OOS metric) / mean(IS metric)

    A value near 1.0 means OOS performance closely matches IS performance,
    indicating robust parameters. Values well below 1.0 suggest overfitting.
    """
    is_vals = [w.best_is_metric for w in result.windows if w.best_is_metric != 0]
    oos_vals = [
        _metric_value(w.oos_metrics, "sharpe_ratio")
        for w in result.windows
    ]
    if not is_vals or not oos_vals:
        return 0.0
    mean_is = np.mean(is_vals)
    mean_oos = np.mean(oos_vals)
    if mean_is == 0:
        return 0.0
    return float(mean_oos / mean_is)


# ── Reporting ───────────────────────────────────────────────────────────────

def print_walk_forward_report(result: WalkForwardResult) -> None:
    """Print a formatted walk-forward summary."""
    m = result.combined_metrics
    wfer = walk_forward_efficiency(result)

    print("\n" + "=" * 70)
    print("WALK-FORWARD BACKTEST RESULTS")
    print("=" * 70)
    print(f"Windows:               {len(result.windows)}")
    print(f"OOS Period:            {m.start_date} to {m.end_date}")
    print(f"Initial Value:         ${m.initial_value:>12,.2f}")
    print(f"Final Value:           ${m.final_value:>12,.2f}")
    print(f"Combined CAGR:         {m.cagr:>11.2%}")
    print(f"Combined Max Drawdown: {m.max_drawdown:>11.2%}")
    print(f"Combined Sharpe:       {m.sharpe_ratio:>11.2f}")
    print(f"Combined Sortino:      {m.sortino_ratio:>11.2f}")
    print(f"WF Efficiency Ratio:   {wfer:>11.2f}")
    print(f"Total OOS Trades:      {m.total_trades:>11d}")
    print()

    print("Per-Window Summary:")
    print("-" * 70)
    fmt = "{:>3}  {:>10} {:>10}  {:>13} {:>8} {:>7} {:>8}"
    print(fmt.format(
        "#", "OOS Start", "OOS End", "Variant", "Sharpe", "CAGR", "MaxDD",
    ))
    print("-" * 70)
    for w in result.windows:
        print(fmt.format(
            w.window_index,
            str(w.oos_start),
            str(w.oos_end),
            w.best_config.variant.value,
            f"{w.oos_metrics.sharpe_ratio:.2f}",
            f"{w.oos_metrics.cagr:.1%}",
            f"{w.oos_metrics.max_drawdown:.1%}",
        ))
    print("=" * 70)
