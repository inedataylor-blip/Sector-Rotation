"""Tests for the walk-forward backtest engine."""

from datetime import date

from sector_momentum_bot.config import (
    MomentumMethod,
    StrategyConfig,
    Variant,
    WeightingScheme,
)
from sector_momentum_bot.walkforward import (
    DEFAULT_PARAM_GRID,
    OptimizationMetric,
    WalkForwardResult,
    _add_months,
    _build_configs,
    _stitch_equity_curves,
    generate_windows,
    run_walk_forward,
    walk_forward_efficiency,
)

import numpy as np
import pandas as pd

from tests.helpers import make_price_series


# ── Date helper tests ───────────────────────────────────────────────────────

def test_add_months_basic():
    assert _add_months(date(2024, 1, 15), 3) == date(2024, 4, 15)


def test_add_months_year_boundary():
    assert _add_months(date(2024, 11, 1), 3) == date(2025, 2, 1)


def test_add_months_clamps_day():
    result = _add_months(date(2024, 1, 31), 1)
    assert result.month == 2
    assert result.day == 29  # 2024 is a leap year


def test_add_months_12():
    assert _add_months(date(2024, 6, 15), 12) == date(2025, 6, 15)


# ── Window generation tests ────────────────────────────────────────────────

def test_generate_windows_rolling():
    windows = generate_windows(
        data_start=date(2020, 1, 1),
        data_end=date(2024, 12, 31),
        in_sample_months=24,
        out_of_sample_months=6,
        step_months=6,
        anchored=False,
    )
    assert len(windows) > 0
    for is_start, is_end, oos_start, oos_end in windows:
        assert is_start < is_end
        assert is_end < oos_start
        assert oos_start < oos_end
        assert oos_end <= date(2024, 12, 31)


def test_generate_windows_anchored():
    windows = generate_windows(
        data_start=date(2020, 1, 1),
        data_end=date(2024, 12, 31),
        in_sample_months=24,
        out_of_sample_months=6,
        step_months=6,
        anchored=True,
    )
    assert len(windows) > 0
    for is_start, _, _, _ in windows:
        assert is_start == date(2020, 1, 1)


def test_generate_windows_no_overlap():
    windows = generate_windows(
        data_start=date(2020, 1, 1),
        data_end=date(2025, 12, 31),
        in_sample_months=24,
        out_of_sample_months=6,
        step_months=6,
        anchored=False,
    )
    for i in range(len(windows) - 1):
        _, _, _, oos_end = windows[i]
        _, _, oos_start_next, _ = windows[i + 1]
        assert oos_end < oos_start_next


def test_generate_windows_too_short_returns_empty():
    windows = generate_windows(
        data_start=date(2024, 1, 1),
        data_end=date(2024, 6, 1),
        in_sample_months=36,
        out_of_sample_months=12,
        step_months=6,
    )
    assert len(windows) == 0


# ── Config grid tests ──────────────────────────────────────────────────────

def test_build_configs_empty_grid():
    cfg = StrategyConfig()
    result = _build_configs(cfg, {})
    assert len(result) == 1


def test_build_configs_single_param():
    cfg = StrategyConfig()
    grid = {"variant": [Variant.TOP_1, Variant.TOP_3_EQUAL]}
    result = _build_configs(cfg, grid)
    assert len(result) == 2
    variants = {c.variant for c in result}
    assert variants == {Variant.TOP_1, Variant.TOP_3_EQUAL}


def test_build_configs_cartesian_product():
    cfg = StrategyConfig()
    grid = {
        "variant": [Variant.TOP_1, Variant.TOP_3_EQUAL],
        "momentum_method": [MomentumMethod.SIMPLE_12M, MomentumMethod.COMPOSITE],
    }
    result = _build_configs(cfg, grid)
    assert len(result) == 4


def test_build_configs_preserves_base():
    cfg = StrategyConfig(allocation_pct=0.95, prefer_leveraged=False)
    grid = {"variant": [Variant.TOP_1]}
    result = _build_configs(cfg, grid)
    assert result[0].allocation_pct == 0.95
    assert result[0].prefer_leveraged is False


# ── Equity curve stitching tests ───────────────────────────────────────────

def test_stitch_single_curve():
    curve = pd.Series(
        [100, 110, 105],
        index=[date(2024, 1, 1), date(2024, 2, 1), date(2024, 3, 1)],
    )
    result = _stitch_equity_curves([curve], 100.0)
    assert len(result) == 3
    assert result.iloc[0] == 100.0


def test_stitch_two_curves():
    c1 = pd.Series(
        [100.0, 120.0],
        index=[date(2024, 1, 1), date(2024, 2, 1)],
    )
    c2 = pd.Series(
        [100.0, 90.0],
        index=[date(2024, 3, 1), date(2024, 4, 1)],
    )
    result = _stitch_equity_curves([c1, c2], 100.0)
    assert result.iloc[0] == 100.0
    assert result.iloc[1] == 120.0
    assert abs(result.iloc[2] - 120.0) < 0.01
    assert abs(result.iloc[3] - 108.0) < 0.01


def test_stitch_empty():
    result = _stitch_equity_curves([], 100_000)
    assert result.empty


# ── Full walk-forward integration tests ─────────────────────────────────────

def _make_wf_price_data() -> dict[str, pd.DataFrame]:
    """Generate ~3 years of synthetic data for all needed symbols."""
    symbols = [
        "XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB",
        "XLU", "XLRE", "XLC", "SPY", "BIL", "BND",
        "TECL", "FAS", "ERX", "CURE", "DRN", "TMF",
    ]
    data = {}
    for i, sym in enumerate(symbols):
        ret = 0.0003 + i * 0.00003
        data[sym] = make_price_series(
            daily_return=ret,
            days=900,
            start_date=date(2020, 1, 2),
            seed=100 + i,
        )
    return data


def test_run_walk_forward_no_optimisation():
    """Walk-forward with fixed config (no param grid)."""
    price_data = _make_wf_price_data()
    config = StrategyConfig(variant=Variant.TOP_3_EQUAL)

    result = run_walk_forward(
        price_data=price_data,
        base_config=config,
        data_start=date(2020, 1, 2),
        data_end=date(2022, 12, 31),
        in_sample_months=12,
        out_of_sample_months=6,
        step_months=6,
        param_grid={},
    )

    assert isinstance(result, WalkForwardResult)
    assert len(result.windows) >= 2
    assert not result.combined_equity_curve.empty
    assert result.combined_metrics.final_value > 0


def test_run_walk_forward_with_optimisation():
    """Walk-forward with a small param grid."""
    price_data = _make_wf_price_data()
    config = StrategyConfig()

    result = run_walk_forward(
        price_data=price_data,
        base_config=config,
        data_start=date(2020, 1, 2),
        data_end=date(2022, 12, 31),
        in_sample_months=12,
        out_of_sample_months=6,
        step_months=6,
        param_grid={
            "variant": [Variant.TOP_1, Variant.TOP_3_EQUAL],
        },
        opt_metric=OptimizationMetric(name="sharpe_ratio"),
    )

    assert len(result.windows) >= 2
    assert result.combined_metrics.cagr != 0
    for w in result.windows:
        assert w.best_config.variant in (Variant.TOP_1, Variant.TOP_3_EQUAL)


def test_run_walk_forward_anchored():
    """Anchored mode: IS always starts from data_start."""
    price_data = _make_wf_price_data()
    config = StrategyConfig()

    result = run_walk_forward(
        price_data=price_data,
        base_config=config,
        data_start=date(2020, 1, 2),
        data_end=date(2022, 12, 31),
        in_sample_months=12,
        out_of_sample_months=6,
        step_months=6,
        anchored=True,
    )

    assert len(result.windows) >= 1
    for w in result.windows:
        assert w.is_start == date(2020, 1, 2)


def test_walk_forward_efficiency_ratio():
    """WFER should be a finite number."""
    price_data = _make_wf_price_data()
    config = StrategyConfig()

    result = run_walk_forward(
        price_data=price_data,
        base_config=config,
        data_start=date(2020, 1, 2),
        data_end=date(2022, 12, 31),
        in_sample_months=12,
        out_of_sample_months=6,
        step_months=6,
    )

    wfer = walk_forward_efficiency(result)
    assert np.isfinite(wfer)


def test_window_summary_dataframe():
    """Summary DataFrame should have expected columns and row count."""
    price_data = _make_wf_price_data()
    config = StrategyConfig()

    result = run_walk_forward(
        price_data=price_data,
        base_config=config,
        data_start=date(2020, 1, 2),
        data_end=date(2022, 12, 31),
        in_sample_months=12,
        out_of_sample_months=6,
        step_months=6,
    )

    df = result.window_summary
    assert len(df) == len(result.windows)
    expected_cols = {
        "window", "is_start", "is_end", "oos_start", "oos_end",
        "variant", "momentum", "weighting", "num_sectors", "leveraged",
        "is_metric", "oos_cagr", "oos_sharpe", "oos_max_dd", "oos_trades",
    }
    assert expected_cols.issubset(set(df.columns))
