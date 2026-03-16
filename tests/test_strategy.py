"""Tests for the sector rotation strategy engine."""

from sector_momentum_bot.config import StrategyConfig, Variant, WeightingScheme
from sector_momentum_bot.strategies.sector_rotation import (
    check_absolute_momentum,
    compute_sector_scores,
    compute_weights,
    generate_target_portfolio,
    map_to_execution_etfs,
    select_top_sectors,
)

from tests.helpers import make_test_broker


def _make_broker_risk_on():
    """SPY drifts up, BIL nearly flat → risk-on."""
    return make_test_broker(
        daily_returns={"SPY": 0.0005, "BIL": 0.0001},
    )


def _make_broker_risk_off():
    """SPY drifts down, BIL slightly up → risk-off."""
    return make_test_broker(
        daily_returns={"SPY": -0.0005, "BIL": 0.0001},
    )


def test_absolute_momentum_risk_on():
    broker = _make_broker_risk_on()
    config = StrategyConfig()
    assert check_absolute_momentum(broker, config, as_of=broker.as_of_date) is True


def test_absolute_momentum_risk_off():
    broker = _make_broker_risk_off()
    config = StrategyConfig()
    assert check_absolute_momentum(broker, config, as_of=broker.as_of_date) is False


def test_compute_sector_scores_returns_sorted():
    broker = _make_broker_risk_on()
    config = StrategyConfig()
    scores = compute_sector_scores(broker, config, as_of=broker.as_of_date)
    assert len(scores) == 11
    # Verify descending order
    for i in range(len(scores) - 1):
        assert scores[i][1] >= scores[i + 1][1]


def test_select_top_sectors_variant_a():
    ranked = [("XLK", 0.3), ("XLF", 0.2), ("XLE", 0.1)]
    config = StrategyConfig(variant=Variant.TOP_1)
    selected = select_top_sectors(ranked, config)
    assert len(selected) == 1
    assert selected[0][0] == "XLK"


def test_select_top_sectors_variant_b():
    ranked = [("XLK", 0.3), ("XLF", 0.2), ("XLE", 0.1), ("XLV", 0.05)]
    config = StrategyConfig(variant=Variant.TOP_3_EQUAL, num_top_sectors=3)
    selected = select_top_sectors(ranked, config)
    assert len(selected) == 3


def test_select_top_sectors_filters_negative():
    ranked = [("XLK", 0.1), ("XLF", -0.05), ("XLE", -0.1)]
    config = StrategyConfig(require_positive_momentum=True)
    selected = select_top_sectors(ranked, config)
    assert len(selected) == 1
    assert selected[0][0] == "XLK"


def test_compute_weights_equal():
    selected = [("XLK", 0.3), ("XLF", 0.2), ("XLE", 0.1)]
    config = StrategyConfig(weighting=WeightingScheme.EQUAL)
    weights = compute_weights(selected, config)
    assert len(weights) == 3
    for w in weights.values():
        assert abs(w - 1 / 3) < 0.001


def test_compute_weights_momentum():
    selected = [("XLK", 0.3), ("XLF", 0.2), ("XLE", 0.1)]
    config = StrategyConfig(weighting=WeightingScheme.MOMENTUM)
    weights = compute_weights(selected, config)
    assert weights["XLK"] > weights["XLE"]
    assert abs(sum(weights.values()) - 1.0) < 0.01


def test_map_to_execution_etfs_leveraged():
    weights = {"XLK": 0.5, "XLF": 0.5}
    config = StrategyConfig(prefer_leveraged=True)
    exec_w = map_to_execution_etfs(weights, config)
    assert "TECL" in exec_w
    assert "FAS" in exec_w


def test_map_to_execution_etfs_unleveraged():
    weights = {"XLK": 0.5, "XLF": 0.5}
    config = StrategyConfig(prefer_leveraged=False)
    exec_w = map_to_execution_etfs(weights, config)
    assert "XLK" in exec_w
    assert "XLF" in exec_w


def test_generate_target_portfolio_risk_on():
    broker = _make_broker_risk_on()
    config = StrategyConfig(variant=Variant.TOP_3_EQUAL)
    target = generate_target_portfolio(broker, config, as_of=broker.as_of_date)
    # Should have sector ETFs, not bonds
    assert "BND" not in target
    assert len(target) >= 1


def test_generate_target_portfolio_risk_off():
    broker = _make_broker_risk_off()
    config = StrategyConfig()
    target = generate_target_portfolio(broker, config, as_of=broker.as_of_date)
    assert "BND" in target
    assert abs(target["BND"] - 1.0) < 0.01
