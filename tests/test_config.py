"""Tests for configuration module."""

from sector_momentum_bot.config import (
    SECTOR_MAP,
    SUBSECTOR_MAP,
    SectorETF,
    StrategyConfig,
    Variant,
    WeightingScheme,
    signal_to_sector,
)


def test_sector_map_has_11_sectors():
    assert len(SECTOR_MAP) == 11


def test_subsector_map_has_4_entries():
    assert len(SUBSECTOR_MAP) == 4


def test_signal_to_sector_found():
    sec = signal_to_sector("XLK")
    assert sec is not None
    assert sec.sector_name == "Technology"
    assert sec.leveraged_etf == "TECL"
    assert sec.leverage_factor == 3


def test_signal_to_sector_unleveraged():
    sec = signal_to_sector("XLY")
    assert sec is not None
    assert sec.leverage_factor == 1
    assert sec.leveraged_etf == "XLY"


def test_signal_to_sector_subsector():
    sec = signal_to_sector("SMH")
    assert sec is not None
    assert sec.leveraged_etf == "SOXL"


def test_signal_to_sector_not_found():
    assert signal_to_sector("AAPL") is None


def test_config_defaults():
    cfg = StrategyConfig()
    assert cfg.variant == Variant.TOP_3_EQUAL
    assert cfg.num_top_sectors == 3
    assert cfg.prefer_leveraged is True
    assert len(cfg.sectors) == 11


def test_effective_num_sectors_top1():
    cfg = StrategyConfig(variant=Variant.TOP_1)
    assert cfg.effective_num_sectors() == 1


def test_effective_num_sectors_top3():
    cfg = StrategyConfig(variant=Variant.TOP_3_EQUAL, num_top_sectors=3)
    assert cfg.effective_num_sectors() == 3


def test_effective_bonds_default():
    cfg = StrategyConfig()
    assert cfg.effective_bonds() == "BND"


def test_effective_bonds_leveraged():
    cfg = StrategyConfig(use_leveraged_bonds=True)
    assert cfg.effective_bonds() == "TMF"
