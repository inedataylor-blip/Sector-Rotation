"""Configuration for the Sector Momentum Strategy."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Variant(Enum):
    """Strategy variant."""
    TOP_1 = "top_1"                  # Variant A: single best sector
    TOP_3_EQUAL = "top_3_equal"      # Variant B: top 3, equal weight
    TOP_3_WEIGHTED = "top_3_weighted" # Variant C: top 3, momentum-weighted
    TOP_1_LEVERAGED = "top_1_leveraged"  # Variant D: top 1, prefer leveraged


class MomentumMethod(Enum):
    """Momentum calculation method."""
    SIMPLE_12M = "simple_12m"
    COMPOSITE = "composite"
    SMA_10M = "sma_10m"
    RISK_ADJUSTED = "risk_adjusted"


class WeightingScheme(Enum):
    """Position weighting scheme."""
    EQUAL = "equal"
    MOMENTUM = "momentum"
    INVERSE_VOL = "inverse_vol"


@dataclass
class SectorETF:
    """Mapping from a signal ETF to its leveraged execution ETF."""
    signal_etf: str
    leveraged_etf: str
    leverage_factor: int
    sector_name: str


# Sector-to-ETF mapping
SECTOR_MAP: dict[str, SectorETF] = {
    "technology":      SectorETF("XLK", "TECL", 3, "Technology"),
    "financials":      SectorETF("XLF", "FAS",  3, "Financials"),
    "energy":          SectorETF("XLE", "ERX",  3, "Energy"),
    "healthcare":      SectorETF("XLV", "CURE", 3, "Health Care"),
    "discretionary":   SectorETF("XLY", "XLY",  1, "Consumer Discretionary"),
    "staples":         SectorETF("XLP", "XLP",  1, "Consumer Staples"),
    "industrials":     SectorETF("XLI", "XLI",  1, "Industrials"),
    "materials":       SectorETF("XLB", "XLB",  1, "Materials"),
    "utilities":       SectorETF("XLU", "XLU",  1, "Utilities"),
    "realestate":      SectorETF("XLRE", "DRN", 3, "Real Estate"),
    "communications":  SectorETF("XLC", "XLC",  1, "Communication Services"),
}

# Subsector leveraged ETFs (optional universe expansion)
SUBSECTOR_MAP: dict[str, SectorETF] = {
    "semiconductors":  SectorETF("SMH",  "SOXL", 3, "Semiconductors"),
    "biotech":         SectorETF("XBI",  "LABU", 3, "Biotech"),
    "retail":          SectorETF("XRT",  "RETL", 3, "Retail"),
    "regional_banks":  SectorETF("KRE",  "DPST", 3, "Regional Banks"),
}

# Default list of signal ETFs (GICS sectors)
DEFAULT_SECTORS = ["XLK", "XLF", "XLE", "XLV", "XLY",
                   "XLP", "XLI", "XLB", "XLU", "XLRE", "XLC"]


def signal_to_sector(signal_etf: str) -> Optional[SectorETF]:
    """Look up a SectorETF by its signal ticker."""
    for s in list(SECTOR_MAP.values()) + list(SUBSECTOR_MAP.values()):
        if s.signal_etf == signal_etf:
            return s
    return None


@dataclass
class StrategyConfig:
    """Full strategy configuration."""

    # Universe
    sectors: list[str] = field(default_factory=lambda: list(DEFAULT_SECTORS))
    include_subsectors: bool = False

    # Signal ETFs
    market_benchmark: str = "SPY"
    riskfree: str = "BIL"
    bonds: str = "BND"
    bonds_leveraged: str = "TMF"

    # Strategy variant
    variant: Variant = Variant.TOP_3_EQUAL

    # Momentum
    momentum_method: MomentumMethod = MomentumMethod.COMPOSITE
    momentum_lookback_days: int = 252
    require_positive_momentum: bool = True

    # Leverage
    prefer_leveraged: bool = True
    max_leverage: int = 3
    use_leveraged_bonds: bool = False

    # Allocation
    weighting: WeightingScheme = WeightingScheme.EQUAL
    allocation_pct: float = 0.99  # leave 1% cash buffer
    num_top_sectors: int = 3

    # Rebalance
    rebalance_threshold: float = 0.05  # only rebalance if >5% drift

    # Risk management
    max_single_sector: float = 0.40
    min_sectors: int = 1
    max_sectors: int = 5

    # Volatility targeting (optional)
    vol_target: Optional[float] = None  # e.g. 0.15 for 15%
    vol_lookback_days: int = 60

    def effective_num_sectors(self) -> int:
        """Return how many sectors to hold based on variant."""
        if self.variant in (Variant.TOP_1, Variant.TOP_1_LEVERAGED):
            return 1
        return self.num_top_sectors

    def effective_bonds(self) -> str:
        """Return bond ticker to use in risk-off regime."""
        if self.use_leveraged_bonds:
            return self.bonds_leveraged
        return self.bonds
