"""Command-line interface for the Sector Momentum Bot."""

import argparse
import sys
from datetime import date
from pathlib import Path

from sector_momentum_bot.config import (
    MomentumMethod,
    StrategyConfig,
    Variant,
    WeightingScheme,
)
from sector_momentum_bot.logger import setup_logging


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Leveraged Sector Momentum Strategy Bot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Broker
    parser.add_argument(
        "--broker",
        choices=["alpaca", "tradier", "backtest"],
        default="alpaca",
        help="Broker to use (default: alpaca)",
    )

    # Strategy variant
    parser.add_argument(
        "--variant",
        choices=[v.value for v in Variant],
        default=Variant.TOP_3_EQUAL.value,
        help="Strategy variant (default: top_3_equal)",
    )

    # Momentum method
    parser.add_argument(
        "--momentum",
        choices=[m.value for m in MomentumMethod],
        default=MomentumMethod.COMPOSITE.value,
        help="Momentum calculation method (default: composite)",
    )

    # Weighting
    parser.add_argument(
        "--weighting",
        choices=[w.value for w in WeightingScheme],
        default=WeightingScheme.EQUAL.value,
        help="Position weighting scheme (default: equal)",
    )

    # Number of sectors
    parser.add_argument(
        "--num-sectors",
        type=int,
        default=3,
        help="Number of top sectors to hold (default: 3)",
    )

    # Leverage
    parser.add_argument(
        "--no-leverage",
        action="store_true",
        help="Disable leveraged ETFs (use unleveraged only)",
    )
    parser.add_argument(
        "--leveraged-bonds",
        action="store_true",
        help="Use TMF (3x Treasury) instead of BND for risk-off",
    )

    # Include subsectors
    parser.add_argument(
        "--include-subsectors",
        action="store_true",
        help="Include subsector ETFs (semiconductors, biotech, etc.)",
    )

    # Execution
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Calculate trades but do not execute",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Run even if not last trading day of month",
    )
    parser.add_argument(
        "--date",
        type=lambda s: date.fromisoformat(s),
        default=None,
        help="Override date for momentum calculations (YYYY-MM-DD)",
    )

    # Logging
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("logs"),
        help="Directory for log files (default: logs/)",
    )

    # Alpaca-specific
    parser.add_argument("--alpaca-paper", action="store_true", default=True)

    # Tradier-specific
    parser.add_argument("--tradier-sandbox", action="store_true", default=True)

    return parser.parse_args(argv)


def build_config(args: argparse.Namespace) -> StrategyConfig:
    return StrategyConfig(
        variant=Variant(args.variant),
        momentum_method=MomentumMethod(args.momentum),
        weighting=WeightingScheme(args.weighting),
        num_top_sectors=args.num_sectors,
        prefer_leveraged=not args.no_leverage,
        use_leveraged_bonds=args.leveraged_bonds,
        include_subsectors=args.include_subsectors,
    )


def create_broker(args: argparse.Namespace):
    if args.broker == "alpaca":
        from sector_momentum_bot.brokers.alpaca_broker import AlpacaBroker
        return AlpacaBroker(paper=args.alpaca_paper)
    elif args.broker == "tradier":
        from sector_momentum_bot.brokers.tradier_broker import TradierBroker
        return TradierBroker(sandbox=args.tradier_sandbox)
    elif args.broker == "backtest":
        raise ValueError(
            "Backtest broker requires price data. Use the backtest module directly."
        )
    else:
        raise ValueError(f"Unknown broker: {args.broker}")


def main(argv=None) -> int:
    args = parse_args(argv)
    setup_logging(level=args.log_level, log_dir=args.log_dir)

    config = build_config(args)
    broker = create_broker(args)

    from sector_momentum_bot.runner import run_strategy

    result = run_strategy(
        broker=broker,
        config=config,
        dry_run=args.dry_run,
        force=args.force,
        as_of=args.date,
        log_dir=args.log_dir,
    )

    if result is None:
        print("Strategy skipped (not rebalance day). Use --force to override.")
        return 0

    if result.errors:
        print(f"Completed with {len(result.errors)} error(s):")
        for err in result.errors:
            print(f"  - {err}")
        return 1

    print(f"Rebalance complete. Value: ${result.account_value_after:,.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
