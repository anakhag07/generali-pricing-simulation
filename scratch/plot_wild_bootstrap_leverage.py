"""Replay completed leverage diagnostics from saved summaries."""

import argparse
from pathlib import Path

from experiments.paths import results_root
from experiments.envelopes.wild_reporting import write_leverage_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=results_root())
    parser.add_argument("--summaries", type=Path, nargs="+", help="Explicit saved studies to compare.")
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--diagnostics", action="store_true", help="Replay moved-point distribution, bands, and radii.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--wide-bounds", action="store_true",
                        help="Compare bounds 0.5, 1, 10, 100, 1000, and unbounded.")
    group.add_argument("--single-outlier", action="store_true",
                       help="Compare one moved observation with a fixed [-3,3] bulk.")
    args = parser.parse_args()
    if args.summaries:
        if args.destination is None or args.wide_bounds or args.single_outlier:
            parser.error("Explicit summaries require --destination and no legacy sweep flag.")
        print(write_leverage_report(args.summaries, args.destination, diagnostics=args.diagnostics))
        return
    if args.single_outlier:
        slugs = ("3", "10", "20", "40", "100", "1000")
        prefix, destination = "wild-bootstrap-outlier", "wild-bootstrap-single-outlier"
    else:
        slugs = ("0p5", "1", "10", "100", "1000", "unbounded") if args.wide_bounds else ("0p5", "1", "3", "10")
        prefix = "wild-bootstrap-leverage-bound"
        destination = "wild-bootstrap-leverage-wide-bounds" if args.wide_bounds else "wild-bootstrap-leverage"
    sources = [args.runs_root / f"{prefix}-{slug}" / "summary.json" for slug in slugs]
    print(write_leverage_report(sources, args.destination or args.runs_root / destination,
                                diagnostics=args.diagnostics))


if __name__ == "__main__":
    main()
