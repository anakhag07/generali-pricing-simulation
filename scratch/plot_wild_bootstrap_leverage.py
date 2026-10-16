"""Replay completed leverage diagnostics from saved summaries."""

import argparse
from pathlib import Path

from experiments.paths import results_root
from experiments.envelopes.wild_reporting import write_leverage_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=results_root())
    parser.add_argument("--wide-bounds", action="store_true",
                        help="Compare bounds 0.5, 1, 10, 100, 1000, and unbounded.")
    args = parser.parse_args()
    slugs = ("0p5", "1", "10", "100", "1000", "unbounded") if args.wide_bounds else ("0p5", "1", "3", "10")
    sources = [args.runs_root / f"wild-bootstrap-leverage-bound-{slug}" / "summary.json"
               for slug in slugs]
    destination = "wild-bootstrap-leverage-wide-bounds" if args.wide_bounds else "wild-bootstrap-leverage"
    print(write_leverage_report(sources, args.runs_root / destination))


if __name__ == "__main__":
    main()
