"""Replay the completed four-bound leverage diagnostic from saved summaries."""

import argparse
from pathlib import Path

from experiments.paths import results_root
from experiments.envelopes.wild_reporting import write_leverage_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=results_root())
    args = parser.parse_args()
    sources = [args.runs_root / f"wild-bootstrap-leverage-bound-{slug}" / "summary.json"
               for slug in ("0p5", "1", "3", "10")]
    print(write_leverage_report(sources, args.runs_root / "wild-bootstrap-leverage"))


if __name__ == "__main__":
    main()
