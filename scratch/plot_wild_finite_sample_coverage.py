"""Replay wild-bootstrap coverage against the thesis's finite-sample lower bound."""

import argparse
from pathlib import Path

from experiments.envelopes.wild_reporting import write_finite_sample_coverage_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("thesis", type=Path)
    parser.add_argument("--destination", type=Path)
    args = parser.parse_args()
    for output in write_finite_sample_coverage_report(args.summary, args.thesis, args.destination):
        print(output)
