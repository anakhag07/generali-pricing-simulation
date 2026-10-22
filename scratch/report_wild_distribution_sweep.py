"""Post-collection distribution comparison and variant-only context averages."""

import argparse
from pathlib import Path

from experiments.envelopes.wild_reporting import write_distribution_comparison
from plot_context_averaged_envelopes import render


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--thesis", type=Path, required=True)
    # The shared Slurm wrapper appends this flag; this replay never submits jobs.
    parser.add_argument("--no-sbatch", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    outputs = write_distribution_comparison(args.summary, args.baseline, args.thesis)
    render(args.summary)
    for path in outputs:
        print(path, flush=True)


if __name__ == "__main__":
    main()
