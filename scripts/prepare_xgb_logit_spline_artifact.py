"""Convert the legacy XGBoost smoothing bundle into portable spline arrays."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence


_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from data.dataset_metadata import DATASET_PATH  # noqa: E402
from data.xgb_logit_spline import (  # noqa: E402
    prepare_xgb_logit_spline_artifact,
    save_xgb_logit_spline_artifact,
)


_DEFAULT_SOURCE = (
    _SRC_DIR
    / "data"
    / "models"
    / "spline_acceptance"
    / "acceptance_smoothing_wrapper_20260706_112929.pkl"
)
_DEFAULT_OUTPUT = (
    _SRC_DIR
    / "data"
    / "models"
    / "xgb_logit_spline"
    / "acceptance_xgb_logit_spline_20260706_112929.npz"
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=_DEFAULT_SOURCE)
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true", help="Overwrite an existing output artifact.")
    return parser


def main(argv: Sequence[str] | None = None) -> Path:
    args = _build_parser().parse_args(argv)
    artifact = prepare_xgb_logit_spline_artifact(args.source, args.dataset)
    output = save_xgb_logit_spline_artifact(artifact, args.output, overwrite=args.force)
    print(
        f"Saved {artifact.policy_ids.size} policy splines over "
        f"[{artifact.u_min:.2f}, {artifact.u_max:.2f}] to {output}"
    )
    return output


if __name__ == "__main__":
    main()
