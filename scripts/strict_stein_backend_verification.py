"""Strict single-driver verification for NumPy/JAX Stein-difference parity."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

try:  # Support both ``python scripts/foo.py`` and test imports as ``scripts.foo``.
    from scripts.diagnose_stein_backend_divergence import (  # type: ignore[import-not-found]
        DEFAULT_OUTPUT_ROOT,
        reconstruct_backend_pair,
        run_instrumented_optimizer,
        write_csv,
    )
except ImportError:  # pragma: no cover - exercised when invoked as a script path.
    from diagnose_stein_backend_divergence import (  # type: ignore[no-redef]
        DEFAULT_OUTPUT_ROOT,
        reconstruct_backend_pair,
        run_instrumented_optimizer,
        write_csv,
    )


DRIVERS = ("numpy", "jax", "both")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one trust-constr driver backend and verify that the peer backend "
            "computes the same Stein-difference quantities at each identical theta "
            "and perturbation block."
        )
    )
    parser.add_argument("--numpy-summary", type=Path, required=True)
    parser.add_argument("--jax-summary", type=Path, required=True)
    parser.add_argument("--estimator", default="stein_difference")
    parser.add_argument("--driver", choices=DRIVERS, default="both")
    parser.add_argument("--maxiter", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--grad-tol", type=float, default=1e-8)
    parser.add_argument("--value-tol", type=float, default=1e-8)
    return parser.parse_args(argv)


def strict_driver_summary(
    events: Sequence[Mapping[str, Any]],
    *,
    driver: str,
    grad_tol: float,
    value_tol: float,
) -> dict[str, Any]:
    """Summarize peer-backend parity for a single driver trace."""
    gradient_events = [event for event in events if event.get("event") == "gradient"]
    value_events = [event for event in events if event.get("event") == "value"]
    return {
        "driver": driver,
        "event_count": len(events),
        "gradient_count": len(gradient_events),
        "value_count": len(value_events),
        "gradient_max": _gradient_maxima(gradient_events),
        "value_max": _value_maxima(value_events),
        "first_gradient_failure": first_gradient_failure(gradient_events, grad_tol),
        "first_value_failure": first_value_failure(value_events, value_tol),
    }


def first_gradient_failure(
    gradient_events: Sequence[Mapping[str, Any]],
    tolerance: float,
) -> dict[str, Any] | None:
    """Return first gradient event whose peer difference exceeds tolerance."""
    for idx, event in enumerate(gradient_events):
        diff = abs(float(event.get("peer_grad_linf_diff", 0.0)))
        if diff > tolerance:
            return {
                "index": idx,
                "source": event.get("source"),
                "peer_grad_linf_diff": diff,
                "w_hash": event.get("w_hash"),
                "theta_hash": event.get("theta_hash"),
            }
    return None


def first_value_failure(
    value_events: Sequence[Mapping[str, Any]],
    tolerance: float,
) -> dict[str, Any] | None:
    """Return first value event whose peer difference exceeds tolerance."""
    for idx, event in enumerate(value_events):
        diff = abs(float(event.get("peer_value_diff", 0.0)))
        if diff > tolerance:
            return {
                "index": idx,
                "peer_value_diff": diff,
                "theta_hash": event.get("theta_hash"),
            }
    return None


def run_driver(
    *,
    pair: Any,
    estimator: str,
    driver: str,
    maxiter: int,
    output_dir: Path,
    grad_tol: float,
    value_tol: float,
) -> dict[str, Any]:
    """Run one strict driver and write event/parity outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    theta_final, trace, recorder = run_instrumented_optimizer(pair, estimator, driver, maxiter)
    events = recorder.events
    gradient_events = [event for event in events if event.get("event") == "gradient"]
    value_events = [event for event in events if event.get("event") == "value"]

    write_csv(output_dir / "events.csv", events)
    write_csv(output_dir / "gradient_parity.csv", gradient_events)
    write_csv(output_dir / "value_parity.csv", value_events)

    summary = strict_driver_summary(
        events,
        driver=driver,
        grad_tol=grad_tol,
        value_tol=value_tol,
    )
    summary.update(
        {
            "maxiter": maxiter,
            "optimizer_steps": len(trace.steps),
            "theta_final_head": theta_final[:5].tolist(),
            "theta_final_l2": float(np.linalg.norm(theta_final)),
        }
    )
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    _write_markdown_summary(output_dir / "summary.md", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    pair = reconstruct_backend_pair(args.numpy_summary, args.jax_summary, args.estimator)
    maxiter = int(args.maxiter if args.maxiter is not None else pair.config.t_steps)
    output_root = args.output_dir or DEFAULT_OUTPUT_ROOT / (
        f"strict_{args.estimator}_{args.numpy_summary.parent.name}_vs_{args.jax_summary.parent.name}"
    )
    drivers = ("numpy", "jax") if args.driver == "both" else (args.driver,)
    output_root.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for driver in drivers:
        target_dir = output_root / f"driver_{driver}" if len(drivers) > 1 else output_root
        summaries[driver] = run_driver(
            pair=pair,
            estimator=args.estimator,
            driver=driver,
            maxiter=maxiter,
            output_dir=target_dir,
            grad_tol=float(args.grad_tol),
            value_tol=float(args.value_tol),
        )
    with (output_root / "strict_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summaries, handle, indent=2, sort_keys=True)
    print(f"Wrote strict Stein backend verification to {output_root}")


def _gradient_maxima(events: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    keys = (
        "peer_grad_linf_diff",
        "peer_grad_l2_diff",
        "peer_u_linf_diff",
        "peer_values_plus_linf_diff",
        "peer_values_minus_linf_diff",
    )
    maxima = {key: 0.0 for key in keys}
    min_cosine = float("inf")
    for event in events:
        for key in keys:
            maxima[key] = max(maxima[key], abs(float(event.get(key, 0.0))))
        if "peer_grad_cosine" in event:
            min_cosine = min(min_cosine, float(event["peer_grad_cosine"]))
    maxima["peer_grad_cosine_min"] = min_cosine if np.isfinite(min_cosine) else float("nan")
    return maxima


def _value_maxima(events: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    max_diff = 0.0
    for event in events:
        max_diff = max(max_diff, abs(float(event.get("peer_value_diff", 0.0))))
    return {"peer_value_diff": max_diff}


def _write_markdown_summary(path: Path, summary: Mapping[str, Any]) -> None:
    lines = [
        "# Strict Stein Backend Verification",
        "",
        f"- Driver: `{summary['driver']}`",
        f"- maxiter: `{summary['maxiter']}`",
        f"- optimizer steps: `{summary['optimizer_steps']}`",
        f"- gradient events: `{summary['gradient_count']}`",
        f"- value events: `{summary['value_count']}`",
        f"- first gradient failure: `{summary['first_gradient_failure']}`",
        f"- first value failure: `{summary['first_value_failure']}`",
        "",
        "## Gradient Maxima",
    ]
    for key, value in sorted(summary.get("gradient_max", {}).items()):
        lines.append(f"- `{key}`: `{value}`")
    lines.append("")
    lines.append("## Value Maxima")
    for key, value in sorted(summary.get("value_max", {}).items()):
        lines.append(f"- `{key}`: `{value}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
