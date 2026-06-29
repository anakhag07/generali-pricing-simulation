from __future__ import annotations

from scripts import strict_stein_backend_verification as script


def test_strict_summary_detects_no_peer_failures_under_tolerance() -> None:
    events = [
        {
            "event": "gradient",
            "source": "jac",
            "peer_grad_linf_diff": 1e-12,
            "peer_grad_l2_diff": 2e-12,
            "peer_u_linf_diff": 1e-13,
            "peer_values_plus_linf_diff": 1e-11,
            "peer_values_minus_linf_diff": 1e-11,
            "peer_grad_cosine": 1.0,
        },
        {"event": "value", "peer_value_diff": 1e-12},
    ]

    summary = script.strict_driver_summary(events, driver="numpy", grad_tol=1e-8, value_tol=1e-8)

    assert summary["first_gradient_failure"] is None
    assert summary["first_value_failure"] is None
    assert summary["gradient_max"]["peer_grad_linf_diff"] == 1e-12
    assert summary["value_max"]["peer_value_diff"] == 1e-12


def test_strict_summary_reports_first_peer_failure() -> None:
    events = [
        {
            "event": "gradient",
            "source": "jac",
            "peer_grad_linf_diff": 1e-12,
            "w_hash": "a",
            "theta_hash": "t0",
        },
        {
            "event": "gradient",
            "source": "callback_record",
            "peer_grad_linf_diff": 1e-4,
            "w_hash": "b",
            "theta_hash": "t1",
        },
        {"event": "value", "peer_value_diff": 2e-4, "theta_hash": "t2"},
    ]

    summary = script.strict_driver_summary(events, driver="jax", grad_tol=1e-8, value_tol=1e-8)

    assert summary["first_gradient_failure"] == {
        "index": 1,
        "source": "callback_record",
        "peer_grad_linf_diff": 1e-4,
        "w_hash": "b",
        "theta_hash": "t1",
    }
    assert summary["first_value_failure"] == {
        "index": 0,
        "peer_value_diff": 2e-4,
        "theta_hash": "t2",
    }
