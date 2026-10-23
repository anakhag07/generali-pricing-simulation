from types import SimpleNamespace

import numpy as np
import pytest
import sympy as sp

from experiments.envelopes.quadratic import (
    VARIABLE, QuadraticErrorCertificate, nonnegative_on_real_line,
)
from experiments.offline_bootstrap_math import (
    TransferredLCBObjective, optimize_transferred_lcb, strict_conjunction_empty,
    transferred_below_level, transferred_contains,
)


def optimizer_config():
    return {"step_rule": "l-bfgs-b", "starts": [0., .5, 1.], "t_steps": 300,
            "gradient_tolerance": 1e-10, "ftol": 1e-14,
            "global_gap_tolerance": 1e-7}


@pytest.mark.parametrize("polynomials,expected", [
    ([], False), ([0], True), ([1], False), ([-1], True),
    ([VARIABLE, -VARIABLE], True), ([VARIABLE, VARIABLE**2], False),
    ([-VARIABLE**2], True), ([VARIABLE**2, -VARIABLE**2], True),
    ([VARIABLE**2 - 1, 4 - VARIABLE**2], False),
])
def test_strict_conjunction_handles_constants_roots_and_tails(polynomials, expected):
    assert strict_conjunction_empty(polynomials) is expected


def test_strict_conjunction_refines_touching_root_intervals():
    a = VARIABLE
    assert not strict_conjunction_empty([-a*(a + sp.Rational(1, 10**20)), -a])
    assert strict_conjunction_empty([-a*(a + sp.Rational(1, 10**20)), a])


@pytest.mark.parametrize("constant", [-1., 1.])
def test_transferred_containment_detects_arbitrarily_narrow_failure(constant):
    a = VARIABLE
    q = sp.Poly((a - 2)**2 + 1 - sp.Rational(1, 10**20), a, domain=sp.QQ)
    assert not transferred_contains([constant, 0, 0], [constant, 0, 0], [0, 0, 0], q)
    touching = sp.Poly((a - 2)**2 + 1, a, domain=sp.QQ)
    assert transferred_contains([constant, 0, 0], [constant, 0, 0], [0, 0, 0], touching)


@pytest.mark.parametrize("candidate", [[0., 0., 0.], [4., -9., 2.], [-20., 1., -3.]])
def test_auxiliary_containment_implies_transferred_containment(candidate):
    cert = QuadraticErrorCertificate(np.eye(3), 1.)
    auxiliary, truth = np.array([.2, .3, -.1]), np.zeros(3)
    assert cert.contains(cert.difference(auxiliary, truth), 1.)
    assert transferred_contains(candidate, auxiliary, truth, cert.q)


def test_transfer_can_cover_even_when_auxiliary_band_does_not():
    q = sp.Poly(0, VARIABLE, domain=sp.QQ)
    assert transferred_contains([0, 0, 0], [2, -4, 8], [0, 0, 0], q)
    assert not transferred_contains([0, 0, 0], [0, 0, 0], [1, 0, 0], q)


@pytest.mark.parametrize("candidate,auxiliary,truth", [
    ([0, 0, 0], [0, 1, 0], [0, .5, 0]),
    ([0, 0, 0], [0, 1, 0], [0, 2, 0]),
    ([0, 0, 0], [0, 1, 0], [1, 0, 0]),
    ([1, 3, -4], [2, 0, -4], [0, 6, -4]),
    ([1, 3, -4], [2, 0, -4], [1, 3, -3]),
])
def test_zero_radius_containment_agrees_with_independent_interval_product(candidate, auxiliary, truth):
    def polynomial(coefficients):
        return sp.Poly.from_list([sp.Rational(float(c)) for c in coefficients[::-1]],
                                 VARIABLE, domain=sp.QQ)
    left = polynomial(auxiliary)
    right = 2*polynomial(candidate) - left
    target = polynomial(truth)
    expected = nonnegative_on_real_line(-(target-left)*(target-right))
    assert transferred_contains(candidate, auxiliary, truth, 0) is expected


def test_lcb_level_has_two_strict_branches_and_handles_the_kink():
    q = sp.Poly(0, VARIABLE, domain=sp.QQ)
    candidate, auxiliary = [0, 0, -1], [0, 1, -1]
    assert transferred_below_level(candidate, auxiliary, q, 0.)
    assert not transferred_below_level(candidate, auxiliary, q, -1e-12)
    # Each branch individually exceeds zero, but their minimum never does.
    assert transferred_below_level([0, 0, 0], [0, 1, 0], q, 0.)


def test_gradient_matches_finite_differences_away_from_kinks():
    v = np.array([[2., .1, -.2], [.1, 1., .15], [-.2, .15, 1.]])
    objective = TransferredLCBObjective([.1, 5.2, -4.9], [-.2, 5., -5.1], v, 2.3)
    for action in (-10., -.5, 0., .5, 1., 10.):
        step = 1e-5
        expected = (objective.value([action+step], None)
                    - objective.value([action-step], None))/(2*step)
        assert objective.grad([action], None)[0] == pytest.approx(expected, rel=1e-8, abs=1e-8)
    kink = TransferredLCBObjective([0, 0, -1], [0, 1, -1], np.eye(3), .5)
    assert kink.grad([0.], None)[0] == -1.


def test_repository_solver_returns_unconstrained_action_with_certified_gap():
    cfg = optimizer_config()
    result = optimize_transferred_lcb([0, 20, -5], [0, 20, -5], np.eye(3), 1., 0., cfg)
    assert result["optimization_state"] == "certified"
    assert result["action"] == pytest.approx(2., abs=1e-8)
    assert len(result["attempts"]) == len(cfg["starts"])
    assert 0 <= result["global_gap_upper"] <= cfg["global_gap_tolerance"]
    assert result["candidate_value_lower"] <= 20. <= result["global_value_upper"]


def test_repository_solver_certifies_cusp_despite_nonsmooth_gradient():
    result = optimize_transferred_lcb([0, 0, -1], [0, 1, -1], np.eye(3), 1., .4,
                                     optimizer_config())
    assert result["optimization_state"] == "certified"
    assert result["action"] == pytest.approx(0., abs=1e-7)
    assert result["candidate_value_lower"] <= -.4 <= result["global_value_upper"]


def test_successful_but_missed_optimizer_action_fails_global_certificate(monkeypatch):
    trace = SimpleNamespace(optimizer_success=True, optimizer_status=0, optimizer_message="mock success")
    monkeypatch.setattr("experiments.offline_bootstrap_math.run_first_order_minimize",
                        lambda *args, **kwargs: (np.array([0.]), trace))
    result = optimize_transferred_lcb([0, 5, -5], [0, 5, -5], np.eye(3), 1., 0.,
                                     optimizer_config())
    assert result["solver_success"]
    assert result["optimization_state"] == "uncertified"
    assert result["global_gap_upper"] is None
    assert result["global_value_upper"] is None


def test_failed_solver_status_does_not_override_valid_global_certificate(monkeypatch):
    trace = SimpleNamespace(optimizer_success=False, optimizer_status=2, optimizer_message="mock line search")
    monkeypatch.setattr("experiments.offline_bootstrap_math.run_first_order_minimize",
                        lambda *args, **kwargs: (np.array([0.]), trace))
    result = optimize_transferred_lcb([0, 0, -1], [0, 1, -1], np.eye(3), 1., .4,
                                     optimizer_config())
    assert not result["solver_success"]
    assert result["optimization_state"] == "certified"


@pytest.mark.parametrize("candidate,auxiliary,state", [
    ([0, 0, 2], [0, 0, 2], "unbounded"),
    ([0, 0, 1], [0, 0, 1], "degenerate_tail"),
    ([0, 0, -1], [0, 0, 2], "certified"),
])
def test_tail_classification_uses_the_lower_branch(candidate, auxiliary, state):
    result = optimize_transferred_lcb(candidate, auxiliary, np.eye(3), 1., 1., optimizer_config())
    assert result["optimization_state"] == state


def test_invalid_squared_radius_is_rejected():
    with pytest.raises(ValueError, match="nonnegative"):
        transferred_contains([0, 0, 0], [0, 0, 0], [0, 0, 0], -VARIABLE**2)
