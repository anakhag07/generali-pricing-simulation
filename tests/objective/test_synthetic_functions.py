"""Shared contract tests for synthetic ladder objectives.

Every implemented rung is enrolled here via ``_CASES``; a new rung gets full
coverage by adding factory cases and flipping it into
``IMPLEMENTED_SYNTHETIC_LADDER``.
"""

from __future__ import annotations

import numpy as np
import pytest

from objective import (
    IMPLEMENTED_SYNTHETIC_LADDER,
    PiecewiseConvex,
    PiecewiseNonconvexDoubleWell,
    SmoothedNonconvex,
    StronglyConvexQuadratic,
    SYNTHETIC_LADDER,
)

_X_DUMMY = np.zeros((1, 1), dtype=float)

_CASES = {
    "quadratic-d5": lambda seed=11: StronglyConvexQuadratic.from_seed(
        seed, dim=5, condition_number=10.0
    ),
    "quadratic-d20-ill": lambda seed=11: StronglyConvexQuadratic.from_seed(
        seed, dim=20, condition_number=1000.0
    ),
    "smoothed-d5": lambda seed=11: SmoothedNonconvex.from_seed(seed, dim=5, n_bumps=3),
    "smoothed-d12": lambda seed=11: SmoothedNonconvex.from_seed(seed, dim=12, n_bumps=5),
}

_STUB_CLASSES = (PiecewiseConvex, PiecewiseNonconvexDoubleWell)


def _probe_points(fn, rng: np.random.Generator, n: int = 8) -> list[np.ndarray]:
    w_star = fn.optimal_theta()
    scales = np.linspace(0.1, 6.0, n)
    return [w_star + scale * rng.normal(size=w_star.size) for scale in scales]


def _central_fd(fn, w: np.ndarray, step: float = 1e-6) -> np.ndarray:
    grad = np.zeros_like(w)
    for i in range(w.size):
        basis = np.zeros_like(w)
        basis[i] = step
        grad[i] = (fn.value(w + basis, _X_DUMMY) - fn.value(w - basis, _X_DUMMY)) / (2.0 * step)
    return grad


@pytest.fixture(params=sorted(_CASES), ids=sorted(_CASES))
def ladder_fn(request):
    return _CASES[request.param]()


class TestSyntheticFunctionContract:
    def test_from_seed_is_deterministic(self, request) -> None:
        for case_name, factory in _CASES.items():
            first, second = factory(seed=7), factory(seed=7)
            rng = np.random.default_rng(0)
            for w in _probe_points(first, rng, n=4):
                assert first.value(w, _X_DUMMY) == second.value(w, _X_DUMMY), case_name

    def test_different_seeds_differ(self) -> None:
        for case_name, factory in _CASES.items():
            first, second = factory(seed=7), factory(seed=8)
            assert not np.array_equal(first.optimal_theta(), second.optimal_theta()), case_name

    def test_grad_matches_central_finite_difference(self, ladder_fn) -> None:
        rng = np.random.default_rng(3)
        step = 1e-6
        for w in _probe_points(ladder_fn, rng):
            if ladder_fn.min_kink_distance(w) < 10.0 * step:
                continue
            analytical = ladder_fn.grad(w, _X_DUMMY)
            numerical = _central_fd(ladder_fn, w, step)
            scale = max(1.0, float(np.max(np.abs(analytical))))
            assert np.max(np.abs(analytical - numerical)) < 1e-4 * scale

    def test_gradient_zero_at_optimum(self, ladder_fn) -> None:
        if not ladder_fn.is_smooth:
            pytest.skip("gradient at the optimum is only required for smooth rungs")
        grad_star = ladder_fn.grad(ladder_fn.optimal_theta(), _X_DUMMY)
        assert np.linalg.norm(grad_star) < 1e-10

    def test_optimum_is_global_on_probes(self, ladder_fn) -> None:
        rng = np.random.default_rng(5)
        f_star = ladder_fn.optimal_value()
        probes = _probe_points(ladder_fn, rng, n=16) + list(ladder_fn.adversarial_probes())
        for w in probes:
            assert ladder_fn.value(np.asarray(w, dtype=float), _X_DUMMY) > f_star

    def test_optimal_value_matches_value_at_optimum(self, ladder_fn) -> None:
        value_at_star = ladder_fn.value(ladder_fn.optimal_theta(), _X_DUMMY)
        assert value_at_star == pytest.approx(ladder_fn.optimal_value(), abs=1e-12)

    def test_x_batch_is_ignored(self, ladder_fn) -> None:
        w = ladder_fn.optimal_theta() + 0.5
        assert ladder_fn.value(w, np.zeros((1, 1))) == ladder_fn.value(w, np.ones((4, 3)))

    def test_theta_validation(self, ladder_fn) -> None:
        wrong = np.zeros(ladder_fn.theta_dim() + 1)
        with pytest.raises(ValueError):
            ladder_fn.value(wrong, _X_DUMMY)
        with pytest.raises(ValueError):
            ladder_fn.value(np.full(ladder_fn.theta_dim(), np.nan), _X_DUMMY)


class TestLadderRegistry:
    def test_implemented_rungs_are_registered(self) -> None:
        assert set(IMPLEMENTED_SYNTHETIC_LADDER) <= set(SYNTHETIC_LADDER)
        assert IMPLEMENTED_SYNTHETIC_LADDER == ("quadratic", "smoothed_nonconvex")

    def test_stub_rungs_raise_until_implemented(self) -> None:
        """When a stub gets implemented this test fails, forcing it into
        IMPLEMENTED_SYNTHETIC_LADDER (and thereby into the contract tests)."""
        stub_names = set(SYNTHETIC_LADDER) - set(IMPLEMENTED_SYNTHETIC_LADDER)
        assert stub_names == {"piecewise_convex", "piecewise_nonconvex"}
        for name in sorted(stub_names):
            stub = SYNTHETIC_LADDER[name].from_seed(2, dim=3)
            assert stub.theta_dim() == 3
            with pytest.raises(NotImplementedError):
                stub.value(np.zeros(3), _X_DUMMY)
            with pytest.raises(NotImplementedError):
                stub.grad(np.zeros(3), _X_DUMMY)

    def test_metadata_flags(self) -> None:
        assert StronglyConvexQuadratic.is_convex and StronglyConvexQuadratic.is_smooth
        assert not SmoothedNonconvex.is_convex and SmoothedNonconvex.is_smooth
        assert PiecewiseConvex.is_convex and not PiecewiseConvex.is_smooth
        assert not PiecewiseNonconvexDoubleWell.is_convex
        assert not PiecewiseNonconvexDoubleWell.is_smooth


class TestSmoothedNonconvexInvariants:
    def test_rejects_trap_support_touching_optimum(self) -> None:
        with pytest.raises(ValueError, match="exclude w_star"):
            SmoothedNonconvex(
                w_star=np.zeros(2),
                center_depth=1.0,
                center_width=1.0,
                bump_centers=np.array([[1.0, 0.0]]),
                bump_depths=np.array([0.1]),
                bump_radii=np.array([1.5]),
            )

    def test_rejects_depth_over_budget(self) -> None:
        with pytest.raises(ValueError, match="budget"):
            SmoothedNonconvex(
                w_star=np.zeros(2),
                center_depth=1.0,
                center_width=1.0,
                bump_centers=np.array([[3.0, 0.0]]),
                bump_depths=np.array([5.0]),  # budget is 0.5 * 2^2 = 2
                bump_radii=np.array([1.0]),
            )

    def test_rejects_overlapping_trap_supports(self) -> None:
        with pytest.raises(ValueError, match="disjoint"):
            SmoothedNonconvex(
                w_star=np.zeros(2),
                center_depth=1.0,
                center_width=1.0,
                bump_centers=np.array([[3.0, 0.0], [3.5, 0.0]]),
                bump_depths=np.array([0.1, 0.1]),
                bump_radii=np.array([1.0, 1.0]),
            )

    def test_optimal_value_is_negative_center_depth(self) -> None:
        fn = SmoothedNonconvex.from_seed(11, dim=4, center_depth=2.5)
        assert fn.optimal_value() == pytest.approx(-2.5, abs=1e-12)
