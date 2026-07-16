"""Synthetic ladder objectives: benchmark functions over the decision vector w = theta.

Each rung is a direct theta-space objective (like `QuadraticObjective`): the
optimization variable is the full vector ``w``, ``x_batch`` is validated but
ignored, and there is no policy or action space. Every instance is
deterministic given its construction seed and knows its global minimizer
exactly by construction, so true-gap metrics need no reference runs.

Ladder registry: `SYNTHETIC_LADDER` maps rung names to classes;
`IMPLEMENTED_SYNTHETIC_LADDER` lists the rungs that are runnable today. The
piecewise rungs are structural stubs: their parametrization and construction
are fixed, but ``_f`` / ``_grad_f`` raise ``NotImplementedError`` until they
are implemented (intended formulas are recorded in MATH.md).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from objective.base import Objective, default_rng


def _fingerprint(function: "SyntheticFunction") -> str:
    """Hash $$w^*$$ plus $$f$$ at deterministic probes, identifying the function itself.

    `from_dict` compares this against the rebuilt instance, so a change to any
    `from_seed` construction fails loudly instead of silently replaying a
    different function under the same spec.
    """
    w_star = np.asarray(function.w_star, dtype=float)
    probes = w_star + np.random.default_rng(0).normal(size=(4, w_star.size))
    values = np.array([function._f(probe) for probe in probes], dtype=float)
    payload = np.concatenate([w_star, values]).astype("<f8").tobytes()
    return hashlib.sha256(payload).hexdigest()[:16]


class SyntheticFunction(Objective):
    """Ladder contract: $$f(w)$$ over the decision vector $$w = \\theta$$; ``x_batch`` is ignored.

    Subclasses set ``w_star`` (the known global minimizer), the metadata flags
    ``is_convex`` / ``is_smooth``, and the hooks ``_f`` / ``_grad_f``. The
    contract surface (`optimal_theta`, `optimal_value`, `min_kink_distance`,
    `adversarial_probes`) is what the shared contract tests exercise.
    """

    is_convex: ClassVar[bool]
    is_smooth: ClassVar[bool]

    w_star: np.ndarray

    def theta_dim(self, state_dim: int | None = None) -> int:
        """Return the decision dimension; state dimension is irrelevant."""
        del state_dim
        return int(np.asarray(self.w_star).size)

    def optimal_theta(self) -> np.ndarray:
        """Return the known global minimizer $$w^*$$."""
        return np.asarray(self.w_star, dtype=float).copy()

    def optimal_value(self) -> float:
        """Return $$f(w^*)$$, exact by construction."""
        return float(self._f(self.optimal_theta()))

    def min_kink_distance(self, theta: np.ndarray) -> float:
        """Distance from ``theta`` to the nearest nonsmooth point; ``inf`` for smooth rungs.

        Contract tests use this to exclude finite-difference probes near kinks.
        """
        del theta
        return float("inf")

    def adversarial_probes(self) -> np.ndarray:
        """Return (k, dim) probe points most likely to violate global minimality.

        Contract tests assert $$f(p) > f(w^*)$$ at every probe (e.g. trap
        centers). Smooth convex rungs have none.
        """
        return np.empty((0, self.theta_dim()), dtype=float)

    def rung_name(self) -> str:
        """Return this instance's `SYNTHETIC_LADDER` registry key."""
        for name, rung_cls in SYNTHETIC_LADDER.items():
            if type(self) is rung_cls:
                return name
        raise ValueError(f"{type(self).__name__} is not registered in SYNTHETIC_LADDER.")

    def to_dict(self) -> dict[str, Any]:
        """Serialize the construction spec, $$w^*$$, and a fingerprint.

        `w_star` is recorded directly so saved runs can report true-gap metrics
        without rebuilding; `spec` is what `from_dict` replays.
        """
        spec = getattr(self, "_spec", None)
        return {
            "type": type(self).__name__,
            "rung": self.rung_name(),
            "spec": dict(spec) if spec is not None else None,
            "w_star": [float(value) for value in self.optimal_theta()],
            "fingerprint": _fingerprint(self),
        }

    @staticmethod
    def from_dict(payload: dict[str, Any]) -> "SyntheticFunction":
        """Rebuild the instance recorded by `to_dict`, verifying the fingerprint."""
        rung = payload["rung"]
        if rung not in SYNTHETIC_LADDER:
            raise ValueError(f"Unknown synthetic ladder rung {rung!r}.")
        spec = payload.get("spec")
        if spec is None:
            raise ValueError(
                f"Rung {rung!r} was built directly rather than through a seeded factory, "
                "so it carries no replayable spec."
            )
        rung_cls = SYNTHETIC_LADDER[rung]
        factory = spec["factory"]
        if factory == "from_seed":
            function = rung_cls.from_seed(
                int(spec["seed"]), dim=int(spec["dim"]), **dict(spec.get("params") or {})
            )
        elif factory == "isotropic":
            function = rung_cls.isotropic(int(spec["dim"]))
        else:
            raise ValueError(f"Unknown synthetic ladder factory {factory!r}.")
        expected = payload.get("fingerprint")
        if expected is not None and _fingerprint(function) != expected:
            raise ValueError(
                f"Rebuilt rung {rung!r} does not match the recorded fingerprint; the "
                f"{factory!r} construction has changed since this run was saved."
            )
        return function

    def _record_spec(self, **spec: Any) -> None:
        """Attach the replay spec set by a seeded factory (frozen-dataclass safe)."""
        object.__setattr__(self, "_spec", spec)

    def value(self, theta: np.ndarray, x_batch: np.ndarray) -> float:
        """Return $$f(\\theta)$$; ``x_batch`` is intentionally ignored."""
        return float(self._f(self._validate_inputs(theta, x_batch)))

    def grad(self, theta: np.ndarray, x_batch: np.ndarray) -> np.ndarray:
        """Return the exact theta-gradient $$\\nabla f(\\theta)$$; ``x_batch`` is ignored."""
        return np.asarray(self._grad_f(self._validate_inputs(theta, x_batch)), dtype=float)

    def _validate_inputs(self, theta: np.ndarray, x_batch: np.ndarray) -> np.ndarray:
        theta_arr = np.asarray(theta, dtype=float)
        dim = self.theta_dim()
        if theta_arr.ndim != 1 or theta_arr.size != dim:
            raise ValueError(f"theta must be a 1D array with dimension {dim}.")
        if not np.all(np.isfinite(theta_arr)):
            raise ValueError("theta must contain only finite values.")
        if np.ndim(x_batch) != 2:
            raise ValueError("x_batch must be a 2D array.")
        return theta_arr

    def _f(self, w: np.ndarray) -> float:
        raise NotImplementedError

    def _grad_f(self, w: np.ndarray) -> np.ndarray:
        raise NotImplementedError


def _validated_vector(name: str, values: np.ndarray, dim: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1 or arr.size != dim:
        raise ValueError(f"{name} must be a 1D array with dimension {dim}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values.")
    return arr


def _validated_rotation(rotation: np.ndarray, dim: int) -> np.ndarray:
    q_matrix = np.asarray(rotation, dtype=float)
    if q_matrix.shape != (dim, dim):
        raise ValueError(f"rotation must have shape ({dim}, {dim}).")
    if not np.allclose(q_matrix.T @ q_matrix, np.eye(dim), atol=1e-8):
        raise ValueError("rotation must be orthogonal.")
    return q_matrix


def _seeded_rotation(rng: np.random.Generator, dim: int) -> np.ndarray:
    q_matrix, r_matrix = np.linalg.qr(rng.normal(size=(dim, dim)))
    signs = np.sign(np.diag(r_matrix))
    signs[signs == 0.0] = 1.0
    return q_matrix * signs


@dataclass(frozen=True)
class StronglyConvexQuadratic(SyntheticFunction):
    """Rung 1: $$f(w) = \\frac{1}{2}(w - w^*)^\\top A (w - w^*)$$ with $$A = Q\\,\\mathrm{diag}(\\lambda)\\,Q^\\top$$.

    ``eigenvalues`` are the spectrum of $$A$$ (all positive), so the function is
    $$\\min_j \\lambda_j$$-strongly convex and $$\\max_j \\lambda_j$$-smooth with
    unique minimizer $$w^*$$ and minimum value 0.
    """

    w_star: np.ndarray
    rotation: np.ndarray
    eigenvalues: np.ndarray

    is_convex: ClassVar[bool] = True
    is_smooth: ClassVar[bool] = True

    def __post_init__(self) -> None:
        w_star = np.asarray(self.w_star, dtype=float)
        if w_star.ndim != 1 or w_star.size < 1:
            raise ValueError("w_star must be a 1D array with at least one element.")
        dim = w_star.size
        rotation = _validated_rotation(self.rotation, dim)
        eigenvalues = _validated_vector("eigenvalues", self.eigenvalues, dim)
        if np.any(eigenvalues <= 0.0):
            raise ValueError("eigenvalues must all be positive.")
        object.__setattr__(self, "w_star", w_star)
        object.__setattr__(self, "rotation", rotation)
        object.__setattr__(self, "eigenvalues", eigenvalues)
        object.__setattr__(self, "_matrix", (rotation * eigenvalues) @ rotation.T)

    @classmethod
    def from_seed(
        cls,
        seed: int,
        *,
        dim: int,
        condition_number: float = 100.0,
        mu: float = 1.0,
        w_star_scale: float = 1.0,
    ) -> "StronglyConvexQuadratic":
        """Build a seeded instance: random $$w^*$$, Haar-like rotation, log-spaced
        spectrum in $$[\\mu, \\mu\\kappa]$$ (``condition_number`` needs ``dim >= 2`` to bite)."""
        if dim < 1:
            raise ValueError("dim must be positive.")
        if condition_number < 1.0:
            raise ValueError("condition_number must be >= 1.")
        if mu <= 0.0:
            raise ValueError("mu must be positive.")
        rng = default_rng(seed)
        w_star = rng.normal(size=dim) * float(w_star_scale)
        rotation = _seeded_rotation(rng, dim)
        eigenvalues = float(mu) * np.logspace(0.0, np.log10(condition_number), dim)
        function = cls(w_star=w_star, rotation=rotation, eigenvalues=eigenvalues)
        function._record_spec(
            factory="from_seed",
            seed=int(seed),
            dim=int(dim),
            params={
                "condition_number": float(condition_number),
                "mu": float(mu),
                "w_star_scale": float(w_star_scale),
            },
        )
        return function

    @classmethod
    def isotropic(cls, dim: int) -> "StronglyConvexQuadratic":
        """Build the unit-conditioned instance $$f(w) = \\frac{1}{2}\\|w\\|_2^2$$ at $$w^*=0$$."""
        if dim < 1:
            raise ValueError("dim must be positive.")
        function = cls(
            w_star=np.zeros(dim, dtype=float),
            rotation=np.eye(dim, dtype=float),
            eigenvalues=np.ones(dim, dtype=float),
        )
        function._record_spec(factory="isotropic", dim=int(dim))
        return function

    def _f(self, w: np.ndarray) -> float:
        diff = w - self.w_star
        return 0.5 * float(diff @ (self._matrix @ diff))

    def _grad_f(self, w: np.ndarray) -> np.ndarray:
        return self._matrix @ (w - self.w_star)


def _mollifier_and_slope(s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return $$\\psi(s) = e^{1 - 1/(1-s)}$$ on $$[0, 1)$$ (0 outside) and $$\\psi'(s)$$.

    The mollifier is $$C^\\infty$$ with compact support; the guard on the
    exponent avoids 0 * inf when $$s \\to 1^-$$.
    """
    s_arr = np.asarray(s, dtype=float)
    psi = np.zeros_like(s_arr)
    slope = np.zeros_like(s_arr)
    inside = s_arr < 1.0
    if np.any(inside):
        exponent = 1.0 - 1.0 / (1.0 - s_arr[inside])
        active = exponent > -700.0
        values = np.zeros_like(exponent)
        values[active] = np.exp(exponent[active])
        psi[inside] = values
        slopes = np.zeros_like(exponent)
        slopes[active] = -values[active] / (1.0 - s_arr[inside][active]) ** 2
        slope[inside] = slopes
    return psi, slope


@dataclass(frozen=True)
class SmoothedNonconvex(SyntheticFunction):
    """Rung 2: quadratic well deepened at $$w^*$$ with compactly supported off-center traps.

    $$f(w) = \\frac{1}{2}\\|w - w^*\\|^2 - a_0\\, e^{-\\|w - w^*\\|^2 / (2 s_0^2)}
    - \\sum_j a_j\\, \\psi\\bigl(\\|w - c_j\\|^2 / \\rho_j^2\\bigr)$$

    with the $$C^\\infty$$ mollifier $$\\psi(s) = e^{1 - 1/(1-s)}$$ on $$[0,1)$$, 0
    outside, so trap $$j$$ has support $$\\{\\|w - c_j\\| < \\rho_j\\}$$. Construction
    enforces (i) support clearance $$\\|c_j - w^*\\| - \\rho_j > 0$$, (ii) pairwise
    disjoint trap supports, and (iii) the per-trap depth budget
    $$a_j < \\frac{1}{2}(\\|c_j - w^*\\| - \\rho_j)^2$$, which together make $$w^*$$
    the unique global minimizer with $$f(w^*) = -a_0$$ exactly (see MATH.md).
    """

    w_star: np.ndarray
    center_depth: float
    center_width: float
    bump_centers: np.ndarray
    bump_depths: np.ndarray
    bump_radii: np.ndarray

    is_convex: ClassVar[bool] = False
    is_smooth: ClassVar[bool] = True

    def __post_init__(self) -> None:
        w_star = np.asarray(self.w_star, dtype=float)
        if w_star.ndim != 1 or w_star.size < 1:
            raise ValueError("w_star must be a 1D array with at least one element.")
        dim = w_star.size
        if float(self.center_depth) < 0.0:
            raise ValueError("center_depth must be non-negative.")
        if float(self.center_width) <= 0.0:
            raise ValueError("center_width must be positive.")
        centers = np.asarray(self.bump_centers, dtype=float)
        if centers.ndim != 2 or centers.shape[1] != dim or centers.shape[0] < 1:
            raise ValueError(f"bump_centers must be a 2D array with shape (n_bumps, {dim}).")
        n_bumps = centers.shape[0]
        depths = _validated_vector("bump_depths", self.bump_depths, n_bumps)
        radii = _validated_vector("bump_radii", self.bump_radii, n_bumps)
        if np.any(depths <= 0.0):
            raise ValueError("bump_depths must all be positive.")
        if np.any(radii <= 0.0):
            raise ValueError("bump_radii must all be positive.")
        clearances = np.linalg.norm(centers - w_star, axis=1) - radii
        if np.any(clearances <= 0.0):
            raise ValueError("every bump support must exclude w_star: ||c_j - w*|| - rho_j must be positive.")
        if np.any(depths >= 0.5 * clearances**2):
            raise ValueError("bump depths violate the global-minimum budget a_j < 0.5 * clearance_j^2.")
        center_gaps = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)
        radius_sums = radii[:, None] + radii[None, :]
        overlapping = (center_gaps <= radius_sums) & ~np.eye(n_bumps, dtype=bool)
        if np.any(overlapping):
            raise ValueError("bump supports must be pairwise disjoint.")
        object.__setattr__(self, "w_star", w_star)
        object.__setattr__(self, "center_depth", float(self.center_depth))
        object.__setattr__(self, "center_width", float(self.center_width))
        object.__setattr__(self, "bump_centers", centers)
        object.__setattr__(self, "bump_depths", depths)
        object.__setattr__(self, "bump_radii", radii)

    @classmethod
    def from_seed(
        cls,
        seed: int,
        *,
        dim: int,
        n_bumps: int = 3,
        center_depth: float = 1.0,
        center_width: float = 1.0,
        bump_distance_range: tuple[float, float] = (3.0, 5.0),
        bump_radius_range: tuple[float, float] = (0.8, 1.2),
        depth_fraction: float = 0.9,
        w_star_scale: float = 1.0,
        max_placement_attempts: int = 200,
    ) -> "SmoothedNonconvex":
        """Build a seeded instance with disjoint traps at distance
        ``bump_distance_range`` from $$w^*$$, each at ``depth_fraction`` of its
        maximum admissible depth (larger distances allow deeper traps).

        ``depth_fraction`` controls whether the traps are *real* local minima, and
        only the budget $$a_j < \\frac{1}{2}\\gamma_j^2$$ (which keeps $$w^*$$ global)
        is enforced. Shallow traps leave the quadratic pull dominant and descent
        rolls straight through them: at the default 0.9 every trap is a local
        minimum, at 0.5-0.3 some are, and by 0.1 none are. Lower it only if a
        deliberately easier rung is wanted -- otherwise the rung stays labelled
        nonconvex while behaving unimodally."""
        if dim < 1:
            raise ValueError("dim must be positive.")
        if n_bumps < 1:
            raise ValueError("n_bumps must be positive.")
        if not 0.0 < depth_fraction < 1.0:
            raise ValueError("depth_fraction must lie in (0, 1).")
        distance_lo, distance_hi = (float(v) for v in bump_distance_range)
        radius_lo, radius_hi = (float(v) for v in bump_radius_range)
        if not 0.0 < distance_lo <= distance_hi:
            raise ValueError("bump_distance_range must be positive and ordered.")
        if not 0.0 < radius_lo <= radius_hi:
            raise ValueError("bump_radius_range must be positive and ordered.")
        if distance_lo - radius_hi <= 0.0:
            raise ValueError("bump_distance_range must clear bump_radius_range so supports exclude w_star.")
        rng = default_rng(seed)
        w_star = rng.normal(size=dim) * float(w_star_scale)
        radii = rng.uniform(radius_lo, radius_hi, size=n_bumps)
        centers = np.empty((n_bumps, dim), dtype=float)
        placed = 0
        for _ in range(int(max_placement_attempts)):
            if placed == n_bumps:
                break
            direction = rng.normal(size=dim)
            norm = np.linalg.norm(direction)
            if norm == 0.0:
                continue
            candidate = w_star + rng.uniform(distance_lo, distance_hi) * direction / norm
            gaps = np.linalg.norm(centers[:placed] - candidate, axis=1)
            if np.all(gaps > radii[:placed] + radii[placed]):
                centers[placed] = candidate
                placed += 1
        if placed < n_bumps:
            raise ValueError(
                f"could not place {n_bumps} disjoint bumps in {max_placement_attempts} attempts; "
                "reduce n_bumps or widen bump_distance_range."
            )
        clearances = np.linalg.norm(centers - w_star, axis=1) - radii
        depths = float(depth_fraction) * 0.5 * clearances**2
        function = cls(
            w_star=w_star,
            center_depth=float(center_depth),
            center_width=float(center_width),
            bump_centers=centers,
            bump_depths=depths,
            bump_radii=radii,
        )
        function._record_spec(
            factory="from_seed",
            seed=int(seed),
            dim=int(dim),
            params={
                "n_bumps": int(n_bumps),
                "center_depth": float(center_depth),
                "center_width": float(center_width),
                "bump_distance_range": [distance_lo, distance_hi],
                "bump_radius_range": [radius_lo, radius_hi],
                "depth_fraction": float(depth_fraction),
                "w_star_scale": float(w_star_scale),
                "max_placement_attempts": int(max_placement_attempts),
            },
        )
        return function

    def adversarial_probes(self) -> np.ndarray:
        """Trap centers plus points halfway between each trap center and its support edge."""
        toward_star = self.w_star - self.bump_centers
        norms = np.linalg.norm(toward_star, axis=1, keepdims=True)
        offsets = 0.5 * self.bump_radii[:, None] * toward_star / norms
        return np.vstack([self.bump_centers, self.bump_centers + offsets])

    def _bump_terms(self, w: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        diffs = w[None, :] - self.bump_centers
        s = np.sum(diffs**2, axis=1) / self.bump_radii**2
        psi, slope = _mollifier_and_slope(s)
        return diffs, psi, slope

    def _f(self, w: np.ndarray) -> float:
        diff = w - self.w_star
        r_sq = float(diff @ diff)
        value = 0.5 * r_sq - self.center_depth * np.exp(-r_sq / (2.0 * self.center_width**2))
        _, psi, _ = self._bump_terms(w)
        return float(value - self.bump_depths @ psi)

    def _grad_f(self, w: np.ndarray) -> np.ndarray:
        diff = w - self.w_star
        r_sq = float(diff @ diff)
        center_coef = 1.0 + (self.center_depth / self.center_width**2) * np.exp(
            -r_sq / (2.0 * self.center_width**2)
        )
        grad = center_coef * diff
        diffs, _, slope = self._bump_terms(w)
        bump_coefs = -self.bump_depths * slope * 2.0 / self.bump_radii**2
        return grad + bump_coefs @ diffs


@dataclass(frozen=True)
class PiecewiseConvex(SyntheticFunction):
    """Rung 3 (STRUCTURAL STUB): separable convex piecewise-quadratic with planted kinks.

    Intended form (not yet implemented; see MATH.md): with rotated coordinates
    $$v = Q^\\top (w - w^*)$$ (identity when ``rotation`` is None),
    $$f(w) = \\sum_i h_i(v_i)$$ where $$h_i$$ is quadratic
    $$\\frac{1}{2} c_i v^2$$ for $$|v| \\le k_i$$ and linear with slope
    $$m_i \\ge c_i k_i$$ beyond, giving kinks at $$\\pm k_i$$ from the optimum.
    Convexity needs $$m_i \\ge c_i k_i$$ (what validation enforces); the kink is
    only genuine when the inequality is strict.
    ``kink_at_optimum`` collapses $$k_i$$ to 0 (weighted-L1 behavior).
    ``grad()`` will return the right derivative at kinks.
    """

    w_star: np.ndarray
    rotation: np.ndarray | None
    kink_offsets: np.ndarray
    inner_curvatures: np.ndarray
    outer_slopes: np.ndarray

    is_convex: ClassVar[bool] = True
    is_smooth: ClassVar[bool] = False

    def __post_init__(self) -> None:
        w_star = np.asarray(self.w_star, dtype=float)
        if w_star.ndim != 1 or w_star.size < 1:
            raise ValueError("w_star must be a 1D array with at least one element.")
        dim = w_star.size
        rotation = None if self.rotation is None else _validated_rotation(self.rotation, dim)
        kink_offsets = _validated_vector("kink_offsets", self.kink_offsets, dim)
        inner_curvatures = _validated_vector("inner_curvatures", self.inner_curvatures, dim)
        outer_slopes = _validated_vector("outer_slopes", self.outer_slopes, dim)
        if np.any(kink_offsets < 0.0):
            raise ValueError("kink_offsets must be non-negative.")
        if np.any(inner_curvatures <= 0.0):
            raise ValueError("inner_curvatures must all be positive.")
        if np.any(outer_slopes < inner_curvatures * kink_offsets):
            raise ValueError("outer_slopes must satisfy m_i >= c_i * k_i for convexity.")
        object.__setattr__(self, "w_star", w_star)
        object.__setattr__(self, "rotation", rotation)
        object.__setattr__(self, "kink_offsets", kink_offsets)
        object.__setattr__(self, "inner_curvatures", inner_curvatures)
        object.__setattr__(self, "outer_slopes", outer_slopes)

    @classmethod
    def from_seed(
        cls,
        seed: int,
        *,
        dim: int,
        kink_at_optimum: bool = False,
        rotate: bool = True,
        w_star_scale: float = 1.0,
    ) -> "PiecewiseConvex":
        """Build a seeded instance; math methods raise until the rung is implemented."""
        if dim < 1:
            raise ValueError("dim must be positive.")
        rng = default_rng(seed)
        w_star = rng.normal(size=dim) * float(w_star_scale)
        rotation = _seeded_rotation(rng, dim) if rotate else None
        kink_offsets = (
            np.zeros(dim, dtype=float) if kink_at_optimum else rng.uniform(0.5, 1.5, size=dim)
        )
        inner_curvatures = rng.uniform(0.5, 2.0, size=dim)
        slope_floor = np.maximum(inner_curvatures * kink_offsets, 0.5)
        outer_slopes = slope_floor * rng.uniform(1.5, 3.0, size=dim)
        return cls(
            w_star=w_star,
            rotation=rotation,
            kink_offsets=kink_offsets,
            inner_curvatures=inner_curvatures,
            outer_slopes=outer_slopes,
        )

    def _f(self, w: np.ndarray) -> float:
        raise NotImplementedError(
            "PiecewiseConvex is a structural stub; implement _f/_grad_f per MATH.md before use."
        )

    def _grad_f(self, w: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            "PiecewiseConvex is a structural stub; implement _f/_grad_f per MATH.md before use."
        )

    def min_kink_distance(self, theta: np.ndarray) -> float:
        raise NotImplementedError(
            "PiecewiseConvex is a structural stub; implement _f/_grad_f per MATH.md before use."
        )


@dataclass(frozen=True)
class PiecewiseNonconvexDoubleWell(SyntheticFunction):
    """Rung 4 (STRUCTURAL STUB): separable piecewise-quadratic double wells, known global minimum.

    Intended form (not yet implemented; see MATH.md): with rotated coordinates
    $$v = Q^\\top (w - w^*)$$, masked coordinates use
    $$h_i(v) = \\min\\bigl(\\frac{1}{2} c_i v^2,\\; \\frac{1}{2} d_i (v - b_i)^2 + \\delta_i\\bigr)$$
    — the decoy well at $$v = b_i$$ sits $$\\delta_i > 0$$ above the true well at
    0 — and unmasked coordinates stay purely quadratic. The global minimum is
    $$w^*$$ with value 0, provable as the sum of coordinate-wise minima; the
    min of two parabolas is nonconvex with kinks at the crossing points.
    """

    w_star: np.ndarray
    rotation: np.ndarray | None
    primary_curvatures: np.ndarray
    decoy_curvatures: np.ndarray
    well_separation: np.ndarray
    depth_margin: np.ndarray
    double_well_mask: np.ndarray

    is_convex: ClassVar[bool] = False
    is_smooth: ClassVar[bool] = False

    def __post_init__(self) -> None:
        w_star = np.asarray(self.w_star, dtype=float)
        if w_star.ndim != 1 or w_star.size < 1:
            raise ValueError("w_star must be a 1D array with at least one element.")
        dim = w_star.size
        rotation = None if self.rotation is None else _validated_rotation(self.rotation, dim)
        primary = _validated_vector("primary_curvatures", self.primary_curvatures, dim)
        decoy = _validated_vector("decoy_curvatures", self.decoy_curvatures, dim)
        separation = _validated_vector("well_separation", self.well_separation, dim)
        margin = _validated_vector("depth_margin", self.depth_margin, dim)
        mask = np.asarray(self.double_well_mask)
        if mask.shape != (dim,) or mask.dtype != np.bool_:
            raise ValueError(f"double_well_mask must be a boolean array with dimension {dim}.")
        if not bool(mask.any()):
            raise ValueError("double_well_mask must flag at least one coordinate.")
        if np.any(primary <= 0.0) or np.any(decoy <= 0.0):
            raise ValueError("well curvatures must all be positive.")
        if np.any(separation[mask] <= 0.0):
            raise ValueError("well_separation must be positive on masked coordinates.")
        if np.any(margin[mask] <= 0.0):
            raise ValueError("depth_margin must be positive on masked coordinates.")
        object.__setattr__(self, "w_star", w_star)
        object.__setattr__(self, "rotation", rotation)
        object.__setattr__(self, "primary_curvatures", primary)
        object.__setattr__(self, "decoy_curvatures", decoy)
        object.__setattr__(self, "well_separation", separation)
        object.__setattr__(self, "depth_margin", margin)
        object.__setattr__(self, "double_well_mask", mask)

    @classmethod
    def from_seed(
        cls,
        seed: int,
        *,
        dim: int,
        decoy_fraction: float = 0.5,
        depth_margin: float = 0.25,
        rotate: bool = True,
        w_star_scale: float = 1.0,
    ) -> "PiecewiseNonconvexDoubleWell":
        """Build a seeded instance; math methods raise until the rung is implemented."""
        if dim < 1:
            raise ValueError("dim must be positive.")
        if not 0.0 < decoy_fraction <= 1.0:
            raise ValueError("decoy_fraction must lie in (0, 1].")
        if depth_margin <= 0.0:
            raise ValueError("depth_margin must be positive.")
        rng = default_rng(seed)
        w_star = rng.normal(size=dim) * float(w_star_scale)
        rotation = _seeded_rotation(rng, dim) if rotate else None
        mask = rng.uniform(size=dim) < float(decoy_fraction)
        if not bool(mask.any()):
            mask[0] = True
        return cls(
            w_star=w_star,
            rotation=rotation,
            primary_curvatures=rng.uniform(0.5, 2.0, size=dim),
            decoy_curvatures=rng.uniform(0.5, 2.0, size=dim),
            well_separation=rng.uniform(2.0, 4.0, size=dim),
            depth_margin=np.full(dim, float(depth_margin)),
            double_well_mask=mask,
        )

    def _f(self, w: np.ndarray) -> float:
        raise NotImplementedError(
            "PiecewiseNonconvexDoubleWell is a structural stub; implement _f/_grad_f per MATH.md before use."
        )

    def _grad_f(self, w: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            "PiecewiseNonconvexDoubleWell is a structural stub; implement _f/_grad_f per MATH.md before use."
        )

    def min_kink_distance(self, theta: np.ndarray) -> float:
        raise NotImplementedError(
            "PiecewiseNonconvexDoubleWell is a structural stub; implement _f/_grad_f per MATH.md before use."
        )


SYNTHETIC_LADDER: dict[str, type[SyntheticFunction]] = {
    "quadratic": StronglyConvexQuadratic,
    "smoothed_nonconvex": SmoothedNonconvex,
    "piecewise_convex": PiecewiseConvex,
    "piecewise_nonconvex": PiecewiseNonconvexDoubleWell,
}

IMPLEMENTED_SYNTHETIC_LADDER: tuple[str, ...] = ("quadratic", "smoothed_nonconvex")


__all__ = [
    "IMPLEMENTED_SYNTHETIC_LADDER",
    "PiecewiseConvex",
    "PiecewiseNonconvexDoubleWell",
    "SmoothedNonconvex",
    "StronglyConvexQuadratic",
    "SYNTHETIC_LADDER",
    "SyntheticFunction",
]
