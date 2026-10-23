"""Composition of a saved band endpoint with the repository policy/optimizer interface."""

from __future__ import annotations

import numpy as np

from .base import Objective


class BandObjective(Objective):
    """Optimize an explicit endpoint with preserved batch scaling and optional acceptance constraints."""

    def __init__(self, band, policy, *, endpoint, sense, artifact, acceptance_floor=None):
        if endpoint not in {"center", "lower", "upper"} or sense not in {"minimize", "maximize"}:
            raise ValueError("Declare endpoint=center/lower/upper and sense=minimize/maximize.")
        self.band, self.policy = band, policy
        self.endpoint, self.sense, self.artifact = endpoint, sense, dict(artifact)
        self.sign = 1. if sense == "minimize" else -1.
        self.acceptance_floor = acceptance_floor
        self._context_dim = band.state["X"].shape[1]
        self._contexts = {tuple(row) for row in band.state["X"]}
        if "acceptance_grid" in band.state:
            self.mean_acceptance = self._acceptance
            self.mean_acceptance_grad = self._acceptance_grad
            self.mean_acceptance_at_u = self._acceptance_at_u

    def _inputs(self, x_batch):
        X = np.asarray(x_batch, dtype=float)
        if self._context_dim == 0:
            return np.empty((len(X), 0))
        if X.ndim != 2 or X.shape[1] != self._context_dim or any(tuple(row) not in self._contexts for row in X):
            raise ValueError("Optimization rows must belong to the fitted band's declared context domain.")
        return X

    def _value_batch(self, x_batch, actions):
        evaluated = self.band.evaluate(self._inputs(x_batch), np.asarray(actions))
        return self.sign * getattr(evaluated, self.endpoint)

    def value(self, theta, x_batch):
        """Return the mean signed endpoint, using the existing policy action convention."""
        return float(np.mean(self._value_batch(x_batch, self.policy.value(theta, x_batch))))

    def value_at_u(self, x_batch, u):
        """Evaluate a fixed action without fitting or selecting a solution."""
        return float(np.mean(self._value_batch(x_batch, np.full(len(x_batch), u))))

    def grad(self, theta, x_batch):
        """Apply the policy chain rule to the declared endpoint action derivative."""
        actions = self.policy.value(theta, x_batch)
        slopes = self.sign*self.band.derivative(self._inputs(x_batch), actions, self.endpoint)
        return self.policy.weighted_grad(theta, x_batch, slopes)/len(x_batch)

    def _acceptance(self, theta, x_batch):
        actions = self.policy.value(theta, x_batch)
        return float(np.mean(self.band.evaluate(self._inputs(x_batch), actions).components["acceptance"]))

    def _acceptance_at_u(self, x_batch, u):
        return float(np.mean(self.band.evaluate(self._inputs(x_batch), np.full(len(x_batch), u)).components["acceptance"]))

    def _acceptance_grad(self, theta, x_batch):
        actions = self.policy.value(theta, x_batch)
        slopes = self.band.evaluate(self._inputs(x_batch), actions).components["acceptance_derivative"]
        return self.policy.weighted_grad(theta, x_batch, slopes)/len(x_batch)

    def to_dict(self):
        """Record the selected target and exact fitted-band artifact reference."""
        return {"type": "BandObjective", "endpoint": self.endpoint, "sense": self.sense,
                "band": self.artifact, "acceptance_floor": self.acceptance_floor}


__all__ = ["BandObjective"]
