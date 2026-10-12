"""Exact finite enumeration checks for the candidate proof's count moments.

This supplements, and does not replace, the general mathematical proof or Lean.
It enumerates probabilities; it does not simulate data or run an optimizer.
"""

from fractions import Fraction as F
from math import factorial


def check(n):
    moments = {"second": F(0), "cross": F(0), "fourth": F(0), "square_cross": F(0)}
    total = F(0)
    for a in range(n + 1):
        for b in range(n - a + 1):
            c = n - a - b
            probability = F(
                factorial(n) * (n - 2) ** c,
                factorial(a) * factorial(b) * factorial(c) * n**n,
            )
            x, y = a - 1, b - 1
            total += probability
            moments["second"] += probability * x**2
            moments["cross"] += probability * x * y
            moments["fourth"] += probability * x**4
            moments["square_cross"] += probability * x**2 * y**2
    assert total == 1
    v = 1 - F(1, n)
    assert moments["second"] == v
    assert moments["cross"] == -F(1, n)
    assert moments["fourth"] - v**2 == 3 - F(11, n) + F(14, n**2) - F(6, n**3)
    assert moments["square_cross"] - v**2 == -F(1, n) + F(6, n**2) - F(6, n**3)


if __name__ == "__main__":
    for sample_size in range(3, 31):
        check(sample_size)
    print("Exact count-moment checks passed for every N from 3 through 30.")
