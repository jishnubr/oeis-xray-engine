"""Exact, degree-truncated univariate plethystic logarithm."""

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction

from . import TransformError
from .mobius import _mobius_values

ExactNumber = int | Fraction


def plethystic_log(
    coefficients: Sequence[ExactNumber],
    degree: int,
) -> list[Fraction]:
    """Compute ``sum(mu(k)/k * log(F(x**k)))`` through degree ``degree``.

    ``coefficients[n]`` is the coefficient of ``x**n`` in ``F``. The constant
    coefficient must be one. The output has exactly ``degree + 1`` entries,
    indexed from zero through the requested degree, and every entry is a
    :class:`fractions.Fraction`.

    The formal logarithm is computed by its exact coefficient recurrence. The
    identity ``log(F(x**k)) = log(F)(x**k)`` then permits exact index skipping
    without constructing substituted series or using numerical evaluation.
    """

    if type(degree) is not int or degree < 0:
        raise TransformError("degree must be a non-negative integer")
    if isinstance(coefficients, (str, bytes, bytearray)) or not isinstance(
        coefficients, Sequence
    ):
        raise TransformError("coefficients must be a finite exact-number sequence")
    if len(coefficients) <= degree:
        raise TransformError(
            "insufficient coefficients for requested degree",
            context={"degree": degree, "available": len(coefficients) - 1},
        )

    values: list[Fraction] = []
    for index, value in enumerate(coefficients[: degree + 1]):
        if type(value) is int:
            values.append(Fraction(value))
        elif isinstance(value, Fraction):
            values.append(value)
        else:
            raise TransformError(
                f"coefficients[{index}] must be an int or Fraction",
                context={"actual_type": type(value).__name__},
            )
    if values[0] != 1:
        raise TransformError(
            "plethystic logarithm requires constant coefficient F(0) = 1",
            context={"constant_term": str(values[0])},
        )

    # log(F)(x) = sum(logarithm[n] * x**n). Since F(0)=1, for n>=1:
    # n*logarithm[n] = n*F[n] - sum(k*logarithm[k]*F[n-k]).
    logarithm = [Fraction(0) for _ in range(degree + 1)]
    for n in range(1, degree + 1):
        numerator = n * values[n]
        numerator -= sum(
            (
                k * logarithm[k] * values[n - k]
                for k in range(1, n)
            ),
            Fraction(0),
        )
        logarithm[n] = numerator / n

    mu = _mobius_values(degree)
    result = [Fraction(0) for _ in range(degree + 1)]
    for divisor in range(1, degree + 1):
        if mu[divisor] == 0:
            continue
        scale = Fraction(mu[divisor], divisor)
        for multiple in range(divisor, degree + 1, divisor):
            result[multiple] += scale * logarithm[multiple // divisor]
    return result
