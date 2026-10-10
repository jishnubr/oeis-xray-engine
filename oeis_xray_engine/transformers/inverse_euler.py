"""Exact Euler transform and discrete divisor-kernel inversion.

The convention is

``A(x) = product_{k >= 1} (1 - x**k)**(-c_k)``.

``euler_transform`` accepts ``c_1, ..., c_N`` and returns ``a_0, ..., a_N``.
``inverse_euler_transform`` accepts ``a_0, ..., a_N`` with ``a_0 = 1`` and
returns the integral exponents ``c_1, ..., c_N``.
"""

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction

from ..models import TermSeries
from . import TransformError
from .mobius import _mobius_values

ExactNumber = int | Fraction


class _FractionValues(Sequence[Fraction]):
    """Validate exact values while retaining a sequence or term-series view."""

    def __init__(
        self,
        values: Sequence[ExactNumber] | TermSeries,
        name: str,
    ) -> None:
        if isinstance(values, (str, bytes, bytearray)) or not (
            isinstance(values, Sequence) or isinstance(values, TermSeries)
        ):
            raise TransformError(f"{name} must be a finite exact-number sequence")
        self._values = values
        self._name = name
        self._length = len(values)
        for index in range(self._length):
            value = self._raw_value(index)
            if type(value) is not int and not isinstance(value, Fraction):
                raise TransformError(
                    f"{name}[{index}] must be an int or Fraction",
                    context={"actual_type": type(value).__name__},
                )

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> Fraction:
        if type(index) is not int:
            raise TypeError("sequence indices must be integers")
        if index < 0:
            index += self._length
        if not 0 <= index < self._length:
            raise IndexError("sequence index out of range")
        value = self._raw_value(index)
        if value is None:
            raise TransformError(
                f"{self._name}[{index}] is missing from the term series"
            )
        return Fraction(value) if type(value) is int else value

    def _raw_value(self, position: int) -> ExactNumber | None:
        if isinstance(self._values, TermSeries):
            return self._values.value_at(self._values.start_index + position)
        return self._values[position]


def _as_fraction_sequence(
    values: Sequence[ExactNumber] | TermSeries,
    name: str,
) -> Sequence[Fraction]:
    return _FractionValues(values, name)


def _resolve_degree(degree: int | None, available: int, name: str) -> int:
    if degree is None:
        return available
    if type(degree) is not int or degree < 0:
        raise TransformError(f"{name} must be a non-negative integer")
    if degree > available:
        raise TransformError(
            f"insufficient input terms for requested {name}",
            context={"requested": degree, "available": available},
        )
    return degree


def euler_transform(
    exponents: Sequence[ExactNumber] | TermSeries,
    degree: int | None = None,
) -> list[int | Fraction]:
    """Expand a finite Euler product through ``x**degree`` exactly.

    ``exponents[i - 1]`` is ``c_i``. If ``degree`` is omitted, all supplied
    exponents are used. For an explicit degree, at least that many exponents
    must be provided; omitted higher exponents are not silently assumed zero.
    Integer output coefficients are returned as ``int``; non-integral exact
    coefficients are returned as :class:`fractions.Fraction`.
    """

    c = _as_fraction_sequence(exponents, "exponents")
    degree = _resolve_degree(degree, len(c), "degree")

    coefficients = [Fraction(0) for _ in range(degree + 1)]
    coefficients[0] = Fraction(1)
    if degree == 0:
        return [1]

    # b_n = sum_{d|n} d*c_d is the coefficient of x^n in
    # x*A'(x)/A(x). Then n*a_n = sum_{k=1}^n b_k*a_{n-k}.
    kernel = [Fraction(0) for _ in range(degree + 1)]
    for index in range(1, degree + 1):
        exponent = c[index - 1]
        for multiple in range(index, degree + 1, index):
            kernel[multiple] += index * exponent

    for n in range(1, degree + 1):
        numerator = sum(
            (kernel[k] * coefficients[n - k] for k in range(1, n + 1)),
            Fraction(0),
        )
        coefficients[n] = numerator / n

    return [_compact(value) for value in coefficients]


def inverse_euler_transform(
    coefficients: Sequence[ExactNumber] | TermSeries,
) -> list[int]:
    """Recover integral Euler exponents using the exact divisor-kernel method.

    The input uses ordinary power-series indexing ``[a_0, ..., a_N]`` and must
    have ``a_0 = 1``. For each ``n >= 1`` this computes exactly:

    ``L(n) = n*a_n - sum(L(k)*a_(n-k), k=1..n-1)``

    followed by Möbius inversion:

    ``c_n = (1/n) * sum(mu(d)*L(n/d), d|n)``.

    A non-integral ``c_n`` raises :class:`TransformError`; no floor division
    or rounding is used.
    """

    a = _as_fraction_sequence(coefficients, "coefficients")
    if not a:
        raise TransformError("coefficients must include the constant term a_0")
    if a[0] != 1:
        raise TransformError(
            "Euler inversion requires a_0 = 1",
            context={"a_0": str(a[0])},
        )

    degree = len(a) - 1
    mu = _mobius_values(degree)
    logarithmic_kernel = [Fraction(0) for _ in range(degree + 1)]

    for n in range(1, degree + 1):
        value = n * a[n]
        value -= sum(
            (
                logarithmic_kernel[k] * a[n - k]
                for k in range(1, n)
            ),
            Fraction(0),
        )
        logarithmic_kernel[n] = value

    numerators = [Fraction(0) for _ in range(degree + 1)]
    for divisor in range(1, degree + 1):
        mobius_value = mu[divisor]
        if mobius_value == 0:
            continue
        for multiple in range(divisor, degree + 1, divisor):
            numerators[multiple] += (
                mobius_value * logarithmic_kernel[multiple // divisor]
            )

    exponents: list[int] = []
    for n in range(1, degree + 1):
        exponent = numerators[n] / n
        if exponent.denominator != 1:
            raise TransformError(
                "inverse Euler transform produced a non-integral exponent",
                context={"n": n, "exponent": str(exponent)},
            )
        exponents.append(exponent.numerator)
    return exponents


def _compact(value: Fraction) -> int | Fraction:
    return value.numerator if value.denominator == 1 else value
