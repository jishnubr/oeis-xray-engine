"""Exact generalized continued-fraction arithmetic used by the legacy search.

The matrix path retains NumPy's object-array matrix multiplication used by
Project A. The efficient convergent path retains the same integer recurrence;
callers can request either Project A-compatible ``mpmath`` output or an exact
:class:`fractions.Fraction`.
"""

from __future__ import annotations

from fractions import Fraction
from math import gcd
from typing import Sequence

import mpmath
import numpy as np


class MobiusTransform:
    """Integer Möbius transform represented by a 2-by-2 object array.

    The matrix ``[[a, b], [c, d]]`` represents ``(a*x+b)/(c*x+d)``.
    Composition follows Project A's right-multiplication convention.
    """

    def __init__(self, data: np.ndarray | Sequence[Sequence[int]] | None = None):
        matrix = np.eye(2, dtype=object) if data is None else np.asarray(data, dtype=object)
        if matrix.shape != (2, 2):
            raise ValueError("Möbius transform data must have shape (2, 2)")
        if any(type(value) is not int for value in matrix.flat):
            raise TypeError("Möbius transform entries must be integers")
        self.data = matrix.copy()
        self.normalize()

    def normalize(self) -> None:
        """Divide all matrix entries by their common positive gcd."""

        common = 0
        for value in self.data.flat:
            common = gcd(common, value)
        if common > 1:
            self.data //= common

    def __mul__(self, other: "MobiusTransform") -> "MobiusTransform":
        """Compose transforms by exact NumPy object-array multiplication."""

        if not isinstance(other, MobiusTransform):
            return NotImplemented
        return MobiusTransform(np.matmul(self.data, other.data))

    def reciprocal(self) -> "MobiusTransform":
        """Return the transform representing the reciprocal of this value."""

        a, b, c, d = (int(value) for value in self.data.flat)
        return MobiusTransform(((c, d), (a, b)))

    def inverse(self) -> "MobiusTransform":
        """Return the projectively equivalent inverse transformation."""

        a, b, c, d = (int(value) for value in self.data.flat)
        return MobiusTransform(((d, -b), (-c, a)))

    def evaluate_at_infinity(self) -> Fraction:
        """Evaluate at the legacy ``None`` operand, i.e. return ``b / d``."""

        denominator = int(self.data[1, 1])
        if denominator == 0:
            raise ZeroDivisionError("Möbius transform has a pole at infinity")
        return Fraction(int(self.data[0, 1]), denominator)


class GeneralizedContinuedFraction:
    """Incrementally accumulated generalized continued fraction.

    As in Project A, each newly appended ``a[i]`` is paired with ``b[i-1]``.
    The finite convergent is exact and represented as a rational number.
    """

    def __init__(
        self,
        a_terms: Sequence[int] | None = None,
        b_terms: Sequence[int] | None = None,
    ) -> None:
        self.mobius = MobiusTransform()
        self.a_terms: list[int] = []
        self.b_terms: list[int] = []
        if a_terms is not None or b_terms is not None:
            if a_terms is None or b_terms is None:
                raise ValueError("a_terms and b_terms must be supplied together")
            self.extend(a_terms, b_terms)

    def extend(self, a_terms: Sequence[int], b_terms: Sequence[int]) -> None:
        """Append terms using Project A's matrix-convergent convention."""

        _validate_integer_terms(a_terms, "a_terms")
        _validate_integer_terms(b_terms, "b_terms")
        self.a_terms.extend(a_terms)
        self.b_terms.extend(b_terms)
        for index in range(min(len(a_terms) - 1, len(b_terms))):
            self.mobius = self.mobius * MobiusTransform(
                ((0, b_terms[index]), (1, a_terms[index + 1]))
            )

    def evaluate(self) -> Fraction:
        """Return the accumulated finite convergent exactly."""

        if not self.a_terms:
            raise ValueError("continued fraction has no a terms")
        return Fraction(self.a_terms[0]) + self.mobius.evaluate_at_infinity()


class EfficientGCF:
    """Project A-compatible arbitrary-precision convergent evaluator."""

    def __init__(self, a_terms: Sequence[int], b_terms: Sequence[int]) -> None:
        if not a_terms or len(a_terms) != len(b_terms):
            raise ValueError("a_terms and b_terms must be non-empty and equal length")
        self.prev_A = 0
        self.A = 1
        self.prev_B = 1
        self.B = a_terms[0]
        for index in range(1, len(a_terms)):
            previous_a = self.A
            previous_b = self.B
            self.A = a_terms[index] * self.A + b_terms[index] * self.prev_A
            self.B = a_terms[index] * self.B + b_terms[index] * self.prev_B
            self.prev_A = previous_a
            self.prev_B = previous_b

    def evaluate(self) -> mpmath.mpf:
        """Return the legacy arbitrary-precision numerical convergent."""

        if self.A == 0:
            return mpmath.mpf(0)
        return mpmath.mpf(self.B) / mpmath.mpf(self.A)


def efficient_gcf_fraction(
    a_terms: Sequence[int],
    b_terms: Sequence[int],
) -> Fraction:
    """Compute a finite GCF using Project A's numerator/denominator recurrence.

    The first ``b_terms`` value is intentionally not used: that is the index
    alignment in the original efficient enumerator and is therefore preserved.
    A zero denominator follows the source implementation and yields zero.
    """

    numerator, denominator = efficient_gcf_numerator_denominator(
        a_terms,
        b_terms,
    )
    return Fraction(0) if denominator == 0 else Fraction(numerator, denominator)


def efficient_gcf_numerator_denominator(
    a_terms: Sequence[int],
    b_terms: Sequence[int],
) -> tuple[int, int]:
    """Return the raw integer convergent pair without rational normalization.

    Project A's key-generation loop uses this recurrence directly. Keeping the
    unreduced pair avoids a gcd operation in each inner-loop candidate.
    """

    state = _efficient_gcf_state(a_terms, b_terms)
    return state[3], state[1]


def _efficient_gcf_state(
    a_terms: Sequence[int],
    b_terms: Sequence[int],
) -> tuple[int, int, int, int]:
    _validate_integer_terms(a_terms, "a_terms")
    _validate_integer_terms(b_terms, "b_terms")
    if not a_terms:
        raise ValueError("a_terms must not be empty")
    if len(a_terms) != len(b_terms):
        raise ValueError("a_terms and b_terms must have equal lengths")

    previous_denominator = 0
    denominator = 1
    previous_numerator = 1
    numerator = a_terms[0]
    for index in range(1, len(a_terms)):
        old_denominator = denominator
        old_numerator = numerator
        denominator = (
            a_terms[index] * denominator
            + b_terms[index] * previous_denominator
        )
        numerator = (
            a_terms[index] * numerator
            + b_terms[index] * previous_numerator
        )
        previous_denominator = old_denominator
        previous_numerator = old_numerator
    return previous_denominator, denominator, previous_numerator, numerator


def _validate_integer_terms(terms: Sequence[int], name: str) -> None:
    if not isinstance(terms, Sequence) or isinstance(terms, (str, bytes)):
        raise TypeError(f"{name} must be an integer sequence")
    if any(type(value) is not int for value in terms):
        raise TypeError(f"{name} must contain only integers")
