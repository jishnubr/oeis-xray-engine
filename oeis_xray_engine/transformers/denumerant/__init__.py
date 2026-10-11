"""Exact extraction of bosonic Euler-product weights.

The extraction follows OEIS Deep Miner's reciprocal-series, logarithmic
derivative, and Möbius-inversion pipeline. It accepts a finite dense sequence
window and returns a finite multiset of positive integer weights. It performs
no file, network, database, or catalog I/O.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar

from ...models import TermSeries
from .. import TransformError
from ..contracts import (
    TransformerContract,
    TransformerExecutionContext,
    TransformerResult,
)

_DEFAULT_TERM_CAP = 100
_DEFAULT_CONFIDENCE_WINDOW = 5
_DEFAULT_INPUT_BIT_LIMIT = 2_000
_DEFAULT_LOOP_BIT_LIMIT = 10_000
_DEFAULT_MULTIPLICITY_LIMIT = 100_000
_DEFAULT_TOTAL_WEIGHT_LIMIT = 100_000


class DenumerantError(TransformError):
    """Input is not certified as a finite pure-bosonic denumerant prefix."""


@dataclass(frozen=True, slots=True)
class DenumerantExtraction:
    """Exact extracted Euler-product multiplicities and validation metadata.

    ``weights`` stores each positive weight once per multiplicity, matching the
    legacy Deep Miner return convention. ``source_unit_index`` is the explicit
    source index of the first coefficient equal to one.
    """

    weights: tuple[int, ...]
    exponents: tuple[int, ...]
    maximum_weight: int
    source_unit_index: int
    confidence_terms: int
    reconstructed_coefficients: tuple[int, ...]


def extract_denominations(
    values: Sequence[int],
    *,
    source_start_index: int = 0,
    term_cap: int = _DEFAULT_TERM_CAP,
    confidence_window: int = _DEFAULT_CONFIDENCE_WINDOW,
    input_bit_limit: int = _DEFAULT_INPUT_BIT_LIMIT,
    loop_bit_limit: int = _DEFAULT_LOOP_BIT_LIMIT,
    multiplicity_limit: int = _DEFAULT_MULTIPLICITY_LIMIT,
    total_weight_limit: int = _DEFAULT_TOTAL_WEIGHT_LIMIT,
) -> DenumerantExtraction:
    """Extract exact positive Euler-product weights from a finite prefix.

    The source convention begins at the first observed coefficient equal to
    one. It then computes the reciprocal coefficients ``b`` and requires a
    trailing-zero confidence window after the final nonzero reciprocal
    coefficient before interpreting that index as the extraction degree.
    Möbius inversion must yield nonnegative integer exponents. The resulting
    finite product is expanded exactly and must reproduce every input
    coefficient through the extraction degree.

    The configured limits intentionally bound source-compatible work and the
    expanded weight list. A limit rejection is not evidence that the input is
    mathematically invalid; it means this implementation did not certify it.
    """

    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise DenumerantError("values must be a finite integer sequence")
    if type(source_start_index) is not int:
        raise DenumerantError("source_start_index must be an integer")
    _positive_limit(term_cap, "term_cap")
    _nonnegative_limit(confidence_window, "confidence_window")
    _positive_limit(input_bit_limit, "input_bit_limit")
    _positive_limit(loop_bit_limit, "loop_bit_limit")
    _positive_limit(multiplicity_limit, "multiplicity_limit")
    _positive_limit(total_weight_limit, "total_weight_limit")

    source_values = tuple(values)
    if any(type(value) is not int for value in source_values):
        raise DenumerantError("values must contain exact Python integers")
    first_one = next(
        (position for position, value in enumerate(source_values) if value == 1),
        None,
    )
    if first_one is None:
        raise DenumerantError("no coefficient equal to one is available")

    suffix = source_values[first_one:]
    if len(suffix) < 15:
        raise DenumerantError(
            "at least 15 terms from the first unit coefficient are required",
            context={"available": len(suffix), "required": 15},
        )
    if any(abs(value).bit_length() > input_bit_limit for value in suffix[:15]):
        raise DenumerantError(
            "input coefficient exceeds the configured bit limit",
            context={"bit_limit": input_bit_limit},
        )

    degree_count = min(len(suffix), term_cap)
    reciprocal = [0] * degree_count
    reciprocal[0] = 1
    for degree in range(1, degree_count):
        value = -sum(
            suffix[index] * reciprocal[degree - index]
            for index in range(1, degree + 1)
        )
        if abs(value).bit_length() > loop_bit_limit:
            raise DenumerantError(
                "reciprocal coefficient exceeds the configured bit limit",
                context={"degree": degree, "bit_limit": loop_bit_limit},
            )
        reciprocal[degree] = value

    nonzero_indices = [
        index for index, coefficient in enumerate(reciprocal) if coefficient
    ]
    if not nonzero_indices:
        raise DenumerantError("reciprocal series has no nonzero coefficients")
    maximum_weight = nonzero_indices[-1]
    confidence_terms = degree_count - 1 - maximum_weight
    if maximum_weight < 1 or confidence_terms < confidence_window:
        raise DenumerantError(
            "reciprocal series does not meet the confidence-window requirement",
            context={
                "maximum_weight": maximum_weight,
                "confidence_terms": confidence_terms,
                "required_confidence_terms": confidence_window,
            },
        )

    logarithmic_kernel = [0] * (maximum_weight + 1)
    for degree in range(1, maximum_weight + 1):
        logarithmic_kernel[degree] = -degree * reciprocal[degree] - sum(
            reciprocal[degree - divisor] * logarithmic_kernel[divisor]
            for divisor in range(1, degree)
        )

    mobius = _mobius_values(maximum_weight)
    exponents: list[int] = []
    weights: list[int] = []
    for weight in range(1, maximum_weight + 1):
        numerator = sum(
            mobius[weight // divisor] * logarithmic_kernel[divisor]
            for divisor in _divisors(weight)
        )
        exponent, remainder = divmod(numerator, weight)
        if remainder:
            raise DenumerantError(
                "Möbius inversion produced a non-integral multiplicity",
                context={"weight": weight, "numerator": numerator},
            )
        if exponent < 0:
            raise DenumerantError(
                "negative Euler multiplicity is outside the pure-bosonic domain",
                context={"weight": weight, "multiplicity": exponent},
            )
        if exponent > multiplicity_limit:
            raise DenumerantError(
                "Euler multiplicity exceeds the configured limit",
                context={"weight": weight, "limit": multiplicity_limit},
            )
        if len(weights) + exponent > total_weight_limit:
            raise DenumerantError(
                "expanded weight list exceeds the configured total limit",
                context={"limit": total_weight_limit},
            )
        exponents.append(exponent)
        weights.extend([weight] * exponent)

    reconstructed = _reconstruct(exponents)
    expected = suffix[: maximum_weight + 1]
    if tuple(reconstructed) != expected:
        mismatch = next(
            index
            for index, (actual, wanted) in enumerate(zip(reconstructed, expected))
            if actual != wanted
        )
        raise DenumerantError(
            "extracted Euler product failed exact forward reconstruction",
            context={
                "degree": mismatch,
                "reconstructed": reconstructed[mismatch],
                "expected": expected[mismatch],
            },
        )

    return DenumerantExtraction(
        weights=tuple(weights),
        exponents=tuple(exponents),
        maximum_weight=maximum_weight,
        source_unit_index=source_start_index + first_one,
        confidence_terms=confidence_terms,
        reconstructed_coefficients=tuple(reconstructed),
    )


def reconstruct_denominations(
    weights: Sequence[int],
    term_count: int,
) -> tuple[int, ...]:
    """Expand a finite positive-weight multiset into exact coefficients.

    The output begins at generating-function degree zero. This is used by
    external validators that need an independently reconstructed finite
    sequence from extracted weights.
    """

    if isinstance(weights, (str, bytes, bytearray)) or not isinstance(
        weights, Sequence
    ):
        raise DenumerantError("weights must be a finite integer sequence")
    if type(term_count) is not int or term_count <= 0:
        raise DenumerantError("term_count must be a positive integer")
    multiplicities: dict[int, int] = {}
    for position, weight in enumerate(weights):
        if type(weight) is not int or weight <= 0:
            raise DenumerantError(
                "weights must contain positive integers",
                context={"position": position},
            )
        multiplicities[weight] = multiplicities.get(weight, 0) + 1

    kernel = [0] * term_count
    for weight, multiplicity in multiplicities.items():
        for multiple in range(weight, term_count, weight):
            kernel[multiple] += weight * multiplicity

    coefficients = [0] * term_count
    coefficients[0] = 1
    for degree in range(1, term_count):
        numerator = sum(
            kernel[part] * coefficients[degree - part]
            for part in range(1, degree + 1)
        )
        coefficient, remainder = divmod(numerator, degree)
        if remainder:
            raise DenumerantError(
                "forward Euler reconstruction produced a non-integral coefficient",
                context={"degree": degree, "numerator": numerator},
            )
        coefficients[degree] = coefficient
    return tuple(coefficients)


class DenumerantTransformer(TransformerContract):
    """Pipeline adapter for pure-bosonic denumerant extraction."""

    name: ClassVar[str] = "denumerant"
    version: ClassVar[str] = "1"

    def transform(
        self,
        context: TransformerExecutionContext,
    ) -> TransformerResult:
        self.validate(context)
        series: TermSeries = context.input_series
        if series.indices is not None:
            expected_indices = tuple(
                range(series.start_index, series.start_index + len(series.values))
            )
            if series.indices != expected_indices:
                raise DenumerantError(
                    "denumerant extraction requires a dense input window"
                )

        parameters = context.spec.parameters
        extraction = extract_denominations(
            series.values,
            source_start_index=series.start_index,
            term_cap=_int_parameter(parameters, "term_cap", _DEFAULT_TERM_CAP),
            confidence_window=_int_parameter(
                parameters, "confidence_window", _DEFAULT_CONFIDENCE_WINDOW
            ),
            input_bit_limit=_int_parameter(
                parameters, "input_bit_limit", _DEFAULT_INPUT_BIT_LIMIT
            ),
            loop_bit_limit=_int_parameter(
                parameters, "loop_bit_limit", _DEFAULT_LOOP_BIT_LIMIT
            ),
            multiplicity_limit=_int_parameter(
                parameters, "multiplicity_limit", _DEFAULT_MULTIPLICITY_LIMIT
            ),
            total_weight_limit=_int_parameter(
                parameters, "total_weight_limit", _DEFAULT_TOTAL_WEIGHT_LIMIT
            ),
        )
        output_start = context.metadata.get("output_start_index", 1)
        if type(output_start) is not int:
            raise DenumerantError("output_start_index metadata must be an integer")
        requested_count = context.metadata.get("output_term_count")
        if requested_count is not None:
            if type(requested_count) is not int or requested_count != len(
                extraction.weights
            ):
                raise DenumerantError(
                    "requested output term count differs from extracted weight count",
                    context={
                        "requested": requested_count,
                        "actual": len(extraction.weights),
                    },
                )
        return TransformerResult(
            start_index=output_start,
            values=extraction.weights,
            metadata={
                "source_unit_index": extraction.source_unit_index,
                "maximum_weight": extraction.maximum_weight,
                "confidence_terms": extraction.confidence_terms,
                "distinct_weight_count": len(extraction.exponents),
            },
        )


def _int_parameter(parameters, name: str, default: int) -> int:
    value = parameters.get(name, default)
    if type(value) is not int:
        raise DenumerantError(
            f"transform parameter {name!r} must be an integer",
            context={"value": value},
        )
    return value


def _positive_limit(value: int, name: str) -> None:
    if type(value) is not int or value <= 0:
        raise DenumerantError(f"{name} must be a positive integer")


def _nonnegative_limit(value: int, name: str) -> None:
    if type(value) is not int or value < 0:
        raise DenumerantError(f"{name} must be a non-negative integer")


def _divisors(number: int) -> tuple[int, ...]:
    lower: list[int] = []
    upper: list[int] = []
    divisor = 1
    while divisor * divisor <= number:
        if number % divisor == 0:
            lower.append(divisor)
            if divisor * divisor != number:
                upper.append(number // divisor)
        divisor += 1
    return tuple(lower + upper[::-1])


def _mobius_values(limit: int) -> tuple[int, ...]:
    """Return μ(0..limit) using a bounded linear sieve."""

    values = [0] * (limit + 1)
    if limit == 0:
        return tuple(values)
    values[1] = 1
    primes: list[int] = []
    composite = [False] * (limit + 1)
    for number in range(2, limit + 1):
        if not composite[number]:
            primes.append(number)
            values[number] = -1
        for prime in primes:
            product = number * prime
            if product > limit:
                break
            composite[product] = True
            if number % prime == 0:
                values[product] = 0
                break
            values[product] = -values[number]
    return tuple(values)


def _reconstruct(exponents: Sequence[int]) -> list[int]:
    """Expand the finite product exactly through its declared degree."""

    degree = len(exponents)
    kernel = [0] * (degree + 1)
    for weight, multiplicity in enumerate(exponents, start=1):
        if multiplicity:
            for multiple in range(weight, degree + 1, weight):
                kernel[multiple] += weight * multiplicity

    coefficients = [0] * (degree + 1)
    coefficients[0] = 1
    for index in range(1, degree + 1):
        numerator = sum(
            kernel[part] * coefficients[index - part]
            for part in range(1, index + 1)
        )
        coefficient, remainder = divmod(numerator, index)
        if remainder:
            raise DenumerantError(
                "forward Euler reconstruction produced a non-integral coefficient",
                context={"degree": index, "numerator": numerator},
            )
        coefficients[index] = coefficient
    return coefficients


__all__ = [
    "DenumerantError",
    "DenumerantExtraction",
    "DenumerantTransformer",
    "extract_denominations",
    "reconstruct_denominations",
]
