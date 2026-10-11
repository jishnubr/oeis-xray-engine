"""Bounded Bloom-filter-assisted search for numerical GCF conjectures.

This ports the behavior of Project A's candidate path: enumerate finite
left-hand linear forms, index their quantized keys in a Bloom filter, enumerate
finite right-hand continued fractions, and high-precision-refine key hits.
Bloom membership is only a prefilter; exact key-bucket lookup follows it.
The output remains a numerical candidate, never an exact relation claim.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Sequence

import mpmath

from .gcf import EfficientGCF


@dataclass(frozen=True, slots=True)
class LinearForm:
    """A rational linear form in the configured constants."""

    numerator: tuple[int, ...]
    denominator: tuple[int, ...]

    def __post_init__(self) -> None:
        _integer_tuple(self.numerator, "numerator")
        _integer_tuple(self.denominator, "denominator")
        if not self.numerator or not self.denominator:
            raise ValueError("linear-form coefficient vectors must not be empty")


@dataclass(frozen=True, slots=True)
class GCFSeries:
    """One finite pair of generalized continued-fraction term vectors."""

    a_terms: tuple[int, ...]
    b_terms: tuple[int, ...]

    def __post_init__(self) -> None:
        _integer_tuple(self.a_terms, "a_terms")
        _integer_tuple(self.b_terms, "b_terms")
        if not self.a_terms or len(self.a_terms) != len(self.b_terms):
            raise ValueError("a_terms and b_terms must be non-empty and equal length")


@dataclass(frozen=True, slots=True)
class CFSearchSpec:
    """Serializable finite search domain and numerical refinement settings."""

    constants: tuple[str, ...]
    lhs_forms: tuple[LinearForm, ...]
    rhs_series: tuple[GCFSeries, ...]
    key_scale: int
    initial_dps: int = 50
    verify_dps: int = 200
    verify_digits: int = 100
    max_pairs: int = 1_000_000
    max_results: int = 10_000

    def __post_init__(self) -> None:
        if not self.constants or not self.lhs_forms or not self.rhs_series:
            raise ValueError("constants and both candidate domains must be non-empty")
        for constant in self.constants:
            if not isinstance(constant, str) or not constant.strip():
                raise ValueError("constants must be non-empty decimal strings")
        for form in self.lhs_forms:
            if not isinstance(form, LinearForm):
                raise TypeError("lhs_forms must contain LinearForm values")
            if len(form.numerator) != len(self.constants) or len(form.denominator) != len(
                self.constants
            ):
                raise ValueError("linear-form coefficient lengths must match constants")
        if any(not isinstance(series, GCFSeries) for series in self.rhs_series):
            raise TypeError("rhs_series must contain GCFSeries values")
        if type(self.key_scale) is not int or self.key_scale <= 0:
            raise ValueError("key_scale must be a positive integer")
        for name, value in (
            ("initial_dps", self.initial_dps),
            ("verify_dps", self.verify_dps),
            ("verify_digits", self.verify_digits),
            ("max_pairs", self.max_pairs),
            ("max_results", self.max_results),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.verify_dps < self.initial_dps or self.verify_digits > self.verify_dps:
            raise ValueError("verification precision must cover requested digits")
        if len(self.lhs_forms) * len(self.rhs_series) > self.max_pairs:
            raise ValueError("search domain exceeds max_pairs")

    def to_json_value(self) -> dict[str, object]:
        """Return a deterministic JSON-compatible task payload."""

        return {
            "constants": list(self.constants),
            "lhs_forms": [
                {"numerator": list(form.numerator), "denominator": list(form.denominator)}
                for form in self.lhs_forms
            ],
            "rhs_series": [
                {"a_terms": list(series.a_terms), "b_terms": list(series.b_terms)}
                for series in self.rhs_series
            ],
            "key_scale": self.key_scale,
            "initial_dps": self.initial_dps,
            "verify_dps": self.verify_dps,
            "verify_digits": self.verify_digits,
            "max_pairs": self.max_pairs,
            "max_results": self.max_results,
        }

    @classmethod
    def from_json_value(cls, value: object) -> "CFSearchSpec":
        """Validate and reconstruct a spec from a worker task payload."""

        if not isinstance(value, dict):
            raise TypeError("CF search spec payload must be an object")
        return cls(
            constants=tuple(value["constants"]),
            lhs_forms=tuple(
                LinearForm(tuple(item["numerator"]), tuple(item["denominator"]))
                for item in value["lhs_forms"]
            ),
            rhs_series=tuple(
                GCFSeries(tuple(item["a_terms"]), tuple(item["b_terms"]))
                for item in value["rhs_series"]
            ),
            key_scale=value["key_scale"],
            initial_dps=value["initial_dps"],
            verify_dps=value["verify_dps"],
            verify_digits=value["verify_digits"],
            max_pairs=value["max_pairs"],
            max_results=value["max_results"],
        )


@dataclass(frozen=True, slots=True)
class CFSearchCandidate:
    """Numerically refined match; explicitly not an identity or proof."""

    lhs_index: int
    rhs_index: int
    lhs_key: int
    lhs_numerator: tuple[int, ...]
    lhs_denominator: tuple[int, ...]
    rhs_a_terms: tuple[int, ...]
    rhs_b_terms: tuple[int, ...]
    lhs_value: str
    rhs_value: str
    compared_digits: int


class BloomKeyIndex:
    """Small deterministic Bloom filter for integer candidate keys.

    Callers must retain and check the exact key set after a positive result;
    Bloom positives are not evidence of a match.
    """

    def __init__(self, expected_items: int, false_positive_rate: float = 0.01) -> None:
        if type(expected_items) is not int or expected_items <= 0:
            raise ValueError("expected_items must be a positive integer")
        if not 0.0 < false_positive_rate < 1.0:
            raise ValueError("false_positive_rate must be between zero and one")
        self.bit_count = max(
            8,
            math.ceil(
                -expected_items
                * math.log(false_positive_rate)
                / (math.log(2.0) ** 2)
            ),
        )
        self.hash_count = max(
            1,
            round(self.bit_count / expected_items * math.log(2.0)),
        )
        self._bits = bytearray((self.bit_count + 7) // 8)

    def add(self, key: int) -> None:
        """Insert one integer key."""

        for position in self._positions(key):
            self._bits[position >> 3] |= 1 << (position & 7)

    def __contains__(self, key: object) -> bool:
        if type(key) is not int:
            return False
        return all(
            self._bits[position >> 3] & (1 << (position & 7))
            for position in self._positions(key)
        )

    def _positions(self, key: int):
        encoded = str(key).encode("ascii")
        digest = hashlib.blake2b(encoded, digest_size=16).digest()
        first = int.from_bytes(digest[:8], "big")
        second = int.from_bytes(digest[8:], "big") | 1
        for index in range(self.hash_count):
            yield (first + index * second) % self.bit_count


def search_gcf_candidates(spec: CFSearchSpec) -> tuple[CFSearchCandidate, ...]:
    """Search a finite domain using legacy key quantization and refinement.

    Each left-hand linear form is evaluated from decimal constant literals at
    ``initial_dps`` and quantized with truncation toward zero. Right-hand GCFs
    use the source's exact integer convergent recurrence for their key. Bloom
    positives are followed by exact bucket lookup and ``mpmath.nstr`` equality
    at the requested verification precision.
    """

    if not isinstance(spec, CFSearchSpec):
        raise TypeError("spec must be a CFSearchSpec")

    lhs_buckets: dict[int, list[tuple[int, LinearForm]]] = {}
    with mpmath.workdps(spec.initial_dps):
        constants = tuple(mpmath.mpf(value) for value in spec.constants)
        for lhs_index, form in enumerate(spec.lhs_forms):
            numerator = _evaluate_form(form.numerator, constants)
            denominator = _evaluate_form(form.denominator, constants)
            if denominator == 0:
                continue
            value = numerator / denominator
            key = int(value * spec.key_scale)
            lhs_buckets.setdefault(key, []).append((lhs_index, form))

    if not lhs_buckets:
        return ()

    bloom = BloomKeyIndex(sum(map(len, lhs_buckets.values())))
    for key in lhs_buckets:
        bloom.add(key)

    matches: list[CFSearchCandidate] = []
    for rhs_index, series in enumerate(spec.rhs_series):
        convergent = EfficientGCF(
            series.a_terms,
            series.b_terms,
        )
        numerator, denominator = convergent.B, convergent.A
        key = _truncating_scaled_key(numerator, denominator, spec.key_scale)
        if key not in bloom:
            continue
        exact_bucket = lhs_buckets.get(key)
        if exact_bucket is None:
            continue
        with mpmath.workdps(spec.verify_dps):
            rhs_value = (
                mpmath.mpf(numerator) / denominator
                if denominator != 0
                else mpmath.mpf(0)
            )
            rhs_text = mpmath.nstr(rhs_value, spec.verify_digits)
            constants = tuple(mpmath.mpf(value) for value in spec.constants)
            for lhs_index, form in exact_bucket:
                numerator = _evaluate_form(form.numerator, constants)
                denominator = _evaluate_form(form.denominator, constants)
                if denominator == 0:
                    continue
                lhs_value = numerator / denominator
                lhs_text = mpmath.nstr(lhs_value, spec.verify_digits)
                if lhs_text == rhs_text:
                    if len(matches) >= spec.max_results:
                        raise ValueError(
                            "candidate count exceeds max_results; narrow the search domain"
                        )
                    matches.append(
                        CFSearchCandidate(
                            lhs_index=lhs_index,
                            rhs_index=rhs_index,
                            lhs_key=key,
                            lhs_numerator=form.numerator,
                            lhs_denominator=form.denominator,
                            rhs_a_terms=series.a_terms,
                            rhs_b_terms=series.b_terms,
                            lhs_value=lhs_text,
                            rhs_value=rhs_text,
                            compared_digits=spec.verify_digits,
                        )
                    )

    return tuple(sorted(matches, key=lambda item: (item.lhs_index, item.rhs_index)))


def _evaluate_form(coefficients: Sequence[int], constants: Sequence[mpmath.mpf]):
    return sum(
        (coefficient * constant for coefficient, constant in zip(coefficients, constants)),
        mpmath.mpf(0),
    )


def _truncating_scaled_key(numerator: int, denominator: int, scale: int) -> int:
    if denominator == 0:
        return 0
    scaled_numerator = numerator * scale
    sign = -1 if (scaled_numerator < 0) != (denominator < 0) else 1
    return sign * (abs(scaled_numerator) // abs(denominator))


def _integer_tuple(values: Sequence[int], name: str) -> None:
    if not isinstance(values, tuple):
        raise TypeError(f"{name} must be a tuple")
    if any(type(value) is not int for value in values):
        raise TypeError(f"{name} must contain only integers")
