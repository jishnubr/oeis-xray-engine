"""Exact roots-of-unity residue filtering by coefficient multisection."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction

from . import TransformError

ExactNumber = int | Fraction


@dataclass(frozen=True, slots=True)
class FilteredSeries:
    """A residue-class subsequence with its original index map.

    ``values[i]`` came from original index ``original_indices[i]``. ``start_index``
    is the first retained original index, or ``None`` when the slice is empty.
    ``source_start_index`` and ``source_end_index`` describe the full input
    range; an empty input has ``source_end_index = source_start_index - 1``.
    """

    values: tuple[ExactNumber, ...]
    original_indices: tuple[int, ...]
    modulus: int
    residue: int
    start_index: int | None
    source_start_index: int
    source_end_index: int

    def __post_init__(self) -> None:
        if len(self.values) != len(self.original_indices):
            raise ValueError("values and original_indices must have equal lengths")
        if self.original_indices:
            if self.start_index != self.original_indices[0]:
                raise ValueError("start_index must be the first retained index")
            if any(
                right - left != self.modulus
                for left, right in zip(
                    self.original_indices, self.original_indices[1:]
                )
            ):
                raise ValueError("retained indices must follow the modulus stride")
        elif self.start_index is not None:
            raise ValueError("start_index must be None for an empty filtered series")


def filter_residue_class(
    coefficients: Sequence[ExactNumber],
    modulus: int,
    residue: int,
    start_index: int = 0,
) -> FilteredSeries:
    """Select coefficients whose original index is ``residue (mod modulus)``.

    This is the exact algebraic multisection equivalent of a roots-of-unity
    filter, implemented by slicing/striding only. It performs no complex or
    floating-point arithmetic. ``residue`` must already be normalized to the
    interval ``0 <= residue < modulus``.
    """

    if isinstance(coefficients, (str, bytes, bytearray)) or not isinstance(
        coefficients, Sequence
    ):
        raise TransformError("coefficients must be a finite exact-number sequence")
    if type(modulus) is not int or modulus <= 0:
        raise TransformError("modulus must be a positive integer")
    if type(residue) is not int or not 0 <= residue < modulus:
        raise TransformError(
            "residue must satisfy 0 <= residue < modulus",
            context={"modulus": modulus, "residue": residue},
        )
    if type(start_index) is not int:
        raise TransformError("start_index must be an integer")

    values = tuple(coefficients)
    for index, value in enumerate(values):
        if type(value) is not int and not isinstance(value, Fraction):
            raise TransformError(
                f"coefficients[{index}] must be an int or Fraction",
                context={"actual_type": type(value).__name__},
            )

    source_end = start_index + len(values) - 1
    first_position = (residue - start_index) % modulus
    selected_values = values[first_position::modulus]
    selected_indices = tuple(
        start_index + first_position + offset * modulus
        for offset in range(len(selected_values))
    )

    return FilteredSeries(
        values=selected_values,
        original_indices=selected_indices,
        modulus=modulus,
        residue=residue,
        start_index=selected_indices[0] if selected_indices else None,
        source_start_index=start_index,
        source_end_index=source_end,
    )
