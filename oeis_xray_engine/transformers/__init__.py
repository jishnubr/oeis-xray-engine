"""Exact, finite sequence transforms.

All exported operations use explicit index conventions and exact arithmetic.
Invalid mathematical domains raise :class:`TransformError`; floating-point
inputs are deliberately not accepted.
"""

from __future__ import annotations

from ..errors import DomainError


class TransformError(DomainError):
    """A transform input, parameter, or exactness condition is invalid."""


from .contracts import (
    TransformerContract,
    TransformerExecutionContext,
    TransformerResult,
)
from .registry import TransformerRegistry, default_transformer_registry
from .inverse_euler import euler_transform, inverse_euler_transform
from .mobius import dirichlet_convolution, divisor_sum_transform, mobius_invert
from .plethystic import plethystic_log
from .roots_of_unity import FilteredSeries, filter_residue_class

__all__ = [
    "FilteredSeries",
    "TransformError",
    "TransformerContract",
    "TransformerExecutionContext",
    "TransformerRegistry",
    "TransformerResult",
    "default_transformer_registry",
    "dirichlet_convolution",
    "divisor_sum_transform",
    "euler_transform",
    "filter_residue_class",
    "inverse_euler_transform",
    "mobius_invert",
    "plethystic_log",
]
