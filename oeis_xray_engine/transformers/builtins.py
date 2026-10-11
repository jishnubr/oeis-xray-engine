"""Pipeline adapters for the exact transforms already implemented in X-Ray."""

from __future__ import annotations

from typing import ClassVar

from ..errors import InsufficientTermsError
from ..models import TermSeries
from . import TransformError, euler_transform, inverse_euler_transform
from .contracts import (
    TransformerContract,
    TransformerExecutionContext,
    TransformerResult,
)


def _integer_values(context: TransformerExecutionContext) -> TermSeries:
    series = context.input_series
    if series.indices is not None:
        expected_length = series.complete_through - series.start_index + 1
        if len(series) != expected_length or any(
            index != series.start_index + position
            for position, index in enumerate(series.indices)
        ):
            raise TransformError("Euler transforms require dense input terms")
    elif len(series) != series.complete_through - series.start_index + 1:
        raise InsufficientTermsError("Euler transforms require dense input terms")
    return series


def _output_start(
    context: TransformerExecutionContext,
    default: int,
) -> int:
    value = context.metadata.get("output_start_index", default)
    if type(value) is not int:
        raise TransformError("output_start_index metadata must be an integer")
    return value


def _requested_output_count(
    context: TransformerExecutionContext,
    available: int,
) -> int | None:
    requested = context.metadata.get("output_term_count")
    if requested is None:
        return None
    if type(requested) is not int or requested <= 0 or requested > available:
        raise TransformError(
            "requested output term count is outside computed transform output",
            context={"requested": requested, "available": available},
        )
    return requested


class InverseEulerTransformer(TransformerContract):
    """Recover exact Euler exponents from a formal-series coefficient window."""

    name: ClassVar[str] = "inverse_euler"
    version: ClassVar[str] = "1"

    def transform(
        self,
        context: TransformerExecutionContext,
    ) -> TransformerResult:
        self.validate(context)
        output = inverse_euler_transform(_integer_values(context))
        requested = _requested_output_count(context, len(output))
        if requested is not None:
            output = output[:requested]
        return TransformerResult(
            start_index=_output_start(context, 1),
            values=tuple(output),
            metadata={"coefficient_count": len(output)},
        )


class EulerTransformer(TransformerContract):
    """Expand exact Euler exponents into formal-series coefficients."""

    name: ClassVar[str] = "euler"
    version: ClassVar[str] = "1"

    def transform(
        self,
        context: TransformerExecutionContext,
    ) -> TransformerResult:
        self.validate(context)
        output = euler_transform(
            _integer_values(context),
            degree=context.spec.output_bound,
        )
        if any(type(value) is not int for value in output):
            raise TransformError(
                "pipeline Euler output must have integer coefficients"
            )
        requested = _requested_output_count(context, len(output))
        if requested is not None:
            output = output[:requested]
        return TransformerResult(
            start_index=_output_start(context, 0),
            values=tuple(output),
            metadata={"coefficient_count": len(output)},
        )
