"""Pure transformer contracts and immutable execution records.

Transformer implementations receive only bounded mathematical input and
immutable metadata. They must not access files, databases, networks, or
orchestration state; adapters and the pipeline own those concerns.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar, Mapping

from ..models import FrozenMapping, TermSeries, TransformSpec
from . import TransformError

MetadataValue = str | int | bool | None


def _freeze_metadata(
    metadata: Mapping[str, MetadataValue],
    field_name: str,
) -> FrozenMapping[str, MetadataValue]:
    if not isinstance(metadata, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    for key, value in metadata.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError(f"{field_name} keys must be non-empty strings")
        if type(value) not in (str, int, bool) and value is not None:
            raise TypeError(
                f"{field_name}[{key!r}] must be a JSON-compatible scalar"
            )
    return FrozenMapping(metadata)


@dataclass(frozen=True, slots=True)
class TransformerExecutionContext:
    """Bounded, read-only input and invocation metadata for a transformer.

    ``input_series`` represents the exact term window supplied to one
    invocation. ``metadata`` contains only JSON-compatible scalar values,
    such as source revision, requested output start index, and count. The
    context deliberately exposes no storage or filesystem handles.
    """

    input_series: TermSeries
    spec: TransformSpec
    metadata: Mapping[str, MetadataValue]

    def __post_init__(self) -> None:
        if not isinstance(self.input_series, TermSeries):
            raise TypeError("input_series must be a TermSeries")
        if not isinstance(self.spec, TransformSpec):
            raise TypeError("spec must be a TransformSpec")
        object.__setattr__(
            self,
            "metadata",
            _freeze_metadata(self.metadata, "metadata"),
        )


@dataclass(frozen=True, slots=True)
class TransformerResult:
    """Exact finite output from a transformer, before persistence."""

    start_index: int
    values: tuple[int, ...]
    metadata: Mapping[str, MetadataValue]

    def __post_init__(self) -> None:
        if type(self.start_index) is not int:
            raise TypeError("start_index must be an integer")
        values = tuple(self.values)
        for position, value in enumerate(values):
            if type(value) is not int:
                raise TypeError(f"values[{position}] must be an integer")
        object.__setattr__(self, "values", values)
        object.__setattr__(
            self,
            "metadata",
            _freeze_metadata(self.metadata, "metadata"),
        )


class TransformerContract(ABC):
    """Abstract contract for a pure, deterministic, versioned transform.

    Subclasses must define stable ``name`` and ``version`` class attributes
    and implement :meth:`transform`. Implementations must not perform I/O or
    mutate shared state. All input data arrives through the execution context;
    outputs are returned as immutable exact values.
    """

    name: ClassVar[str]
    version: ClassVar[str]

    def validate(self, context: TransformerExecutionContext) -> None:
        """Validate that this implementation matches the requested spec."""

        if not isinstance(context, TransformerExecutionContext):
            raise TypeError("context must be a TransformerExecutionContext")
        if context.spec.name != self.name or context.spec.version != self.version:
            raise TransformError(
                "transform spec does not match the registered implementation",
                context={
                    "requested": f"{context.spec.name}@{context.spec.version}",
                    "implementation": f"{self.name}@{self.version}",
                },
            )

    @abstractmethod
    def transform(self, context: TransformerExecutionContext) -> TransformerResult:
        """Compute an exact finite transform without performing I/O."""

