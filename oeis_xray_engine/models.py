"""Immutable, serializable domain models shared by X-Ray subsystems.

Term indices are explicit. ``TermSeries`` supports sparse observations by
storing the indices corresponding to ``values``; omitted indices within the
declared range are known missing entries, not zero-valued terms.
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from typing import Generic, Iterator, Literal, Mapping, Sequence, TypeVar

Completeness = Literal["prefix", "finite", "unknown"]
ParseStatus = Literal["valid", "partial", "invalid"]
EvidenceLevel = Literal["E0", "E1", "E2", "E3", "E4"]

_COMPLETENESS_VALUES = frozenset(("prefix", "finite", "unknown"))
_PARSE_STATUS_VALUES = frozenset(("valid", "partial", "invalid"))
_EVIDENCE_LEVELS = frozenset(("E0", "E1", "E2", "E3", "E4"))
_K = TypeVar("_K")
_V = TypeVar("_V")


@dataclass(frozen=True, slots=True, init=False)
class FrozenMapping(Mapping[_K, _V], Generic[_K, _V]):
    """Small immutable, pickleable mapping for model parameters."""

    _items: tuple[tuple[_K, _V], ...]

    def __init__(self, values: Mapping[_K, _V]) -> None:
        object.__setattr__(
            self,
            "_items",
            tuple(sorted(values.items(), key=lambda item: item[0])),
        )

    def __getitem__(self, key: _K) -> _V:
        for candidate, value in self._items:
            if candidate == key:
                return value
        raise KeyError(key)

    def __iter__(self) -> Iterator[_K]:
        return (key for key, _ in self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __reduce__(self):
        return (type(self), (dict(self._items),))


def _required_text(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


def _integer(value: int, field_name: str, *, minimum: int | None = None) -> None:
    # bool is an int subclass, but is not a meaningful sequence index/value here.
    if type(value) is not int:
        raise TypeError(f"{field_name} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{field_name} must be at least {minimum}")


@dataclass(frozen=True, slots=True)
class SequenceRef:
    """Identify one immutable sequence revision within a source snapshot."""

    sequence_id: str
    revision_id: str
    snapshot_id: str

    def __post_init__(self) -> None:
        _required_text(self.sequence_id, "sequence_id")
        _required_text(self.revision_id, "revision_id")
        _required_text(self.snapshot_id, "snapshot_id")


@dataclass(frozen=True, slots=True)
class SequenceDescriptor:
    """Metadata for a parsed sequence revision.

    ``term_count`` counts present terms, including for sparse series.
    ``offset`` is the declared starting index when available, while
    ``known_end_index`` is the greatest index whose source data is known.
    """

    ref: SequenceRef
    offset: int | None
    term_count: int
    known_end_index: int | None
    completeness: Completeness
    term_digest: str
    source_digest: str
    parse_status: ParseStatus

    def __post_init__(self) -> None:
        if not isinstance(self.ref, SequenceRef):
            raise TypeError("ref must be a SequenceRef")
        if self.offset is not None:
            _integer(self.offset, "offset")
        _integer(self.term_count, "term_count", minimum=0)
        if self.known_end_index is not None:
            _integer(self.known_end_index, "known_end_index")
        if self.term_count and self.known_end_index is None:
            raise ValueError("known_end_index is required when term_count is non-zero")
        if (
            self.offset is not None
            and self.known_end_index is not None
            and self.known_end_index < self.offset
        ):
            raise ValueError("known_end_index cannot precede offset")
        if self.completeness not in _COMPLETENESS_VALUES:
            raise ValueError(f"unsupported completeness value: {self.completeness!r}")
        if self.parse_status not in _PARSE_STATUS_VALUES:
            raise ValueError(f"unsupported parse_status value: {self.parse_status!r}")
        if not isinstance(self.term_digest, str):
            raise TypeError("term_digest must be a string")
        _required_text(self.source_digest, "source_digest")


@dataclass(frozen=True, slots=True)
class SequenceRevision:
    """Immutable parsed source revision persisted in the catalog."""

    ref: SequenceRef
    source_digest: str
    term_digest: str | None
    offset: int | None
    term_count: int
    known_end_index: int | None
    completeness: Completeness
    parse_status: ParseStatus
    raw_metadata_json: str
    diagnostics_json: str

    def __post_init__(self) -> None:
        if not isinstance(self.ref, SequenceRef):
            raise TypeError("ref must be a SequenceRef")
        _required_text(self.source_digest, "source_digest")
        if self.term_digest is not None and not isinstance(self.term_digest, str):
            raise TypeError("term_digest must be a string or None")
        if self.offset is not None:
            _integer(self.offset, "offset")
        _integer(self.term_count, "term_count", minimum=0)
        if self.known_end_index is not None:
            _integer(self.known_end_index, "known_end_index")
        if self.term_count and self.known_end_index is None:
            raise ValueError("known_end_index is required when terms are present")
        if self.completeness not in _COMPLETENESS_VALUES:
            raise ValueError(f"unsupported completeness value: {self.completeness!r}")
        if self.parse_status not in _PARSE_STATUS_VALUES:
            raise ValueError(f"unsupported parse_status value: {self.parse_status!r}")
        if not isinstance(self.raw_metadata_json, str):
            raise TypeError("raw_metadata_json must be a string")
        if not isinstance(self.diagnostics_json, str):
            raise TypeError("diagnostics_json must be a string")

    @property
    def descriptor(self) -> SequenceDescriptor:
        """Return the lightweight descriptor view of this revision."""

        return SequenceDescriptor(
            ref=self.ref,
            offset=self.offset,
            term_count=self.term_count,
            known_end_index=self.known_end_index,
            completeness=self.completeness,
            term_digest=self.term_digest or "",
            source_digest=self.source_digest,
            parse_status=self.parse_status,
        )


@dataclass(frozen=True, slots=True)
class TermSeries:
    """A bounded, exact sequence slice with optional sparse indices.

    If ``indices`` is omitted, values are dense and map to consecutive indices
    beginning at ``start_index``. If supplied, ``indices`` must contain one
    strictly increasing index per value. Missing indices inside
    ``[start_index, complete_through]`` are absent observations; they are never
    interpreted as zero.

    ``complete_through`` is the end of the known input range, not necessarily
    the index of the last present value. Thus a sparse series can distinguish
    a missing term from an unobserved tail.
    """

    ref: SequenceRef
    start_index: int
    values: Sequence[int]
    complete_through: int
    indices: Sequence[int] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.ref, SequenceRef):
            raise TypeError("ref must be a SequenceRef")
        _integer(self.start_index, "start_index")
        _integer(self.complete_through, "complete_through")
        if self.complete_through < self.start_index:
            raise ValueError("complete_through cannot precede start_index")

        values = tuple(self.values)
        for position, value in enumerate(values):
            _integer(value, f"values[{position}]")
        object.__setattr__(self, "values", values)

        if self.indices is None:
            last_index = self.start_index + len(values) - 1
            if values and last_index > self.complete_through:
                raise ValueError("dense values extend beyond complete_through")
            return

        indices = tuple(self.indices)
        if len(indices) != len(values):
            raise ValueError("indices and values must have the same length")
        previous: int | None = None
        for position, index in enumerate(indices):
            _integer(index, f"indices[{position}]")
            if index < self.start_index or index > self.complete_through:
                raise ValueError(
                    f"indices[{position}]={index} is outside the known range "
                    f"[{self.start_index}, {self.complete_through}]"
                )
            if previous is not None and index <= previous:
                raise ValueError("indices must be strictly increasing")
            previous = index
        object.__setattr__(self, "indices", indices)

    def __len__(self) -> int:
        """Return the number of present terms, not the width of the index range."""

        return len(self.values)

    def iter_indexed(self) -> Iterator[tuple[int, int]]:
        """Yield present ``(index, value)`` pairs in increasing index order."""

        if self.indices is None:
            for offset, value in enumerate(self.values):
                yield self.start_index + offset, value
        else:
            yield from zip(self.indices, self.values)

    def value_at(self, index: int) -> int | None:
        """Return a term, or ``None`` when its index is known to be missing.

        Raise ``IndexError`` for indices outside the declared known range.
        """

        _integer(index, "index")
        if index < self.start_index or index > self.complete_through:
            raise IndexError(
                f"index {index} is outside the known range "
                f"[{self.start_index}, {self.complete_through}]"
            )
        if self.indices is None:
            position = index - self.start_index
            if position < len(self.values):
                return self.values[position]
            return None
        position = bisect_left(self.indices, index)
        if position < len(self.indices) and self.indices[position] == index:
            return self.values[position]
        return None

    def is_missing(self, index: int) -> bool:
        """Return whether ``index`` is an in-range absent observation."""

        return self.value_at(index) is None


@dataclass(frozen=True, slots=True)
class TransformSpec:
    """Versioned, bounded mathematical transform request."""

    name: str
    version: str
    parameters: Mapping[str, str | int | bool]
    input_domain: str
    output_bound: int

    def __post_init__(self) -> None:
        _required_text(self.name, "name")
        _required_text(self.version, "version")
        _required_text(self.input_domain, "input_domain")
        _integer(self.output_bound, "output_bound", minimum=0)
        if not isinstance(self.parameters, Mapping):
            raise TypeError("parameters must be a mapping")
        for key, value in self.parameters.items():
            _required_text(key, "parameter key")
            if type(value) not in (str, int, bool):
                raise TypeError(
                    f"parameter {key!r} must be a string, integer, or boolean"
                )
        object.__setattr__(self, "parameters", FrozenMapping(self.parameters))


@dataclass(frozen=True, slots=True)
class RelationClaim:
    """A provenance-bearing claim about a relation between two revisions."""

    relation_kind: str
    source: SequenceRef
    target: SequenceRef
    index_map: Mapping[str, int | str]
    parameters: Mapping[str, str | int]
    compared_start: int
    compared_end: int
    evidence_level: EvidenceLevel
    evidence_digest: str
    transform_spec: TransformSpec | None = None

    def __post_init__(self) -> None:
        _required_text(self.relation_kind, "relation_kind")
        if not isinstance(self.source, SequenceRef):
            raise TypeError("source must be a SequenceRef")
        if not isinstance(self.target, SequenceRef):
            raise TypeError("target must be a SequenceRef")
        for field_name, mapping in (
            ("index_map", self.index_map),
            ("parameters", self.parameters),
        ):
            if not isinstance(mapping, Mapping):
                raise TypeError(f"{field_name} must be a mapping")
            for key, value in mapping.items():
                _required_text(key, f"{field_name} key")
                if field_name == "index_map":
                    valid = type(value) is int or isinstance(value, str)
                else:
                    valid = type(value) is int or isinstance(value, str)
                if not valid:
                    raise TypeError(f"{field_name}[{key!r}] has an unsupported value")
        object.__setattr__(self, "index_map", FrozenMapping(self.index_map))
        object.__setattr__(self, "parameters", FrozenMapping(self.parameters))
        _integer(self.compared_start, "compared_start")
        _integer(self.compared_end, "compared_end")
        if self.compared_end < self.compared_start:
            raise ValueError("compared_end cannot precede compared_start")
        if self.evidence_level not in _EVIDENCE_LEVELS:
            raise ValueError(f"unsupported evidence level: {self.evidence_level!r}")
        _required_text(self.evidence_digest, "evidence_digest")
        if self.transform_spec is not None and not isinstance(
            self.transform_spec, TransformSpec
        ):
            raise TypeError("transform_spec must be a TransformSpec or None")


@dataclass(frozen=True, slots=True)
class ChunkRef:
    """Catalog reference to an immutable file-backed or bundled term chunk."""

    chunk_id: str
    revision_id: str
    first_index: int
    term_count: int
    last_index: int
    codec: str
    format_version: int
    uncompressed_bytes: int
    payload_digest: str
    content_path: str
    bundle_id: str | None = None
    byte_offset: int | None = None
    byte_length: int | None = None

    def __post_init__(self) -> None:
        _required_text(self.chunk_id, "chunk_id")
        _required_text(self.revision_id, "revision_id")
        _integer(self.first_index, "first_index")
        _integer(self.term_count, "term_count", minimum=1)
        _integer(self.last_index, "last_index")
        if self.last_index != self.first_index + self.term_count - 1:
            raise ValueError("last_index must match first_index and term_count")
        _required_text(self.codec, "codec")
        _integer(self.format_version, "format_version", minimum=1)
        _integer(self.uncompressed_bytes, "uncompressed_bytes", minimum=1)
        _required_text(self.payload_digest, "payload_digest")
        if self.bundle_id is None:
            _required_text(self.content_path, "content_path")
            if self.byte_offset is not None or self.byte_length is not None:
                raise ValueError("file-backed chunks cannot have bundle byte ranges")
        else:
            _required_text(self.bundle_id, "bundle_id")
            if self.content_path:
                raise ValueError("bundled chunks cannot have a content_path")
            _integer(self.byte_offset, "byte_offset", minimum=0)
            _integer(self.byte_length, "byte_length", minimum=1)


@dataclass(frozen=True, slots=True)
class IntegrityReport:
    """Result of validating a stored term chunk."""

    valid: bool
    chunk_id: str
    payload_digest: str
    actual_digest: str | None
    term_count: int | None
    error: str | None = None
