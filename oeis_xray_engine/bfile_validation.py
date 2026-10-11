"""Coordinator-side finite corroboration against OEIS B-files.

B-file retrieval is injected through :class:`BFileFetcher`; this module does
not perform network or filesystem I/O. Results are orthogonal validation
statuses and never change a relation claim's E0--E4 evidence level.
"""

from __future__ import annotations

import hashlib
import io
from collections.abc import Iterator
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from .models import TermSeries

BFILE_VALIDATOR_VERSION = "bfile-indexed-v1"


class BFileValidationStatus(str, Enum):
    """Result of comparing indexed B-file data with a stored revision."""

    MATCH = "match"
    MISMATCH = "mismatch"
    UNAVAILABLE = "unavailable"
    MALFORMED = "malformed"
    INSUFFICIENT_DATA = "insufficient_data"


class BFileValidationKind(str, Enum):
    """Data whose finite consistency is being checked against a B-file."""

    SEQUENCE = "sequence"
    DENUMERANT = "denumerant"


@dataclass(frozen=True, slots=True)
class BFileFetchResult:
    """Payload returned by an external B-file acquisition service."""

    available: bool
    content: bytes | None
    source_url: str | None = None
    error_message: str | None = None

    def __post_init__(self) -> None:
        if type(self.available) is not bool:
            raise TypeError("available must be a bool")
        if self.available and not isinstance(self.content, bytes):
            raise TypeError("available fetch results require bytes content")
        if not self.available and self.content is not None:
            raise ValueError("unavailable fetch results cannot contain content")
        if self.source_url is not None and not isinstance(self.source_url, str):
            raise TypeError("source_url must be a string or None")
        if self.error_message is not None and not isinstance(
            self.error_message, str
        ):
            raise TypeError("error_message must be a string or None")


class BFileFetcher(Protocol):
    """External adapter contract; implementations own retrieval and I/O."""

    def fetch(self, sequence_id: str) -> BFileFetchResult:
        """Return B-file bytes or an explicit unavailable result."""


@dataclass(frozen=True, slots=True)
class BFileValidationResult:
    """Finite B-file comparison outcome and reproducible source metadata."""

    validation_kind: BFileValidationKind
    status: BFileValidationStatus
    source_url: str | None
    source_digest: str | None
    comparison_digest: str
    compared_start: int | None
    compared_end: int | None
    terms_checked: int
    first_mismatch_index: int | None = None
    error_message: str | None = None
    validator_version: str = BFILE_VALIDATOR_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.status, BFileValidationStatus):
            raise TypeError("status must be a BFileValidationStatus")
        if not isinstance(self.validation_kind, BFileValidationKind):
            raise TypeError("validation_kind must be a BFileValidationKind")
        if not isinstance(self.comparison_digest, str) or not self.comparison_digest:
            raise ValueError("comparison_digest must be a non-empty string")
        if type(self.terms_checked) is not int or self.terms_checked < 0:
            raise ValueError("terms_checked must be a non-negative integer")
        if (self.compared_start is None) != (self.compared_end is None):
            raise ValueError("comparison range endpoints must be both set or absent")
        if (
            self.compared_start is not None
            and self.compared_end is not None
            and self.compared_end < self.compared_start
        ):
            raise ValueError("compared_end cannot precede compared_start")


class BFileValidationService:
    """Parse and compare bounded B-file bytes against exact stored terms."""

    def __init__(self, *, max_bytes: int = 16 * 1024 * 1024) -> None:
        if type(max_bytes) is not int or max_bytes <= 0:
            raise ValueError("max_bytes must be a positive integer")
        self.max_bytes = max_bytes

    def validate(
        self,
        fetch_result: BFileFetchResult,
        series: TermSeries | None,
        *,
        validation_kind: BFileValidationKind = BFileValidationKind.SEQUENCE,
    ) -> BFileValidationResult:
        """Compare B-file records by their explicit indices, never by position.

        All data lines must contain exactly an integer index and value, with
        strictly increasing unique indices. Lines with malformed data cause a
        ``malformed`` result; they are not silently ignored.
        """

        if not isinstance(fetch_result, BFileFetchResult):
            raise TypeError("fetch_result must be a BFileFetchResult")
        if not isinstance(validation_kind, BFileValidationKind):
            raise TypeError("validation_kind must be a BFileValidationKind")
        comparison_digest = _comparison_digest(series)
        if not fetch_result.available:
            return BFileValidationResult(
                validation_kind=validation_kind,
                status=BFileValidationStatus.UNAVAILABLE,
                source_url=fetch_result.source_url,
                source_digest=None,
                comparison_digest=comparison_digest,
                compared_start=None,
                compared_end=None,
                terms_checked=0,
                error_message=fetch_result.error_message or "B-file unavailable",
            )

        content = fetch_result.content
        if content is None:
            raise ValueError("available fetch result has no content")
        source_digest = hashlib.sha256(content).hexdigest()
        if len(content) > self.max_bytes:
            return self._result(
                BFileValidationStatus.MALFORMED,
                fetch_result.source_url,
                source_digest,
                validation_kind=validation_kind,
                comparison_digest=comparison_digest,
                error_message=f"B-file exceeds configured {self.max_bytes}-byte limit",
            )
        overlap_start: int | None = None
        overlap_end: int | None = None
        mismatch_index: int | None = None
        mismatch_values: tuple[int, int] | None = None
        missing_count = 0
        missing_examples: list[int] = []
        checked = 0
        try:
            for index, bfile_value in _iter_records(content):
                if series is None or not len(series):
                    continue
                if not series.start_index <= index <= series.complete_through:
                    continue
                if overlap_start is None:
                    overlap_start = index
                overlap_end = index
                stored_value = series.value_at(index)
                if stored_value is None:
                    missing_count += 1
                    if len(missing_examples) < 20:
                        missing_examples.append(index)
                    continue
                checked += 1
                if stored_value != bfile_value and mismatch_index is None:
                    mismatch_index = index
                    mismatch_values = (stored_value, bfile_value)
        except (UnicodeDecodeError, ValueError) as exc:
            return self._result(
                BFileValidationStatus.MALFORMED,
                fetch_result.source_url,
                source_digest,
                validation_kind=validation_kind,
                comparison_digest=comparison_digest,
                error_message=str(exc),
            )

        if series is None or not len(series):
            return self._result(
                BFileValidationStatus.INSUFFICIENT_DATA,
                fetch_result.source_url,
                source_digest,
                validation_kind=validation_kind,
                comparison_digest=comparison_digest,
                error_message="sequence revision has no indexed terms",
            )
        if overlap_start is None or overlap_end is None:
            return self._result(
                BFileValidationStatus.INSUFFICIENT_DATA,
                fetch_result.source_url,
                source_digest,
                validation_kind=validation_kind,
                comparison_digest=comparison_digest,
                error_message="B-file has no indices in the stored term range",
            )
        if mismatch_index is not None and mismatch_values is not None:
            return BFileValidationResult(
                validation_kind=validation_kind,
                status=BFileValidationStatus.MISMATCH,
                source_url=fetch_result.source_url,
                source_digest=source_digest,
                comparison_digest=comparison_digest,
                compared_start=overlap_start,
                compared_end=overlap_end,
                terms_checked=checked,
                first_mismatch_index=mismatch_index,
                error_message=(
                    f"value differs at index {mismatch_index}: "
                    f"stored={mismatch_values[0]}, bfile={mismatch_values[1]}"
                ),
            )

        if missing_count:
            return BFileValidationResult(
                validation_kind=validation_kind,
                status=BFileValidationStatus.INSUFFICIENT_DATA,
                source_url=fetch_result.source_url,
                source_digest=source_digest,
                comparison_digest=comparison_digest,
                compared_start=overlap_start,
                compared_end=overlap_end,
                terms_checked=checked,
                error_message=(
                    "stored term series has missing indices in the B-file overlap: "
                    + ",".join(map(str, missing_examples))
                ),
            )

        return BFileValidationResult(
            validation_kind=validation_kind,
            status=BFileValidationStatus.MATCH,
            source_url=fetch_result.source_url,
            source_digest=source_digest,
            comparison_digest=comparison_digest,
            compared_start=overlap_start,
            compared_end=overlap_end,
            terms_checked=checked,
        )

    @staticmethod
    def _result(
        status: BFileValidationStatus,
        source_url: str | None,
        source_digest: str | None,
        *,
        validation_kind: BFileValidationKind,
        comparison_digest: str,
        error_message: str | None = None,
    ) -> BFileValidationResult:
        return BFileValidationResult(
            validation_kind=validation_kind,
            status=status,
            source_url=source_url,
            source_digest=source_digest,
            comparison_digest=comparison_digest,
            compared_start=None,
            compared_end=None,
            terms_checked=0,
            error_message=error_message,
        )


def _iter_records(content: bytes) -> Iterator[tuple[int, int]]:
    """Strictly stream OEIS two-column B-file lines in source order."""

    previous_index: int | None = None
    record_count = 0
    for line_number, raw_line in enumerate(io.BytesIO(content), start=1):
        line = raw_line.decode("utf-8", errors="strict")
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        fields = stripped.split()
        if len(fields) != 2:
            raise ValueError(f"line {line_number}: expected index and value")
        try:
            index, value = (int(field, 10) for field in fields)
        except ValueError as exc:
            raise ValueError(f"line {line_number}: non-integer index or value") from exc
        if previous_index is not None and index <= previous_index:
            raise ValueError(
                f"line {line_number}: B-file indices must be strictly increasing"
            )
        record_count += 1
        previous_index = index
        yield index, value
    if not record_count:
        raise ValueError("B-file contains no term records")


def _comparison_digest(series: TermSeries | None) -> str:
    """Hash the exact indexed comparison input without decimal conversion."""

    digest = hashlib.sha256(b"bfile-comparison-v1\0")
    if series is None:
        digest.update(b"no-series")
        return digest.hexdigest()

    for value in (series.start_index, series.complete_through):
        _update_integer(digest, value)
    for index, value in series.iter_indexed():
        _update_integer(digest, index)
        _update_integer(digest, value)
    return digest.hexdigest()


def _update_integer(digest, value: int) -> None:
    sign = b"-" if value < 0 else b"+"
    magnitude = abs(value)
    encoded = (
        b"\0"
        if magnitude == 0
        else magnitude.to_bytes((magnitude.bit_length() + 7) // 8, "big")
    )
    payload = sign + encoded
    digest.update(len(payload).to_bytes(8, "big"))
    digest.update(payload)


__all__ = [
    "BFILE_VALIDATOR_VERSION",
    "BFileFetcher",
    "BFileFetchResult",
    "BFileValidationKind",
    "BFileValidationResult",
    "BFileValidationService",
    "BFileValidationStatus",
]
