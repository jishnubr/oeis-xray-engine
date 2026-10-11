"""Typed exceptions for input validation, mathematics, workers, and storage."""

from __future__ import annotations

from typing import Any, Mapping


class XRayError(Exception):
    """Base class for errors that callers may handle at the engine boundary."""

    def __init__(
        self,
        message: str,
        *,
        context: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.context = dict(context or {})

    def __str__(self) -> str:
        if not self.context:
            return self.message
        details = ", ".join(
            f"{key}={value!r}" for key, value in sorted(self.context.items())
        )
        return f"{self.message} ({details})"


class ValidationError(XRayError):
    """An input or persisted record violates a declared schema contract."""


class DomainError(XRayError):
    """A mathematical operation is undefined for the supplied input domain."""


class InsufficientTermsError(DomainError):
    """The requested operation requires terms beyond the available range."""


class UnsupportedTransformError(DomainError):
    """A transform name, version, parameter set, or domain is unsupported."""


class StorageError(XRayError):
    """Base class for durable storage failures."""


class CatalogError(StorageError):
    """SQLite catalog operation failed or violated catalog invariants."""


class TermStoreError(StorageError):
    """Term chunk could not be safely written, located, or read."""


class CorruptChunkError(TermStoreError):
    """A chunk is truncated, malformed, inconsistent, or has a bad digest."""


class WorkerError(XRayError):
    """A worker task failed and includes its task identity when available."""
