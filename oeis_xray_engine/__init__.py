"""Core data and storage primitives for the OEIS X-Ray Engine.

The package deliberately exposes only stable domain models and typed errors at
its root. Computational subsystems are imported from their own modules so that
importing the package does not initialize a database or load optional tools.
"""

from .errors import (
    CatalogError,
    CorruptChunkError,
    DomainError,
    InsufficientTermsError,
    StorageError,
    TermStoreError,
    UnsupportedTransformError,
    ValidationError,
    WorkerError,
    XRayError,
)
from .models import (
    ChunkRef,
    IntegrityReport,
    RelationClaim,
    SequenceDescriptor,
    SequenceRef,
    SequenceRevision,
    TermSeries,
    TransformSpec,
)

__all__ = [
    "CatalogError",
    "ChunkRef",
    "CorruptChunkError",
    "DomainError",
    "InsufficientTermsError",
    "IntegrityReport",
    "RelationClaim",
    "SequenceDescriptor",
    "SequenceRef",
    "SequenceRevision",
    "StorageError",
    "TermSeries",
    "TermStoreError",
    "TransformSpec",
    "UnsupportedTransformError",
    "ValidationError",
    "WorkerError",
    "XRayError",
]

__version__ = "0.1.0"
