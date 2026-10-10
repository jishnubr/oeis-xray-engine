"""Deterministic, versioned fingerprints for candidate retrieval.

Fingerprints are only candidate-generation evidence (E0). Equality of a digest
never verifies a relation; callers must submit candidates to an exact verifier.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Iterable, Iterator, Literal

from ..errors import CatalogError, TermStoreError, ValidationError
from ..models import RelationClaim, SequenceRef, TermSeries
from ..storage.catalog import Catalog
from ..storage.term_store import TermStore

FingerprintMode = Literal["exact", "abs", "diff"]
_MODES = frozenset(("exact", "abs", "diff"))
_CANONICALIZATION_VERSION = "integer-indexed-blake2b-v1"
_MAX_SQLITE_INTEGER = (1 << 63) - 1
_MIN_SQLITE_INTEGER = -(1 << 63)


@dataclass(frozen=True, slots=True)
class FingerprintProfile:
    """Describe how a bounded sequence range is normalized before hashing.

    ``term_count`` is the width of the source index range. For ``diff`` mode,
    this must be at least two and fingerprints the ``term_count - 1`` adjacent
    differences. If ``start_index`` is omitted, the sequence's declared
    offset is used; an absent offset makes fingerprinting fail explicitly.
    """

    name: str
    version: str
    mode: FingerprintMode
    term_count: int
    start_index: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("profile name must be a non-empty string")
        if not isinstance(self.version, str) or not self.version.strip():
            raise ValueError("profile version must be a non-empty string")
        if self.mode not in _MODES:
            raise ValueError(f"unsupported fingerprint mode: {self.mode!r}")
        if type(self.term_count) is not int or self.term_count <= 0:
            raise ValueError("term_count must be a positive integer")
        if self.mode == "diff" and self.term_count < 2:
            raise ValueError("diff profiles require term_count >= 2")
        if self.start_index is not None and type(self.start_index) is not int:
            raise TypeError("start_index must be an integer or None")

    @property
    def profile_key(self) -> str:
        """Stable digest of the complete canonicalization profile."""

        payload = json.dumps(
            {
                "name": self.name,
                "version": self.version,
                "mode": self.mode,
                "term_count": self.term_count,
                "start_index": self.start_index,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    @property
    def output_count(self) -> int:
        """Number of normalized terms hashed by this profile."""

        return self.term_count - 1 if self.mode == "diff" else self.term_count


@dataclass(frozen=True, slots=True)
class Fingerprint:
    """A stored or transient fingerprint. Evidence is always E0."""

    ref: SequenceRef
    profile: FingerprintProfile
    range_start: int
    range_end: int
    digest: str
    canonicalization_version: str = _CANONICALIZATION_VERSION
    evidence_level: Literal["E0"] = "E0"

    def __post_init__(self) -> None:
        if self.evidence_level != "E0":
            raise ValueError("fingerprints can only carry E0 evidence")
        if self.range_end < self.range_start:
            raise ValueError("range_end cannot precede range_start")
        if not self.digest:
            raise ValueError("digest must not be empty")


@dataclass(frozen=True, slots=True)
class Candidate:
    """An E0 pair retrieved from a shared fingerprint bucket."""

    source: SequenceRef
    target: SequenceRef
    profile: FingerprintProfile
    range_start: int
    range_end: int
    digest: str
    evidence_level: Literal["E0"] = "E0"

    def __post_init__(self) -> None:
        if self.evidence_level != "E0":
            raise ValueError("hash candidates can only carry E0 evidence")
        if self.source == self.target:
            raise ValueError("a candidate must link two distinct revisions")
        if self.range_end < self.range_start:
            raise ValueError("range_end cannot precede range_start")


class HashIndex:
    """Hash canonical integer/index records and retrieve matching candidates.

    Args:
        catalog: Coordinator-owned SQLite catalog.
        term_store: Store used to load exact indexed values.
        run_id: Existing catalog run that owns newly persisted fingerprints.

    A fingerprint's canonical input is length-delimited and includes each
    original index and transformed integer. Missing indices have an explicit
    marker. Hash matching is strictly an E0 retrieval operation.
    """

    def __init__(
        self,
        catalog: Catalog,
        term_store: TermStore,
        run_id: str,
        *,
        max_bucket_size: int = 10_000,
    ) -> None:
        if not isinstance(catalog, Catalog):
            raise TypeError("catalog must be a Catalog")
        if not isinstance(term_store, TermStore):
            raise TypeError("term_store must be a TermStore")
        if term_store.catalog is not catalog:
            raise ValueError("term_store must use the same catalog instance")
        if not isinstance(run_id, str) or not run_id.strip():
            raise ValueError("run_id must be a non-empty string")
        if type(max_bucket_size) is not int or max_bucket_size <= 0:
            raise ValueError("max_bucket_size must be a positive integer")
        run = catalog.fetch_one("SELECT run_id FROM run WHERE run_id = ?", (run_id,))
        if run is None:
            raise CatalogError(
                "fingerprint run does not exist",
                context={"run_id": run_id},
            )
        self.catalog = catalog
        self.term_store = term_store
        self.run_id = run_id
        self.max_bucket_size = max_bucket_size

    def fingerprint(
        self,
        ref: SequenceRef,
        profile: FingerprintProfile,
    ) -> Fingerprint:
        """Get or compute and persist one sequence fingerprint.

        Existing fingerprints are reused only when their canonicalization
        version and digest metadata agree with this implementation.
        """

        self._validate_inputs(ref, profile)
        self._validate_catalog_ref(ref)
        range_start = self._resolve_start_index(ref, profile)
        range_end = range_start + profile.term_count - 1
        self._validate_sqlite_range(range_start, range_end)
        profile_key = profile.profile_key

        row = self.catalog.fetch_one(
            """
            SELECT fingerprint_id, digest, canonicalization_version
            FROM fingerprint
            WHERE revision_id = ? AND profile_key = ?
              AND range_start = ? AND range_end = ?
            """,
            (ref.revision_id, profile_key, range_start, range_end),
        )
        if row is not None:
            if row["canonicalization_version"] != _CANONICALIZATION_VERSION:
                raise CatalogError(
                    "stored fingerprint uses an unsupported canonicalization version",
                    context={
                        "fingerprint_id": row["fingerprint_id"],
                        "stored_version": row["canonicalization_version"],
                        "expected_version": _CANONICALIZATION_VERSION,
                    },
                )
            return Fingerprint(
                ref=ref,
                profile=profile,
                range_start=range_start,
                range_end=range_end,
                digest=row["digest"],
            )

        terms = self.term_store.read_range(ref, range_start, profile.term_count)
        digest = self.digest_series(terms, profile)
        result = Fingerprint(
            ref=ref,
            profile=profile,
            range_start=range_start,
            range_end=range_end,
            digest=digest,
        )
        self._persist(result)
        return result

    def fingerprint_values(
        self,
        ref: SequenceRef,
        profile: FingerprintProfile,
        values: Iterable[int],
        *,
        start_index: int,
    ) -> Fingerprint:
        """Create a transient fingerprint for computed, consecutive values.

        The fingerprint is not persisted. This is intended for transforming a
        source sequence and probing the target-sequence index without writing a
        synthetic transformed sequence into the corpus.
        """

        self._validate_inputs(ref, profile)
        detached_values = tuple(values)
        if len(detached_values) != profile.term_count:
            raise ValidationError(
                "computed value count does not match fingerprint profile",
                context={
                    "expected": profile.term_count,
                    "actual": len(detached_values),
                },
            )
        for position, value in enumerate(detached_values):
            if type(value) is not int:
                raise ValidationError(
                    f"computed values[{position}] must be an integer",
                    context={"actual_type": type(value).__name__},
                )
        range_end = start_index + profile.term_count - 1
        self._validate_sqlite_range(start_index, range_end)
        series = TermSeries(
            ref=ref,
            start_index=start_index,
            values=detached_values,
            complete_through=range_end,
        )
        return Fingerprint(
            ref=ref,
            profile=profile,
            range_start=start_index,
            range_end=range_end,
            digest=self.digest_series(series, profile),
        )

    def find_candidates(self, fingerprint: Fingerprint) -> Iterator[Candidate]:
        """Stream E0 pairs between one fingerprint and its matching bucket.

        A bucket larger than the configured limit raises ``ValidationError``
        rather than triggering an unbounded pairwise comparison.
        """

        if not isinstance(fingerprint, Fingerprint):
            raise TypeError("fingerprint must be a Fingerprint")
        rows = self._bucket_rows(fingerprint)
        for row in rows:
            if row["revision_id"] == fingerprint.ref.revision_id:
                continue
            candidate = Candidate(
                source=fingerprint.ref,
                target=self.catalog.get_sequence_ref(row["revision_id"]),
                profile=fingerprint.profile,
                range_start=fingerprint.range_start,
                range_end=fingerprint.range_end,
                digest=fingerprint.digest,
            )
            self.record_candidate(candidate)
            yield candidate

    def record_candidate(self, candidate: Candidate) -> RelationClaim:
        """Persist and return the E0 claim represented by a hash candidate."""

        if not isinstance(candidate, Candidate):
            raise TypeError("candidate must be a Candidate")
        if candidate.evidence_level != "E0":
            raise ValidationError("only E0 candidates may be recorded by HashIndex")
        evidence_digest = hashlib.sha256(
            json.dumps(
                {
                    "kind": "fingerprint-candidate",
                    "source_revision_id": candidate.source.revision_id,
                    "target_revision_id": candidate.target.revision_id,
                    "profile_key": candidate.profile.profile_key,
                    "range_start": candidate.range_start,
                    "range_end": candidate.range_end,
                    "digest": candidate.digest,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        claim = RelationClaim(
            relation_kind=f"fingerprint:{candidate.profile.mode}",
            source=candidate.source,
            target=candidate.target,
            index_map={
                "range_start": candidate.range_start,
                "range_end": candidate.range_end,
            },
            parameters={
                "profile_key": candidate.profile.profile_key,
                "profile_name": candidate.profile.name,
                "profile_version": candidate.profile.version,
                "digest": candidate.digest,
            },
            compared_start=candidate.range_start,
            compared_end=candidate.range_end,
            evidence_level="E0",
            evidence_digest=evidence_digest,
        )
        claim_id = hashlib.sha256(
            f"E0:{candidate.source.revision_id}:{candidate.target.revision_id}:"
            f"{candidate.profile.profile_key}:{candidate.range_start}:"
            f"{candidate.range_end}:{candidate.digest}".encode("utf-8")
        ).hexdigest()
        with self.catalog.transaction():
            self.catalog.execute(
                """
                INSERT INTO relation_claim (
                  claim_id, relation_kind, source_revision_id,
                  target_revision_id, transform_key, index_map_json,
                  parameters_json, compared_start, compared_end,
                  evidence_level, status, evidence_digest, created_by_run,
                  supersedes_claim_id, created_at
                ) VALUES (?, ?, ?, ?, NULL, ?, ?, ?, ?, 'E0', 'candidate',
                          ?, ?, NULL, CURRENT_TIMESTAMP)
                ON CONFLICT(claim_id) DO NOTHING
                """,
                (
                    claim_id,
                    claim.relation_kind,
                    claim.source.revision_id,
                    claim.target.revision_id,
                    json.dumps(dict(claim.index_map), sort_keys=True),
                    json.dumps(dict(claim.parameters), sort_keys=True),
                    claim.compared_start,
                    claim.compared_end,
                    claim.evidence_digest,
                    self.run_id,
                ),
            )
        return claim

    def find_matching_refs(self, fingerprint: Fingerprint) -> list[SequenceRef]:
        """Return matching stored revisions, bounded by the bucket-size limit."""

        if not isinstance(fingerprint, Fingerprint):
            raise TypeError("fingerprint must be a Fingerprint")
        rows = self._bucket_rows(fingerprint)
        return [
            self.catalog.get_sequence_ref(row["revision_id"])
            for row in rows
            if row["revision_id"] != fingerprint.ref.revision_id
        ]

    def _bucket_rows(self, fingerprint: Fingerprint) -> list[object]:
        rows = self.catalog.fetch_all(
            """
            SELECT revision_id
            FROM fingerprint
            WHERE profile_key = ? AND range_start = ? AND range_end = ?
              AND canonicalization_version = ? AND digest = ?
            ORDER BY revision_id
            LIMIT ?
            """,
            (
                fingerprint.profile.profile_key,
                fingerprint.range_start,
                fingerprint.range_end,
                fingerprint.canonicalization_version,
                fingerprint.digest,
                self.max_bucket_size + 1,
            ),
        )
        if len(rows) > self.max_bucket_size:
            raise ValidationError(
                "fingerprint bucket exceeds configured candidate limit",
                context={
                    "profile_key": fingerprint.profile.profile_key,
                    "bucket_limit": self.max_bucket_size,
                },
            )
        return rows

    @staticmethod
    def digest_series(series: TermSeries, profile: FingerprintProfile) -> str:
        """Return a BLAKE2b digest of indexed, length-delimited profile values."""

        if profile.start_index is not None and profile.start_index != series.start_index:
            raise ValidationError(
                "series start index does not match fingerprint profile",
                context={
                    "profile_start": profile.start_index,
                    "series_start": series.start_index,
                },
            )
        expected_end = series.start_index + profile.term_count - 1
        if series.complete_through < expected_end:
            raise ValidationError(
                "series does not cover the fingerprint profile range",
                context={
                    "complete_through": series.complete_through,
                    "required_end": expected_end,
                },
            )

        hasher = hashlib.blake2b(digest_size=32)
        HashIndex._feed_field(hasher, b"profile", profile.profile_key.encode("ascii"))
        HashIndex._feed_field(hasher, b"range-start", _encode_integer(series.start_index))
        HashIndex._feed_field(hasher, b"range-end", _encode_integer(expected_end))
        indexed_values = dict(series.iter_indexed())

        if profile.mode == "diff":
            for index in range(series.start_index, expected_end):
                left = indexed_values.get(index)
                right = indexed_values.get(index + 1)
                if left is None or right is None:
                    HashIndex._feed_field(
                        hasher,
                        b"missing-difference",
                        _encode_integer(index),
                    )
                else:
                    HashIndex._feed_field(
                        hasher,
                        b"difference",
                        _encode_integer(index),
                        _encode_integer(right - left),
                    )
        else:
            for index in range(series.start_index, expected_end + 1):
                value = indexed_values.get(index)
                if value is None:
                    HashIndex._feed_field(
                        hasher,
                        b"missing-term",
                        _encode_integer(index),
                    )
                    continue
                normalized = abs(value) if profile.mode == "abs" else value
                HashIndex._feed_field(
                    hasher,
                    b"term",
                    _encode_integer(index),
                    _encode_integer(normalized),
                )
        return hasher.hexdigest()

    def _persist(self, fingerprint: Fingerprint) -> None:
        fingerprint_id = hashlib.sha256(
            json.dumps(
                [
                    fingerprint.ref.revision_id,
                    fingerprint.profile.profile_key,
                    fingerprint.range_start,
                    fingerprint.range_end,
                ],
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        with self.catalog.transaction():
            existing = self.catalog.fetch_one(
                """
                SELECT fingerprint_id, digest, canonicalization_version
                FROM fingerprint
                WHERE revision_id = ? AND profile_key = ?
                  AND range_start = ? AND range_end = ?
                """,
                (
                    fingerprint.ref.revision_id,
                    fingerprint.profile.profile_key,
                    fingerprint.range_start,
                    fingerprint.range_end,
                ),
            )
            if existing is not None:
                if (
                    existing["digest"] != fingerprint.digest
                    or existing["canonicalization_version"]
                    != fingerprint.canonicalization_version
                ):
                    raise CatalogError(
                        "fingerprint key conflicts with persisted content",
                        context={"revision_id": fingerprint.ref.revision_id},
                    )
                return
            self.catalog.execute(
                """
                INSERT INTO fingerprint (
                  fingerprint_id, revision_id, profile_key, range_start,
                  range_end, digest, canonicalization_version, created_by_run
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    fingerprint_id,
                    fingerprint.ref.revision_id,
                    fingerprint.profile.profile_key,
                    fingerprint.range_start,
                    fingerprint.range_end,
                    fingerprint.digest,
                    fingerprint.canonicalization_version,
                    self.run_id,
                ),
            )

    def _resolve_start_index(
        self,
        ref: SequenceRef,
        profile: FingerprintProfile,
    ) -> int:
        if profile.start_index is not None:
            return profile.start_index
        row = self.catalog.fetch_one(
            """
            SELECT offset
            FROM sequence_revision
            WHERE revision_id = ? AND sequence_id = ? AND snapshot_id = ?
            """,
            (ref.revision_id, ref.sequence_id, ref.snapshot_id),
        )
        if row is None:
            raise CatalogError(
                "sequence revision does not exist in catalog",
                context={"revision_id": ref.revision_id},
            )
        if row["offset"] is None:
            chunk = self.catalog.fetch_one(
                """
                SELECT first_index FROM term_chunk
                WHERE revision_id = ? ORDER BY first_index LIMIT 1
                """,
                (ref.revision_id,),
            )
            if chunk is None:
                raise TermStoreError(
                    "sequence offset is unknown and the revision has no term chunks",
                    context={"revision_id": ref.revision_id},
                )
            return int(chunk["first_index"])
        return int(row["offset"])

    def _validate_catalog_ref(self, ref: SequenceRef) -> None:
        stored_ref = self.catalog.get_sequence_ref(ref.revision_id)
        if stored_ref != ref:
            raise CatalogError(
                "sequence reference does not match catalog revision metadata",
                context={
                    "requested": ref.revision_id,
                    "catalog_sequence_id": stored_ref.sequence_id,
                    "catalog_snapshot_id": stored_ref.snapshot_id,
                },
            )

    @staticmethod
    def _validate_inputs(ref: SequenceRef, profile: FingerprintProfile) -> None:
        if not isinstance(ref, SequenceRef):
            raise TypeError("ref must be a SequenceRef")
        if not isinstance(profile, FingerprintProfile):
            raise TypeError("profile must be a FingerprintProfile")

    @staticmethod
    def _validate_sqlite_range(start: int, end: int) -> None:
        if not (_MIN_SQLITE_INTEGER <= start <= _MAX_SQLITE_INTEGER):
            raise ValidationError("fingerprint range start exceeds SQLite integer range")
        if not (_MIN_SQLITE_INTEGER <= end <= _MAX_SQLITE_INTEGER):
            raise ValidationError("fingerprint range end exceeds SQLite integer range")

    @staticmethod
    def _feed_field(hasher: hashlib._Hash, *fields: bytes) -> None:
        for field in fields:
            hasher.update(len(field).to_bytes(8, "big"))
            hasher.update(field)


def _encode_integer(value: int) -> bytes:
    """Encode any Python integer as sign byte, byte length, and magnitude."""

    if type(value) is not int:
        raise TypeError("canonical integer encoding accepts exact integers only")
    sign = b"\x01" if value < 0 else b"\x00"
    magnitude = abs(value)
    if magnitude == 0:
        encoded = b"\x00"
    else:
        encoded = magnitude.to_bytes((magnitude.bit_length() + 7) // 8, "big")
    return sign + len(encoded).to_bytes(8, "big") + encoded
