"""Exact, ordered collision verification for E0 candidate pairs."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal

from ..errors import CatalogError, ValidationError
from ..indexing.hash_index import Candidate
from ..models import RelationClaim, SequenceRef, TransformSpec
from ..storage.catalog import Catalog
from ..storage.term_store import TermStore
from ..transformers import TransformError, inverse_euler_transform
from ..transformers.denumerant import extract_denominations
TransformFunction = Callable[
    [tuple[int, ...], TransformSpec], list[int] | tuple[int, ...]
]
VerificationStatus = Literal["exact_match", "mismatch", "insufficient_data"]


@dataclass(frozen=True, slots=True)
class RelationSpec:
    """Fully specified finite comparison, optionally after a transform.

    ``source_start_index`` and ``input_term_count`` identify the exact source
    terms provided to the transform. For an untransformed relation,
    ``input_term_count`` must be omitted or equal ``term_count``.
    ``output_start_index`` identifies the first output index emitted by a
    transform; it is ignored for direct comparisons.
    """

    relation_kind: str
    source_start_index: int
    target_start_index: int
    term_count: int
    transform_spec: TransformSpec | None = None
    input_term_count: int | None = None
    output_start_index: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.relation_kind, str) or not self.relation_kind.strip():
            raise ValueError("relation_kind must be a non-empty string")
        for name, value in (
            ("source_start_index", self.source_start_index),
            ("target_start_index", self.target_start_index),
        ):
            if type(value) is not int:
                raise TypeError(f"{name} must be an integer")
        if type(self.term_count) is not int or self.term_count <= 0:
            raise ValueError("term_count must be a positive integer")
        if self.input_term_count is not None and (
            type(self.input_term_count) is not int or self.input_term_count <= 0
        ):
            raise ValueError("input_term_count must be a positive integer or None")
        if self.output_start_index is not None and type(self.output_start_index) is not int:
            raise TypeError("output_start_index must be an integer or None")
        if self.transform_spec is None:
            if self.input_term_count not in (None, self.term_count):
                raise ValueError(
                    "direct comparisons require input_term_count == term_count"
                )
            if self.output_start_index is not None:
                raise ValueError(
                    "output_start_index is only valid for transformed comparisons"
                )
        elif not isinstance(self.transform_spec, TransformSpec):
            raise TypeError("transform_spec must be a TransformSpec or None")


@dataclass(frozen=True, slots=True)
class VerificationResult:
    """Exact verifier outcome and its optional promoted E1 claim."""

    status: VerificationStatus
    checked_count: int
    first_mismatch: tuple[int, int | None, int | None] | None
    missing_index: tuple[str, int] | None
    evidence_digest: str
    claim: RelationClaim | None

    def __post_init__(self) -> None:
        if self.status == "exact_match":
            if self.claim is None or self.claim.evidence_level != "E1":
                raise ValueError("an exact match must carry an E1 relation claim")
        elif self.claim is not None:
            raise ValueError("non-matching results must not carry a relation claim")


class CollisionDetector:
    """Verify a hash candidate by comparing exact ordered terms.

    Sparse missing values produce ``insufficient_data``. A mismatch or missing
    term can never be promoted to E1. If a catalog and run ID are supplied,
    successful claims and their transform specification are persisted in the
    schema's ``relation_claim`` and ``transform_spec`` tables.
    """

    def __init__(
        self,
        term_store: TermStore,
        catalog: Catalog | None = None,
        *,
        run_id: str | None = None,
        transforms: Mapping[tuple[str, str], TransformFunction] | None = None,
    ) -> None:
        if not isinstance(term_store, TermStore):
            raise TypeError("term_store must be a TermStore")
        if catalog is None:
            catalog = term_store.catalog
        if catalog is not None and term_store.catalog is not catalog:
            raise ValueError("term_store and detector must use the same catalog")
        if run_id is not None and not isinstance(run_id, str):
            raise TypeError("run_id must be a string or None")
        if (catalog is None) != (run_id is None):
            raise ValueError("catalog and run_id must either both be supplied or omitted")
        if catalog is not None:
            run = catalog.fetch_one("SELECT run_id FROM run WHERE run_id = ?", (run_id,))
            if run is None:
                raise CatalogError(
                    "relation claim run does not exist",
                    context={"run_id": run_id},
                )

        self.term_store = term_store
        self.catalog = catalog
        self.run_id = run_id
        self._transforms: dict[tuple[str, str], TransformFunction] = {
            ("inverse_euler", "1"): (
                lambda values, _spec: inverse_euler_transform(values)
            ),
            ("denumerant", "1"): _denumerant_transform,
        }
        if transforms:
            self._transforms.update(transforms)

    def register_transform(
        self,
        name: str,
        version: str,
        function: TransformFunction,
    ) -> None:
        """Register an exact transform callable for verification.

        Functions receive a dense tuple of exact integer terms and their full
        ``TransformSpec`` and must return a finite sequence of exact integers.
        The transform's formal domain and index conventions remain the
        responsibility of its ``TransformSpec``.
        """

        if not name or not version:
            raise ValueError("transform name and version must be non-empty")
        if not callable(function):
            raise TypeError("function must be callable")
        key = (name, version)
        if key in self._transforms:
            raise ValueError(f"transform {name!r} version {version!r} is registered")
        self._transforms[key] = function

    def has_transform(self, spec: TransformSpec) -> bool:
        """Return whether an exact implementation is registered for ``spec``."""

        return (spec.name, spec.version) in self._transforms

    def apply_transform(
        self,
        spec: TransformSpec,
        values: tuple[int, ...],
    ) -> tuple[int, ...]:
        """Apply a registered transform after exact output validation."""

        transform = self._transforms.get((spec.name, spec.version))
        if transform is None:
            raise TransformError(
                "transform is not registered with the collision detector",
                context={"name": spec.name, "version": spec.version},
            )
        try:
            raw_output = transform(values, spec)
        except (TransformError, InsufficientTermsError):
            raise
        except Exception as exc:
            raise TransformError(
                "registered transform failed",
                context={
                    "name": spec.name,
                    "version": spec.version,
                    "error": str(exc),
                },
            ) from exc
        if not isinstance(raw_output, (list, tuple)):
            raise TransformError("registered transform must return a list or tuple")
        output: list[int] = []
        if len(raw_output) > spec.output_bound:
            raise TransformError(
                "registered transform output exceeds its declared bound",
                context={
                    "name": spec.name,
                    "version": spec.version,
                    "output_count": len(raw_output),
                    "output_bound": spec.output_bound,
                },
            )
        for index, value in enumerate(raw_output):
            if type(value) is not int:
                raise TransformError(
                    "registered transform returned a non-integer term",
                    context={
                        "output_index": index,
                        "actual_type": type(value).__name__,
                    },
                )
            output.append(value)
        return tuple(output)

    def verify(
        self,
        candidate: Candidate,
        relation: RelationSpec,
        *,
        persist: bool = True,
    ) -> VerificationResult:
        """Check every term in a candidate's full declared comparison range."""

        if not isinstance(candidate, Candidate):
            raise TypeError("candidate must be a Candidate")
        if candidate.evidence_level != "E0":
            raise ValidationError("collision verification requires an E0 candidate")
        if not isinstance(relation, RelationSpec):
            raise TypeError("relation must be a RelationSpec")
        if candidate.source == candidate.target:
            raise ValidationError("candidate source and target must differ")

        candidate_width = candidate.range_end - candidate.range_start + 1
        if candidate_width != relation.term_count:
            raise ValidationError(
                "candidate range width and relation term_count differ",
                context={
                    "candidate_width": candidate_width,
                    "relation_term_count": relation.term_count,
                },
            )
        if candidate.range_start != relation.target_start_index:
            raise ValidationError(
                "candidate range must match relation target start index",
                context={
                    "candidate_start": candidate.range_start,
                    "target_start": relation.target_start_index,
                },
            )

        input_count = relation.input_term_count or relation.term_count
        source_series = self.term_store.read_range(
            candidate.source, relation.source_start_index, input_count
        )
        evidence_hasher = hashlib.sha256()
        evidence_hasher.update(
            json.dumps(
                {
                    "relation_kind": relation.relation_kind,
                    "source": candidate.source.revision_id,
                    "target": candidate.target.revision_id,
                    "source_start": relation.source_start_index,
                    "target_start": relation.target_start_index,
                    "term_count": relation.term_count,
                    "input_term_count": input_count,
                    "transform": _transform_manifest(relation.transform_spec),
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        if relation.transform_spec is None:
            output_values = None
            output_start = relation.source_start_index
        else:
            spec = relation.transform_spec
            dense_input: list[int] = []
            for offset in range(input_count):
                source_index = relation.source_start_index + offset
                value = source_series.value_at(source_index)
                if value is None:
                    _hash_comparison(
                        evidence_hasher,
                        source_index,
                        relation.target_start_index,
                        None,
                        None,
                    )
                    return VerificationResult(
                        status="insufficient_data",
                        checked_count=0,
                        first_mismatch=None,
                        missing_index=("source", source_index),
                        evidence_digest=evidence_hasher.hexdigest(),
                        claim=None,
                    )
                dense_input.append(value)
            output_values = self.apply_transform(spec, tuple(dense_input))
            if relation.output_start_index is None:
                raise ValidationError(
                    "transformed relations require output_start_index"
                )
            output_start = relation.output_start_index

        target_series = self.term_store.read_range(
            candidate.target, relation.target_start_index, relation.term_count
        )

        checked_count = 0
        first_mismatch: tuple[int, int | None, int | None] | None = None
        missing_index: tuple[str, int] | None = None
        for offset in range(relation.term_count):
            target_index = relation.target_start_index + offset
            target_value = target_series.value_at(target_index)
            if relation.transform_spec is None:
                source_index = relation.source_start_index + offset
                source_value = source_series.value_at(source_index)
            else:
                source_index = output_start + offset
                output_offset = source_index - output_start
                source_value = (
                    output_values[output_offset]
                    if output_offset < len(output_values)
                    else None
                )

            _hash_comparison(
                evidence_hasher,
                source_index,
                target_index,
                source_value,
                target_value,
            )
            if source_value is None:
                missing_index = ("source", source_index)
                break
            if target_value is None:
                missing_index = ("target", target_index)
                break
            checked_count += 1
            if source_value != target_value:
                first_mismatch = (target_index, source_value, target_value)
                break

        evidence_digest = evidence_hasher.hexdigest()
        if missing_index is not None:
            return VerificationResult(
                status="insufficient_data",
                checked_count=checked_count,
                first_mismatch=None,
                missing_index=missing_index,
                evidence_digest=evidence_digest,
                claim=None,
            )
        if first_mismatch is not None:
            return VerificationResult(
                status="mismatch",
                checked_count=checked_count,
                first_mismatch=first_mismatch,
                missing_index=None,
                evidence_digest=evidence_digest,
                claim=None,
            )
        if checked_count != relation.term_count:
            raise RuntimeError("verification ended without checking the full range")

        claim = self._make_claim(candidate, relation, input_count, evidence_digest)
        if persist and self.catalog is not None:
            self._persist_claim(claim)
        return VerificationResult(
            status="exact_match",
            checked_count=checked_count,
            first_mismatch=None,
            missing_index=None,
            evidence_digest=evidence_digest,
            claim=claim,
        )

    def persist_claim(self, claim: RelationClaim) -> None:
        """Persist a previously verified claim through the coordinator catalog."""

        if claim.evidence_level != "E1":
            raise ValidationError("only verified E1 claims may be persisted here")
        self._persist_claim(claim)

    @staticmethod
    def _make_claim(
        candidate: Candidate,
        relation: RelationSpec,
        input_count: int,
        evidence_digest: str,
    ) -> RelationClaim:
        index_map: dict[str, int | str] = {
            "source_start_index": relation.source_start_index,
            "target_start_index": relation.target_start_index,
            "term_count": relation.term_count,
        }
        if relation.transform_spec is not None:
            index_map["transform_output_start_index"] = (
                relation.output_start_index  # type: ignore[assignment]
            )
        parameters: dict[str, str | int] = {
            "input_term_count": input_count,
            "candidate_profile": candidate.profile.profile_key,
            "candidate_digest": candidate.digest,
        }
        if relation.transform_spec is not None:
            parameters["transform_name"] = relation.transform_spec.name
            parameters["transform_version"] = relation.transform_spec.version
            for key, value in relation.transform_spec.parameters.items():
                parameters[f"transform_parameter:{key}"] = (
                    str(value) if type(value) is bool else value
                )

        return RelationClaim(
            relation_kind=relation.relation_kind,
            source=candidate.source,
            target=candidate.target,
            index_map=index_map,
            parameters=parameters,
            compared_start=relation.target_start_index,
            compared_end=relation.target_start_index + relation.term_count - 1,
            evidence_level="E1",
            evidence_digest=evidence_digest,
            transform_spec=relation.transform_spec,
        )

    def _persist_claim(self, claim: RelationClaim) -> None:
        if self.catalog is None or self.run_id is None:
            raise CatalogError("claim persistence requires a catalog and run_id")
        transform_key = None
        with self.catalog.transaction():
            if claim.transform_spec is not None:
                spec = claim.transform_spec
                transform_key = _transform_key(spec)
                self.catalog.execute(
                    """
                    INSERT INTO transform_spec (
                      transform_key, name, implementation_version,
                      canonical_parameters, domain_spec
                    ) VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(transform_key) DO NOTHING
                    """,
                    (
                        transform_key,
                        spec.name,
                        spec.version,
                        json.dumps(
                            dict(spec.parameters),
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                        spec.input_domain,
                    ),
                )
                stored = self.catalog.fetch_one(
                    """
                    SELECT name, implementation_version, canonical_parameters,
                           domain_spec
                    FROM transform_spec WHERE transform_key = ?
                    """,
                    (transform_key,),
                )
                expected = (
                    spec.name,
                    spec.version,
                    json.dumps(
                        dict(spec.parameters),
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    spec.input_domain,
                )
                if stored is None or tuple(stored) != expected:
                    raise CatalogError(
                        "transform key conflicts with stored transform metadata",
                        context={"transform_key": transform_key},
                    )

            claim_id = hashlib.sha256(
                json.dumps(
                    {
                        "relation_kind": claim.relation_kind,
                        "source_revision_id": claim.source.revision_id,
                        "target_revision_id": claim.target.revision_id,
                        "index_map": dict(claim.index_map),
                        "parameters": dict(claim.parameters),
                        "compared_start": claim.compared_start,
                        "compared_end": claim.compared_end,
                        "evidence_digest": claim.evidence_digest,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            self.catalog.execute(
                """
                INSERT INTO relation_claim (
                  claim_id, relation_kind, source_revision_id,
                  target_revision_id, transform_key, index_map_json,
                  parameters_json, compared_start, compared_end,
                  evidence_level, status, evidence_digest, created_by_run,
                  supersedes_claim_id, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?)
                ON CONFLICT(claim_id) DO NOTHING
                """,
                (
                    claim_id,
                    claim.relation_kind,
                    claim.source.revision_id,
                    claim.target.revision_id,
                    transform_key,
                    json.dumps(dict(claim.index_map), sort_keys=True),
                    json.dumps(dict(claim.parameters), sort_keys=True),
                    claim.compared_start,
                    claim.compared_end,
                    claim.evidence_level,
                    "exact_match",
                    claim.evidence_digest,
                    self.run_id,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            stored_claim = self.catalog.fetch_one(
                """
                SELECT evidence_level, evidence_digest, source_revision_id,
                       target_revision_id, transform_key
                FROM relation_claim WHERE claim_id = ?
                """,
                (claim_id,),
            )
            if stored_claim is None or tuple(stored_claim) != (
                claim.evidence_level,
                claim.evidence_digest,
                claim.source.revision_id,
                claim.target.revision_id,
                transform_key,
            ):
                raise CatalogError(
                    "persisted relation claim does not match verified evidence",
                    context={"claim_id": claim_id},
                )


def _hash_comparison(
    hasher: object,
    source_index: int,
    target_index: int,
    source_value: int | None,
    target_value: int | None,
) -> None:
    payload = json.dumps(
        [source_index, target_index, source_value, target_value],
        separators=(",", ":"),
    ).encode("ascii")
    hasher.update(len(payload).to_bytes(8, "big"))  # type: ignore[attr-defined]
    hasher.update(payload)  # type: ignore[attr-defined]


def _denumerant_transform(
    values: tuple[int, ...],
    spec: TransformSpec,
) -> tuple[int, ...]:
    """Adapt the pure Deep Miner extractor to the exact bridge contract."""

    parameters = spec.parameters
    integer_parameters: dict[str, int] = {}
    defaults = {
        "input_start_index": 0,
        "term_cap": 100,
        "confidence_window": 5,
        "input_bit_limit": 2_000,
        "loop_bit_limit": 10_000,
        "multiplicity_limit": 100_000,
        "total_weight_limit": 100_000,
    }
    for name, default in defaults.items():
        value = parameters.get(name, default)
        if type(value) is not int:
            raise TransformError(
                f"denumerant parameter {name!r} must be an integer",
                context={"value": value},
            )
        integer_parameters[name] = value
    extraction = extract_denominations(
        values,
        source_start_index=integer_parameters["input_start_index"],
        term_cap=integer_parameters["term_cap"],
        confidence_window=integer_parameters["confidence_window"],
        input_bit_limit=integer_parameters["input_bit_limit"],
        loop_bit_limit=integer_parameters["loop_bit_limit"],
        multiplicity_limit=integer_parameters["multiplicity_limit"],
        total_weight_limit=integer_parameters["total_weight_limit"],
    )
    return extraction.weights


def _transform_manifest(spec: TransformSpec | None) -> dict[str, object] | None:
    if spec is None:
        return None
    return {
        "name": spec.name,
        "version": spec.version,
        "parameters": dict(spec.parameters),
        "input_domain": spec.input_domain,
        "output_bound": spec.output_bound,
    }


def _transform_key(spec: TransformSpec) -> str:
    canonical = json.dumps(
        _transform_manifest(spec),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()
