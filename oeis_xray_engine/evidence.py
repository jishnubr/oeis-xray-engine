"""Evidence promotion backed by exact, independently reconstructed checks."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Mapping

from .errors import CatalogError, InsufficientTermsError, ValidationError
from .models import SequenceRef
from .storage.catalog import Catalog
from .storage.term_store import TermStore
from .transformers import euler_transform

_MAX_HOLDOUT_DEGREE = 100_000
_EULER_VERIFIER_ID = "oeis_xray_engine.euler_transform@1"


@dataclass(frozen=True, slots=True)
class EvidencePromotion:
    """A persisted higher-level evidence artifact and its promoted claim."""

    claim_id: str
    evidence_level: str
    evidence_digest: str
    range_start: int
    range_end: int
    checked_terms: int


class EvidenceLadder:
    """Promote eligible E1 inverse-Euler claims after a disjoint exact holdout.

    E2 is established by a separate forward Euler reconstruction of the
    target exponent sequence and an exact comparison against source
    coefficients outside the E1 comparison interval. Symbolic-proof (E3) and
    human-review (E4) promotion are intentionally not implemented here:
    neither a proof-producing CAS nor an authenticated review process is
    configured by ADR-001.
    """

    def __init__(self, catalog: Catalog, term_store: TermStore) -> None:
        if not isinstance(catalog, Catalog) or catalog.read_only:
            raise TypeError("EvidenceLadder requires a writable Catalog")
        if not isinstance(term_store, TermStore) or term_store.catalog is not catalog:
            raise ValueError("term_store must use the EvidenceLadder catalog")
        self.catalog = catalog
        self.term_store = term_store

    def verify_e2_euler_holdout(
        self,
        claim_id: str,
        *,
        holdout_start: int,
        holdout_end: int,
    ) -> EvidencePromotion:
        """Verify and persist an E2 inverse-Euler claim on new source degrees.

        The claimed E1 relation must have the canonical index map
        ``a_0..a_N -> c_1..c_N``. The requested source-degree interval must be
        strictly beyond the E1 comparison range. Missing terms and mismatches
        reject promotion without writing any evidence record.
        """

        if not isinstance(claim_id, str) or not claim_id.strip():
            raise ValueError("claim_id must be a non-empty string")
        if type(holdout_start) is not int or type(holdout_end) is not int:
            raise TypeError("holdout range endpoints must be integers")
        if holdout_start < 1 or holdout_end < holdout_start:
            raise ValueError("holdout range must be a positive, ordered interval")
        if holdout_end > _MAX_HOLDOUT_DEGREE:
            raise ValueError(
                f"holdout_end must not exceed {_MAX_HOLDOUT_DEGREE}"
            )

        claim = self._load_claim(claim_id)
        self._validate_inverse_euler_claim(claim)
        if holdout_start <= claim["compared_end"]:
            raise ValidationError(
                "E2 holdout must begin strictly after the E1 comparison range",
                context={
                    "holdout_start": holdout_start,
                    "e1_compared_end": claim["compared_end"],
                },
            )

        source_ref = claim["source_ref"]
        target_ref = claim["target_ref"]
        source_series = self.term_store.read_range(
            source_ref, 0, holdout_end + 1
        )
        target_series = self.term_store.read_range(
            target_ref, 1, holdout_end
        )
        source_values = _dense_values(source_series, 0, holdout_end)
        exponents = _dense_values(target_series, 1, holdout_end)
        reconstructed = euler_transform(exponents, degree=holdout_end)

        for degree in range(holdout_start, holdout_end + 1):
            actual = source_values[degree]
            expected = reconstructed[degree]
            if actual != expected:
                raise ValidationError(
                    "independent Euler reconstruction failed on the E2 holdout",
                    context={
                        "degree": degree,
                        "source_value": actual,
                        "reconstructed_value": expected,
                    },
                )

        source_digest = _digest_indexed_values(0, source_values)
        target_digest = _digest_indexed_values(1, exponents)
        reconstruction_digest = _digest_indexed_values(0, reconstructed)
        evidence_json = json.dumps(
            {
                "claim_id": claim_id,
                "method": "independent-forward-euler",
                "verifier_id": _EULER_VERIFIER_ID,
                "holdout": [holdout_start, holdout_end],
                "transform_key": claim["transform_key"],
                "transform_parameters": claim["transform_parameters"],
                "source_terms_digest": source_digest,
                "target_exponents_digest": target_digest,
                "reconstructed_terms_digest": reconstruction_digest,
                "checked_terms": holdout_end - holdout_start + 1,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        evidence_digest = hashlib.sha256(evidence_json.encode("utf-8")).hexdigest()
        promoted_claim_id = hashlib.sha256(
            json.dumps(
                ["E2", claim_id, evidence_digest],
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        evidence_id = hashlib.sha256(
            json.dumps(
                [promoted_claim_id, "E2", evidence_digest],
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()

        with self.catalog.transaction():
            self.catalog.execute(
                """
                INSERT INTO relation_claim (
                  claim_id, relation_kind, source_revision_id, target_revision_id,
                  transform_key, index_map_json, parameters_json, compared_start,
                  compared_end, evidence_level, status, evidence_digest,
                  created_by_run, supersedes_claim_id, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'E2',
                          'independently_reconstructed', ?, ?, ?, ?)
                ON CONFLICT(claim_id) DO NOTHING
                """,
                (
                    promoted_claim_id,
                    claim["relation_kind"],
                    claim["source_revision_id"],
                    claim["target_revision_id"],
                    claim["transform_key"],
                    claim["index_map_json"],
                    claim["parameters_json"],
                    claim["compared_start"],
                    claim["compared_end"],
                    evidence_digest,
                    claim["created_by_run"],
                    claim_id,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            stored_claim = self.catalog.fetch_one(
                "SELECT evidence_level, supersedes_claim_id, evidence_digest "
                "FROM relation_claim WHERE claim_id = ?",
                (promoted_claim_id,),
            )
            if stored_claim is None or tuple(stored_claim) != (
                "E2",
                claim_id,
                evidence_digest,
            ):
                raise CatalogError("E2 claim identity conflicts with stored data")
            self.catalog.execute(
                """
                INSERT INTO relation_evidence (
                  evidence_id, claim_id, evidence_level, evidence_kind,
                  verifier_id, range_start, range_end, evidence_digest,
                  evidence_json, created_at
                ) VALUES (?, ?, 'E2', 'independent-forward-euler', ?, ?, ?, ?, ?, ?)
                ON CONFLICT(evidence_id) DO NOTHING
                """,
                (
                    evidence_id,
                    promoted_claim_id,
                    _EULER_VERIFIER_ID,
                    holdout_start,
                    holdout_end,
                    evidence_digest,
                    evidence_json,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            stored_evidence = self.catalog.fetch_one(
                """
                SELECT claim_id, evidence_level, evidence_kind, verifier_id,
                       range_start, range_end, evidence_digest, evidence_json
                FROM relation_evidence WHERE evidence_id = ?
                """,
                (evidence_id,),
            )
            if stored_evidence is None or tuple(stored_evidence) != (
                promoted_claim_id,
                "E2",
                "independent-forward-euler",
                _EULER_VERIFIER_ID,
                holdout_start,
                holdout_end,
                evidence_digest,
                evidence_json,
            ):
                raise CatalogError("E2 evidence identity conflicts with stored data")

        return EvidencePromotion(
            claim_id=promoted_claim_id,
            evidence_level="E2",
            evidence_digest=evidence_digest,
            range_start=holdout_start,
            range_end=holdout_end,
            checked_terms=holdout_end - holdout_start + 1,
        )

    def _load_claim(self, claim_id: str) -> Mapping[str, object]:
        row = self.catalog.fetch_one(
            """
            SELECT rc.claim_id, rc.relation_kind, rc.source_revision_id,
                   rc.target_revision_id, rc.transform_key, rc.index_map_json,
                   rc.parameters_json, rc.compared_start, rc.compared_end,
                   rc.evidence_level, rc.created_by_run,
                   source.sequence_id AS source_sequence_id,
                   source.snapshot_id AS source_snapshot_id,
                   target.sequence_id AS target_sequence_id,
                   target.snapshot_id AS target_snapshot_id,
                   ts.name AS transform_name,
                   ts.implementation_version AS transform_version,
                   ts.canonical_parameters AS transform_parameters_json
            FROM relation_claim AS rc
            JOIN sequence_revision AS source
              ON source.revision_id = rc.source_revision_id
            JOIN sequence_revision AS target
              ON target.revision_id = rc.target_revision_id
            LEFT JOIN transform_spec AS ts ON ts.transform_key = rc.transform_key
            WHERE rc.claim_id = ?
            """,
            (claim_id,),
        )
        if row is None:
            raise CatalogError("relation claim does not exist", context={"claim_id": claim_id})
        result: dict[str, object] = dict(row)
        result["index_map"] = json.loads(str(row["index_map_json"]))
        result["parameters"] = json.loads(str(row["parameters_json"]))
        result["transform_parameters"] = (
            json.loads(str(row["transform_parameters_json"]))
            if row["transform_parameters_json"] is not None
            else None
        )
        result["source_ref"] = SequenceRef(
            str(row["source_sequence_id"]),
            str(row["source_revision_id"]),
            str(row["source_snapshot_id"]),
        )
        result["target_ref"] = SequenceRef(
            str(row["target_sequence_id"]),
            str(row["target_revision_id"]),
            str(row["target_snapshot_id"]),
        )
        return result

    @staticmethod
    def _validate_inverse_euler_claim(claim: Mapping[str, object]) -> None:
        index_map = claim["index_map"]
        transform_parameters = claim["transform_parameters"]
        if (
            claim["evidence_level"] != "E1"
            or claim["transform_name"] != "inverse_euler"
            or claim["transform_version"] != "1"
            or not isinstance(index_map, dict)
            or not isinstance(transform_parameters, dict)
            or index_map.get("source_start_index") != 0
            or index_map.get("target_start_index") != 1
            or index_map.get("transform_output_start_index") != 1
            or transform_parameters.get("input_start_index", 0) != 0
            or transform_parameters.get("target_start_index", 1) != 1
            or transform_parameters.get("output_start_index", 1) != 1
            or claim["compared_end"] is None
        ):
            raise ValidationError(
                "E2 forward Euler verification requires a canonical E1 "
                "inverse-Euler claim"
            )


def _dense_values(series, start_index: int, end_index: int) -> tuple[int, ...]:
    """Return a verified dense range or reject incomplete stored data."""

    values: list[int] = []
    for index in range(start_index, end_index + 1):
        value = series.value_at(index)
        if value is None:
            raise InsufficientTermsError(
                "E2 reconstruction range contains a missing term",
                context={"index": index},
            )
        values.append(value)
    return tuple(values)


def _digest_indexed_values(start_index: int, values: tuple[int, ...]) -> str:
    """Hash an exact indexed integer stream without a large JSON buffer."""

    digest = hashlib.sha256()
    for offset, value in enumerate(values):
        payload = json.dumps(
            [start_index + offset, value],
            separators=(",", ":"),
        ).encode("ascii")
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()
