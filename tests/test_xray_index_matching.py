"""Focused integration verification for X-Ray indexing and matching."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from oeis_xray_engine.errors import CatalogError, ValidationError
from oeis_xray_engine.evidence import EvidenceLadder
from oeis_xray_engine.indexing import Candidate, FingerprintProfile, HashIndex
from oeis_xray_engine.matching import (
    BridgeFinder,
    CollisionDetector,
    RelationSpec,
)
from oeis_xray_engine.models import SequenceRef, TransformSpec
from oeis_xray_engine.storage.catalog import Catalog
from oeis_xray_engine.storage.term_store import TermStore
from oeis_xray_engine.transformers import euler_transform


class XRayIndexMatchingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.catalog = Catalog(self.root / "catalog.sqlite")
        self.catalog.execute(
            """
            INSERT INTO corpus_snapshot (
              snapshot_id, source_name, source_revision, source_digest,
              created_at, parser_version, state
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            ("snapshot-test", "test", "revision-1", "source-digest", "now", "p1", "ready"),
        )
        self.catalog.execute(
            """
            INSERT INTO run (
              run_id, snapshot_id, run_kind, config_digest, code_revision,
              state, manifest_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            ("run-test", "snapshot-test", "test", "config-digest", "code", "running", "{}"),
        )
        self.term_store = TermStore(self.root / "term-store", self.catalog)
        self.hash_index = HashIndex(self.catalog, self.term_store, "run-test")
        self.detector = CollisionDetector(
            self.term_store, self.catalog, run_id="run-test"
        )

    def tearDown(self) -> None:
        self.catalog.close()
        self.temp.cleanup()

    def add_sequence(
        self,
        sequence_id: str,
        revision_id: str,
        values: list[int],
        *,
        offset: int,
    ) -> SequenceRef:
        ref = SequenceRef(sequence_id, revision_id, "snapshot-test")
        self.catalog.execute(
            "INSERT INTO sequence VALUES (?, ?, ?)",
            (sequence_id, revision_id, "snapshot-test"),
        )
        self.catalog.execute(
            """
            INSERT INTO sequence_revision (
              revision_id, sequence_id, snapshot_id, source_digest, term_digest,
              offset, term_count, known_end_index, completeness, parse_status,
              raw_metadata_json, diagnostics_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                revision_id,
                sequence_id,
                "snapshot-test",
                f"source-{revision_id}",
                f"terms-{revision_id}",
                offset,
                len(values),
                offset + len(values) - 1,
                "finite",
                "valid",
                "{}",
                "[]",
            ),
        )
        self.term_store.put_chunk(ref, offset, values)
        return ref

    def test_prefix_hash_is_e0_and_exact_verifier_promotes_to_e1(self) -> None:
        left = self.add_sequence("A000001", "rev-a", [4, 8, 15, 16, 23], offset=0)
        right = self.add_sequence("A000002", "rev-b", [4, 8, 15, 16, 999], offset=0)
        profile = FingerprintProfile("test-prefix", "1", "exact", 4, start_index=0)

        left_fingerprint = self.hash_index.fingerprint(left, profile)
        self.hash_index.fingerprint(right, profile)
        candidates = list(self.hash_index.find_candidates(left_fingerprint))
        self.assertEqual(len(candidates), 1)
        candidate = candidates[0]
        self.assertEqual(candidate.evidence_level, "E0")
        self.assertEqual(
            self.catalog.fetch_one(
                """
                SELECT evidence_level FROM relation_claim
                WHERE source_revision_id = ? AND target_revision_id = ?
                ORDER BY evidence_level LIMIT 1
                """,
                (candidate.source.revision_id, candidate.target.revision_id),
            )["evidence_level"],
            "E0",
        )

        result = self.detector.verify(
            candidate,
            RelationSpec(
                relation_kind="equal-prefix",
                source_start_index=0,
                target_start_index=0,
                term_count=4,
            ),
        )
        self.assertEqual(result.status, "exact_match")
        self.assertEqual(result.checked_count, 4)
        self.assertIsNotNone(result.claim)
        self.assertEqual(result.claim.evidence_level, "E1")
        persisted = self.catalog.fetch_one(
            "SELECT evidence_level FROM relation_claim WHERE evidence_digest = ?",
            (result.evidence_digest,),
        )
        self.assertIsNotNone(persisted)
        self.assertEqual(persisted["evidence_level"], "E1")

    def test_hash_collision_with_trailing_mismatch_is_not_promoted(self) -> None:
        left = self.add_sequence("A000003", "rev-c", [1, 2, 3, 4, 5], offset=0)
        right = self.add_sequence("A000004", "rev-d", [1, 2, 3, 4, 6], offset=0)
        profile = FingerprintProfile("forced-collision", "1", "exact", 5, 0)

        # Construct an E0 collision candidate directly to model a digest
        # collision or corrupt candidate index. The verifier must trust no hash.
        candidate = Candidate(
            source=left,
            target=right,
            profile=profile,
            range_start=0,
            range_end=4,
            digest="deliberately-colliding-digest",
        )
        self.hash_index.record_candidate(candidate)
        result = self.detector.verify(
            candidate,
            RelationSpec("equal-sequence", 0, 0, 5),
        )
        self.assertEqual(result.status, "mismatch")
        self.assertEqual(result.first_mismatch, (4, 5, 6))
        self.assertIsNone(result.claim)
        self.assertIsNone(
            self.catalog.fetch_one(
                """
                SELECT 1 FROM relation_claim
                WHERE source_revision_id = ? AND target_revision_id = ?
                  AND evidence_level = 'E1'
                """,
                (left.revision_id, right.revision_id),
            )
        )
        self.assertIsNotNone(
            self.catalog.fetch_one(
                """
                SELECT 1 FROM relation_claim
                WHERE source_revision_id = ? AND target_revision_id = ?
                  AND evidence_level = 'E0'
                """,
                (left.revision_id, right.revision_id),
            )
        )

    def test_abs_and_difference_profiles_are_retrieval_only_normalizations(self) -> None:
        positive = self.add_sequence("A000009", "rev-i", [2, -3, 7], offset=0)
        negative = self.add_sequence("A000010", "rev-j", [-2, 3, -7], offset=0)
        shifted = self.add_sequence("A000011", "rev-k", [12, 13, 14], offset=0)
        base = self.add_sequence("A000012", "rev-l", [2, 3, 4], offset=0)

        abs_profile = FingerprintProfile("abs-test", "1", "abs", 3, 0)
        abs_fingerprint = self.hash_index.fingerprint(positive, abs_profile)
        self.hash_index.fingerprint(negative, abs_profile)
        self.assertEqual(
            [candidate.target for candidate in self.hash_index.find_candidates(abs_fingerprint)],
            [negative],
        )

        diff_profile = FingerprintProfile("diff-test", "1", "diff", 3, 0)
        diff_fingerprint = self.hash_index.fingerprint(base, diff_profile)
        self.hash_index.fingerprint(shifted, diff_profile)
        self.assertEqual(
            [candidate.target for candidate in self.hash_index.find_candidates(diff_fingerprint)],
            [shifted],
        )

    def test_sparse_missing_term_returns_insufficient_data(self) -> None:
        left = self.add_sequence("A000005", "rev-e", [10, 30], offset=0)
        right = self.add_sequence("A000006", "rev-f", [10, 20, 30], offset=0)
        # Remove the first sequence's only chunk and replace it with two
        # one-term chunks separated by a known missing index.
        self.catalog.execute("DELETE FROM term_chunk WHERE revision_id = ?", ("rev-e",))
        self.term_store.put_chunk(left, 0, [10])
        self.term_store.put_chunk(left, 2, [30])
        profile = FingerprintProfile("sparse-test", "1", "exact", 3, 0)
        candidate = Candidate(left, right, profile, 0, 2, "e0")
        result = self.detector.verify(
            candidate,
            RelationSpec("equal-sequence", 0, 0, 3),
        )
        self.assertEqual(result.status, "insufficient_data")
        self.assertEqual(result.missing_index, ("source", 1))
        self.assertIsNone(result.claim)

    def test_inverse_euler_bridge_emits_e1_with_transform_spec(self) -> None:
        exponents = [1, 0, 2, -1, 1, 3, 0, 2, -2, 1]
        coefficients = euler_transform(exponents)
        source = self.add_sequence(
            "A000007", "rev-source", coefficients, offset=0
        )
        target = self.add_sequence(
            "A000008", "rev-target", exponents, offset=1
        )
        spec = TransformSpec(
            name="inverse_euler",
            version="1",
            parameters={
                "input_start_index": 0,
                "input_term_count": len(exponents) + 1,
                "output_start_index": 1,
                "target_start_index": 1,
            },
            input_domain="formal_series_constant_term_one",
            output_bound=len(exponents),
        )
        finder = BridgeFinder(self.term_store, self.hash_index, self.detector)
        results = list(
            finder.find_bridges(
                [source],
                [target],
                [spec],
                compare_terms=len(exponents),
            )
        )
        self.assertEqual(len(results), 1)
        result = results[0]
        self.assertEqual(result.candidate.evidence_level, "E0")
        self.assertEqual(result.verification.status, "exact_match")
        claim = result.verification.claim
        self.assertIsNotNone(claim)
        self.assertEqual(claim.evidence_level, "E1")
        self.assertEqual(claim.transform_spec, spec)
        self.assertEqual(claim.compared_start, 1)
        self.assertEqual(claim.compared_end, len(exponents))
        stored = self.catalog.fetch_one(
            """
            SELECT rc.evidence_level, ts.name, ts.implementation_version,
                   ts.canonical_parameters, ts.domain_spec
            FROM relation_claim AS rc
            JOIN transform_spec AS ts ON ts.transform_key = rc.transform_key
            WHERE rc.evidence_digest = ?
            """,
            (result.verification.evidence_digest,),
        )
        self.assertIsNotNone(stored)
        self.assertEqual(stored["evidence_level"], "E1")
        self.assertEqual(stored["name"], "inverse_euler")
        self.assertEqual(stored["implementation_version"], "1")

    def test_denumerant_bridge_emits_e1_against_indexed_weights(self) -> None:
        source = self.add_sequence(
            "A000012", "rev-ones", [1] * 20, offset=0
        )
        target = self.add_sequence(
            "A000009", "rev-weights", [1], offset=1
        )
        spec = TransformSpec(
            name="denumerant",
            version="1",
            parameters={
                "input_start_index": 0,
                "input_term_count": 20,
                "term_cap": 20,
                "confidence_window": 5,
                "target_start_index": 1,
                "output_start_index": 1,
            },
            input_domain="finite_integer_sequence_pure_bosonic_euler_product",
            output_bound=1,
        )
        finder = BridgeFinder(self.term_store, self.hash_index, self.detector)
        results = list(
            finder.find_bridges(
                [source],
                [target],
                [spec],
                compare_terms=1,
            )
        )
        self.assertEqual(len(results), 1)
        result = results[0]
        self.assertEqual(result.candidate.evidence_level, "E0")
        self.assertEqual(result.verification.status, "exact_match")
        self.assertIsNotNone(result.verification.claim)
        claim = result.verification.claim
        self.assertEqual(claim.evidence_level, "E1")
        self.assertEqual(claim.transform_spec, spec)
        self.assertEqual(claim.compared_start, 1)
        self.assertEqual(claim.compared_end, 1)

    def test_inverse_euler_bridge_can_be_independently_reconstructed_on_holdout(
        self,
    ) -> None:
        coefficients = euler_transform([1] * 15, degree=15)
        source = self.add_sequence(
            "A000041", "rev-partitions", coefficients, offset=0
        )
        target = self.add_sequence(
            "A000012", "rev-exponents", [1] * 15, offset=1
        )
        spec = TransformSpec(
            name="inverse_euler",
            version="1",
            parameters={
                "input_start_index": 0,
                "input_term_count": 9,
                "output_start_index": 1,
                "target_start_index": 1,
            },
            input_domain="formal_series_constant_term_one",
            output_bound=8,
        )
        results = list(
            BridgeFinder(self.term_store, self.hash_index, self.detector).find_bridges(
                [source],
                [target],
                [spec],
                compare_terms=8,
            )
        )
        self.assertEqual(len(results), 1)
        initial_claim = results[0].verification.claim
        self.assertIsNotNone(initial_claim)
        row = self.catalog.fetch_one(
            "SELECT claim_id FROM relation_claim WHERE evidence_digest = ?",
            (initial_claim.evidence_digest,),
        )
        self.assertIsNotNone(row)

        promotion = EvidenceLadder(self.catalog, self.term_store).verify_e2_euler_holdout(
            row["claim_id"],
            holdout_start=9,
            holdout_end=15,
        )
        self.assertEqual(promotion.evidence_level, "E2")
        self.assertEqual(promotion.checked_terms, 7)
        promoted = self.catalog.fetch_one(
            "SELECT evidence_level, supersedes_claim_id FROM relation_claim "
            "WHERE claim_id = ?",
            (promotion.claim_id,),
        )
        self.assertEqual(promoted["evidence_level"], "E2")
        self.assertEqual(promoted["supersedes_claim_id"], row["claim_id"])
        evidence = self.catalog.fetch_one(
            "SELECT evidence_level, evidence_kind, range_start, range_end "
            "FROM relation_evidence WHERE claim_id = ?",
            (promotion.claim_id,),
        )
        self.assertEqual(evidence["evidence_level"], "E2")
        self.assertEqual(evidence["evidence_kind"], "independent-forward-euler")
        self.assertEqual((evidence["range_start"], evidence["range_end"]), (9, 15))

    def test_e2_reconstruction_rejects_a_holdout_mismatch(self) -> None:
        coefficients = euler_transform([1] * 15, degree=15)
        altered = list(coefficients)
        altered[12] += 1
        source = self.add_sequence(
            "A000041", "rev-altered-partitions", altered, offset=0
        )
        target = self.add_sequence(
            "A000012", "rev-altered-exponents", [1] * 15, offset=1
        )
        spec = TransformSpec(
            name="inverse_euler",
            version="1",
            parameters={
                "input_start_index": 0,
                "input_term_count": 9,
                "output_start_index": 1,
                "target_start_index": 1,
            },
            input_domain="formal_series_constant_term_one",
            output_bound=8,
        )
        result = next(
            BridgeFinder(self.term_store, self.hash_index, self.detector).find_bridges(
                [source],
                [target],
                [spec],
                compare_terms=8,
            )
        )
        claim_row = self.catalog.fetch_one(
            "SELECT claim_id FROM relation_claim WHERE evidence_digest = ?",
            (result.verification.evidence_digest,),
        )
        self.assertIsNotNone(claim_row)

        with self.assertRaises(ValidationError):
            EvidenceLadder(
                self.catalog, self.term_store
            ).verify_e2_euler_holdout(
                claim_row["claim_id"],
                holdout_start=9,
                holdout_end=15,
            )
        evidence = self.catalog.fetch_one(
            "SELECT COUNT(*) AS count FROM relation_evidence"
        )
        self.assertEqual(evidence["count"], 0)


if __name__ == "__main__":
    unittest.main()
