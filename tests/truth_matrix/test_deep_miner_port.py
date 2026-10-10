"""Truth-matrix and integration tests for the Deep Miner port."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from oeis_xray_engine.indexing.sequence_index import SequenceIndex
from oeis_xray_engine.models import ChunkRef, SequenceRef, TermSeries, TransformSpec
from oeis_xray_engine.pipeline import PipelineCoordinator, PipelineSpec
from oeis_xray_engine.storage.catalog import Catalog
from oeis_xray_engine.storage.term_store import TermStore
from oeis_xray_engine.transformers import (
    TransformerExecutionContext,
    default_transformer_registry,
)
from oeis_xray_engine.transformers.denumerant import (
    DenumerantError,
    DenumerantTransformer,
    extract_denominations,
)
from oeis_xray_engine.transformers import euler_transform
from oeis_xray_engine.bfile_validation import (
    BFileFetchResult,
    BFileValidationStatus,
)


class _FixtureFetcher:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.calls: list[str] = []

    def fetch(self, sequence_id: str) -> BFileFetchResult:
        self.calls.append(sequence_id)
        return BFileFetchResult(
            available=True,
            content=self.payload,
            source_url=f"fixture://{sequence_id}",
        )


class DeepMinerPortTests(unittest.TestCase):
    """Check exact extraction, enhanced metadata parsing, and coordinator validation."""

    def test_partition_sequence_extracts_unit_multiplicities_exactly(self) -> None:
        partition_numbers = (
            1, 1, 2, 3, 5, 7, 11, 15, 22, 30, 42, 56, 77, 101, 135,
            176, 231, 297, 385, 490, 627, 792, 1002, 1255, 1575, 1958,
            2436, 3010, 3718, 4565, 5604, 6842,
        )
        result = extract_denominations(
            partition_numbers,
            confidence_window=5,
        )
        self.assertEqual(result.exponents, (1,) * 26)
        self.assertEqual(
            result.reconstructed_coefficients,
            partition_numbers[: result.maximum_weight + 1],
        )
        self.assertEqual(result.source_unit_index, 0)

    def test_denominator_transformer_uses_exact_term_series_contract(self) -> None:
        values = (
            1, 1, 2, 3, 5, 7, 11, 15, 22, 30, 42, 56, 77, 101, 135,
            176, 231, 297, 385, 490, 627, 792, 1002, 1255, 1575, 1958,
            2436, 3010, 3718, 4565, 5604, 6842,
        )
        ref = SequenceRef("A000041", "revision", "truth")
        spec = TransformSpec(
            name="denumerant",
            version="1",
            parameters={
                "input_start_index": 0,
                "output_start_index": 1,
                "confidence_window": 5,
            },
            input_domain="ordinary_generating_function",
            output_bound=30,
        )
        context = TransformerExecutionContext(
            input_series=TermSeries(
                ref=ref,
                start_index=0,
                values=values,
                complete_through=len(values) - 1,
            ),
            spec=spec,
            metadata={"output_start_index": 1},
        )
        output = DenumerantTransformer().transform(context)
        self.assertEqual(output.start_index, 1)
        self.assertEqual(output.values, tuple(range(1, 27)))

    def test_invalid_or_non_bosonic_inputs_are_rejected(self) -> None:
        with self.assertRaises(DenumerantError):
            extract_denominations((1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987))
        with self.assertRaises(DenumerantError):
            extract_denominations((1, 1, 2))

    def test_generated_finite_euler_product_round_trips_exactly(self) -> None:
        coefficients = euler_transform((1, 0, 2) + (0,) * 37, degree=40)
        result = extract_denominations(coefficients, source_start_index=7)
        self.assertEqual(result.weights, (1, 3, 3))
        self.assertEqual(result.exponents[:3], (1, 0, 2))
        self.assertEqual(result.source_unit_index, 7)

    def test_default_transformer_registry_exposes_denumerant(self) -> None:
        transformer = default_transformer_registry().resolve("denumerant", "1")
        self.assertIsInstance(transformer, DenumerantTransformer)

    def test_denumerant_runs_through_the_registered_pipeline_worker(self) -> None:
        values = (
            1, 1, 2, 3, 5, 7, 11, 15, 22, 30, 42, 56, 77, 101, 135,
            176, 231, 297, 385, 490, 627, 792, 1002, 1255, 1575, 1958, 2436,
            3010, 3718, 4565, 5604, 6842,
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            catalog = Catalog(root / "catalog.sqlite")
            try:
                store = TermStore(root / "terms", catalog)
                index = SequenceIndex(catalog, store, chunk_terms=8)
                index.ingest_lines(
                    iter(
                        (
                            "%I A000041 M0000\n",
                            "%S A000041 " + ",".join(map(str, values[:6])) + "\n",
                            "%T A000041 " + ",".join(map(str, values[6:])) + "\n",
                            "%O A000041 0\n",
                        )
                    ),
                    snapshot_id="denumerant-worker",
                    source_revision="partition-fixture-v1",
                )
                transform = TransformSpec(
                    name="denumerant",
                    version="1",
                    parameters={
                        "input_start_index": 0,
                        "input_term_count": len(values),
                        "output_start_index": 1,
                        "confidence_window": 5,
                    },
                    input_domain="ordinary_generating_function",
                    output_bound=100_000,
                )
                coordinator = PipelineCoordinator(catalog, store)
                run_id = coordinator.plan(
                    PipelineSpec(
                        run_id="denumerant-worker-run",
                        snapshot_id="denumerant-worker",
                        stages=("transform",),
                        transform_specs=(transform,),
                        page_size=1,
                    )
                )
                report = coordinator.execute(run_id, workers=2)
                self.assertEqual(report.state, "complete")
                self.assertEqual(report.counts.complete, 1)

                manifest_row = catalog.fetch_one(
                    """
                    SELECT term_manifest_json FROM derived_artifact
                    WHERE created_by_run = ?
                    """,
                    (run_id,),
                )
                manifest = json.loads(manifest_row["term_manifest_json"])
                output_ref = SequenceRef(
                    f"XRAY-{manifest['revision_id'][:24].upper()}",
                    manifest["revision_id"],
                    "denumerant-worker",
                )
                terms = TermStore(root / "terms").read_chunk(
                    ChunkRef(
                        chunk_id=manifest["chunk_id"],
                        revision_id=manifest["revision_id"],
                        first_index=manifest["first_index"],
                        term_count=manifest["term_count"],
                        last_index=manifest["last_index"],
                        codec=manifest["codec"],
                        format_version=manifest["format_version"],
                        uncompressed_bytes=manifest["uncompressed_bytes"],
                        payload_digest=manifest["payload_digest"],
                        content_path=manifest["content_path"],
                        bundle_id=manifest.get("bundle_id"),
                        byte_offset=manifest.get("byte_offset"),
                        byte_length=manifest.get("byte_length"),
                    ),
                    output_ref,
                )
                self.assertEqual(terms.start_index, 1)
                self.assertEqual(terms.values, tuple(range(1, 27)))
            finally:
                catalog.close()

    def test_sequence_index_preserves_deep_miner_metadata_and_offset_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            catalog = Catalog(root / "catalog.sqlite")
            try:
                store = TermStore(root / "terms", catalog)
                index = SequenceIndex(catalog, store, chunk_terms=3)
                report = index.ingest_lines(
                    iter(
                        (
                            "%I A000041 M0697 # Jul 2026\n",
                            "%N A000041 Partitions\n",
                            "%F A000041 a(n) = number of partitions of n.\n",
                            "%C A000041 Also called partition numbers.\n",
                            "%K A000041 core,nonn\n",
                            "%Y A000041 Cf. A000009.\n",
                            "%H A000041 <a href=\"b000041.txt\">B-file</a>\n",
                            "%S A000041 1,1,2,3,5,7\n",
                            "%T A000041 11,15,22,30\n",
                            "%O A000041 0,1\n",
                        )
                    ),
                    snapshot_id="deep-miner-parser",
                    source_revision="deep-miner-fixture-v1",
                )
                self.assertEqual(report.parsed, 1)
                descriptor = next(index.iter_sequences("deep-miner-parser"))
                revision = index.get_revision(descriptor.ref)
                raw = json.loads(revision.raw_metadata_json)
                self.assertEqual(revision.offset, 0)
                self.assertEqual(raw["%N"], ["A000041 Partitions"])
                self.assertEqual(raw["%F"], ["A000041 a(n) = number of partitions of n."])
                self.assertEqual(raw["%K"], ["A000041 core,nonn"])
                self.assertEqual(raw["%Y"], ["A000041 Cf. A000009."])
                self.assertEqual(raw["%H"], ['A000041 <a href="b000041.txt">B-file</a>'])
            finally:
                catalog.close()

    def test_coordinator_calls_external_fetcher_and_persists_orthogonal_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            catalog = Catalog(root / "catalog.sqlite")
            try:
                store = TermStore(root / "terms", catalog)
                index = SequenceIndex(catalog, store)
                index.ingest_lines(
                    iter(
                        (
                            "%I A000041 M0000\n",
                            "%S A000041 1,1,2,3,5,7,11,15,22,30,42\n",
                            "%O A000041 0\n",
                        )
                    ),
                    snapshot_id="bfile-fixture",
                    source_revision="fixture-v1",
                )
                ref = next(index.iter_sequences("bfile-fixture")).ref
                fetcher = _FixtureFetcher(
                    b"# fixture b-file\n0 1\n1 1\n2 2\n3 3\n4 5\n5 7\n"
                    b"6 11\n7 15\n8 22\n9 30\n10 42\n"
                )

                coordinator = PipelineCoordinator(catalog, store)
                result = coordinator.validate_bfile(ref, fetcher, max_terms=20)

                self.assertEqual(fetcher.calls, ["A000041"])
                self.assertEqual(result.status, BFileValidationStatus.MATCH)
                self.assertEqual(result.terms_checked, 11)
                self.assertEqual(result.compared_start, 0)
                self.assertEqual(result.compared_end, 10)
                candidate_result = coordinator.validate_denumerant_bfile(
                    ref,
                    tuple(range(1, 11)),
                    0,
                    fetcher,
                    max_terms=20,
                )
                self.assertEqual(
                    candidate_result.status,
                    BFileValidationStatus.MATCH,
                )
                self.assertEqual(candidate_result.validation_kind.value, "denumerant")
                self.assertNotEqual(
                    candidate_result.comparison_digest,
                    result.comparison_digest,
                )
                self.assertEqual(
                    catalog.fetch_one(
                        """
                        SELECT validation_status, terms_checked, source_url
                        FROM bfile_validation
                        WHERE revision_id = ? AND validation_kind = 'sequence'
                        """,
                        (ref.revision_id,),
                    )["validation_status"],
                    "match",
                )
                self.assertEqual(
                    catalog.fetch_one(
                        """
                        SELECT validation_status FROM bfile_validation
                        WHERE revision_id = ? AND validation_kind = 'denumerant'
                        """,
                        (ref.revision_id,),
                    )["validation_status"],
                    "match",
                )
                self.assertEqual(
                    catalog.fetch_all(
                        "SELECT claim_id FROM relation_claim WHERE source_revision_id = ?",
                        (ref.revision_id,),
                    ),
                    [],
                )
            finally:
                catalog.close()

    def test_bfile_mismatch_and_out_of_range_are_distinct(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            catalog = Catalog(root / "catalog.sqlite")
            try:
                store = TermStore(root / "terms", catalog)
                index = SequenceIndex(catalog, store)
                index.ingest_lines(
                    iter(
                        (
                            "%I A000012 M0000\n",
                            "%S A000012 1,1,1\n",
                            "%O A000012 0\n",
                        )
                    ),
                    snapshot_id="bfile-negative",
                    source_revision="fixture-v1",
                )
                ref = next(index.iter_sequences("bfile-negative")).ref
                coordinator = PipelineCoordinator(catalog, store)
                mismatch = coordinator.validate_bfile(
                    ref,
                    _FixtureFetcher(b"0 1\n1 9\n"),
                )
                self.assertEqual(mismatch.status, BFileValidationStatus.MISMATCH)
                self.assertEqual(mismatch.first_mismatch_index, 1)

                outside = coordinator.validate_bfile(
                    ref,
                    _FixtureFetcher(b"100 1\n101 2\n"),
                )
                self.assertEqual(
                    outside.status,
                    BFileValidationStatus.INSUFFICIENT_DATA,
                )
            finally:
                catalog.close()

    def test_bfile_unavailable_and_malformed_are_not_mismatches(self) -> None:
        from oeis_xray_engine.bfile_validation import BFileValidationService

        service = BFileValidationService()
        unavailable = service.validate(
            BFileFetchResult(
                available=False,
                content=None,
                error_message="fixture offline",
            ),
            None,
        )
        malformed = service.validate(
            BFileFetchResult(
                available=True,
                content=b"0 1\nnot-a-record\n",
            ),
            None,
        )
        self.assertEqual(unavailable.status, BFileValidationStatus.UNAVAILABLE)
        self.assertEqual(malformed.status, BFileValidationStatus.MALFORMED)

    def test_bfile_comparison_uses_declared_indices_not_first_unit_alignment(self) -> None:
        from oeis_xray_engine.bfile_validation import BFileValidationService

        ref = SequenceRef("A000123", "revision", "snapshot")
        series = TermSeries(
            ref=ref,
            start_index=10,
            values=(7, 1, 2),
            complete_through=12,
        )
        result = BFileValidationService().validate(
            BFileFetchResult(
                available=True,
                content=b"10 7\n11 1\n12 2\n",
            ),
            series,
        )
        self.assertEqual(result.status, BFileValidationStatus.MATCH)
        self.assertEqual((result.compared_start, result.compared_end), (10, 12))
        self.assertEqual(result.terms_checked, 3)


if __name__ == "__main__":
    unittest.main()
