"""Integration tests for streaming ingestion and resumable multiprocessing."""

from __future__ import annotations

import json
import tempfile
import time
import tracemalloc
import unittest
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from queue import Queue
from types import SimpleNamespace
from unittest.mock import patch

from oeis_xray_engine.checkpoint import CheckpointStore
from oeis_xray_engine.errors import CorruptChunkError
from oeis_xray_engine.indexing import Fingerprint, FingerprintProfile, HashIndex
from oeis_xray_engine.indexing.sequence_index import (
    Partition,
    SequenceIndex,
)
from oeis_xray_engine.matching import BridgeFinder, CollisionDetector
from oeis_xray_engine.models import SequenceRef, TermSeries, TransformSpec
from oeis_xray_engine.pipeline import (
    PipelineCoordinator,
    PipelineSpec,
    _heartbeat_listener,
)
from oeis_xray_engine.storage.catalog import Catalog
from oeis_xray_engine.storage.term_store import TermStore
from oeis_xray_engine.transformers import TransformerExecutionContext, euler_transform
from oeis_xray_engine.transformers.builtins import InverseEulerTransformer
from oeis_xray_engine.indexing.hash_index import FingerprintProfile


def _oeis_record(sequence_id: str, terms: list[int], offset: int = 0) -> str:
    fields = [f"%I {sequence_id} M0000"]
    fields.append(f"%S {sequence_id} " + ",".join(map(str, terms[:4])))
    if len(terms) > 4:
        fields.append(f"%T {sequence_id} " + ",".join(map(str, terms[4:]))
                      )
    fields.append(f"%O {sequence_id} {offset}")
    fields.append(f"%N {sequence_id} test fixture")
    return "\n".join(fields) + "\n"


class XRayPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.catalog = Catalog(self.root / "catalog.sqlite")
        self.term_store = TermStore(self.root / "terms", self.catalog)
        self.extra_catalogs: list[Catalog] = []

    def tearDown(self) -> None:
        for catalog in self.extra_catalogs:
            catalog.close()
        self.catalog.close()
        self.temp.cleanup()

    def test_directory_ingestion_persists_descriptors_revisions_and_chunks(self) -> None:
        source = self.root / "oeis"
        (source / "0000").mkdir(parents=True)
        (source / "0000" / "A000001.seq").write_text(
            "%I A000001 M0000\n"
            "%S A000001 0,1,1\n"
            "%T A000001 2,-5\n"
            "%O A000001 0,1\n"
            "%N A000001 fixture sequence\n",
            encoding="utf-8",
        )
        index = SequenceIndex(
            self.catalog,
            self.term_store,
            chunk_terms=2,
            page_size=1,
        )
        report = index.ingest_directory(
            source,
            snapshot_id="snapshot-ingest",
            source_revision="fixture-v1",
        )
        self.assertEqual(report.records_seen, 1)
        self.assertEqual(report.added, 1)
        self.assertEqual(report.parsed, 1)
        self.assertEqual(report.terms_ingested, 5)
        self.assertEqual(report.chunks_written, 3)

        descriptors = list(
            index.iter_sequences("snapshot-ingest", Partition(page_size=1))
        )
        self.assertEqual(len(descriptors), 1)
        descriptor = descriptors[0]
        self.assertEqual(descriptor.ref.sequence_id, "A000001")
        self.assertEqual(descriptor.offset, 0)
        self.assertEqual(descriptor.term_count, 5)
        self.assertEqual(descriptor.parse_status, "valid")

        revision = index.get_revision(descriptor.ref)
        self.assertEqual(revision.ref, descriptor.ref)
        self.assertEqual(revision.term_count, descriptor.term_count)
        self.assertEqual(json.loads(revision.raw_metadata_json)["%O"], ["A000001 0,1"])

        chunks = list(self.term_store.iter_chunks(descriptor.ref))
        self.assertEqual([chunk.term_count for chunk in chunks], [2, 2, 1])
        self.assertEqual(len({chunk.bundle_id for chunk in chunks}), 1)
        self.assertEqual(
            [
                left.byte_offset + left.byte_length
                for left in chunks[:-1]
            ],
            [chunk.byte_offset for chunk in chunks[1:]],
        )
        first_chunk = chunks[0]
        self.assertIsNotNone(first_chunk.bundle_id)
        self.assertEqual(first_chunk.byte_offset, 0)
        self.assertGreater(first_chunk.byte_length, 0)
        self.assertTrue(self.term_store.verify(first_chunk).valid)
        tampered_chunk = replace(first_chunk, payload_digest="0" * 64)
        with self.assertRaises(CorruptChunkError):
            self.term_store.read_chunk(tampered_chunk, descriptor.ref)
        terms = self.term_store.read_range(descriptor.ref, 0, 5)
        self.assertEqual(tuple(terms.values), (0, 1, 1, 2, -5))
        self.assertEqual(self.catalog.integrity_check(), ["ok"])

    def test_directory_ingestion_persists_chunks_before_scanning_all_files(self) -> None:
        source = (self.root / "oeis-immediate").resolve()
        source.mkdir()
        first = (source / "A000001.seq").resolve()
        second = (source / "A000002.seq").resolve()
        first.write_text(_oeis_record("A000001", [1, 2]), encoding="utf-8")
        second.write_text(_oeis_record("A000002", [3, 4]), encoding="utf-8")
        index = SequenceIndex(self.catalog, self.term_store, chunk_terms=1)
        chunks_seen_between_files: list[int] = []

        def file_stream(_root: Path):
            del _root
            yield first
            row = self.catalog.fetch_one("SELECT COUNT(*) AS n FROM term_chunk")
            chunks_seen_between_files.append(row["n"])
            yield second

        with (
            patch.object(index, "_iter_seq_files", side_effect=file_stream),
            patch.object(
                index,
                "_digest_directory",
                side_effect=AssertionError("directory must not be pre-scanned"),
            ),
        ):
            report = index.ingest_directory(
                source,
                snapshot_id="snapshot-immediate",
                source_revision="immediate-fixture-v1",
            )

        self.assertEqual(chunks_seen_between_files, [2])
        self.assertEqual(report.records_seen, 2)
        self.assertEqual(report.chunks_written, 4)
        self.assertEqual(
            report.source_digest,
            SequenceIndex._digest_directory(source),
        )
        snapshot = self.catalog.fetch_one(
            "SELECT source_digest, state FROM corpus_snapshot WHERE snapshot_id = ?",
            ("snapshot-immediate",),
        )
        self.assertEqual(snapshot["source_digest"], report.source_digest)
        self.assertEqual(snapshot["state"], "ready")

    def test_interrupted_directory_ingestion_resumes_existing_revisions(self) -> None:
        source = (self.root / "oeis-resume").resolve()
        source.mkdir()
        first = source / "A000001.seq"
        second = source / "A000002.seq"
        first.write_text(_oeis_record("A000001", [1, 2]), encoding="utf-8")
        second.write_text(_oeis_record("A000002", [3, 4]), encoding="utf-8")
        index = SequenceIndex(self.catalog, self.term_store, chunk_terms=1)

        def interrupted_files(_root: Path):
            del _root
            yield first
            raise RuntimeError("simulated interruption between files")

        with patch.object(index, "_iter_seq_files", side_effect=interrupted_files):
            with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                index.ingest_directory(
                    source,
                    snapshot_id="snapshot-resume",
                    source_revision="resume-fixture-v1",
                )

        snapshot = self.catalog.fetch_one(
            "SELECT state FROM corpus_snapshot WHERE snapshot_id = ?",
            ("snapshot-resume",),
        )
        self.assertEqual(snapshot["state"], "ingesting")
        self.assertEqual(
            self.catalog.fetch_one("SELECT COUNT(*) AS n FROM term_chunk")["n"],
            2,
        )

        report = index.ingest_directory(
            source,
            snapshot_id="snapshot-resume",
            source_revision="resume-fixture-v1",
        )

        self.assertEqual(report.records_seen, 2)
        self.assertEqual(report.unchanged, 1)
        self.assertEqual(report.added, 1)
        snapshot = self.catalog.fetch_one(
            "SELECT source_digest, state FROM corpus_snapshot WHERE snapshot_id = ?",
            ("snapshot-resume",),
        )
        self.assertEqual(snapshot["source_digest"], report.source_digest)
        self.assertEqual(snapshot["state"], "ready")

    def test_raw_stream_ingestion_is_idempotent_and_records_partial_offsets(self) -> None:
        index = SequenceIndex(self.catalog, self.term_store, chunk_terms=2)
        lines = [
            "%I A000010 M0000\n",
            "%S A000010 3,4,5\n",
            "%O A000010 7\n",
            "%I A000011 M0000\n",
            "%S A000011 8,9\n",
        ]
        first = index.ingest_lines(
            iter(lines),
            snapshot_id="snapshot-stream",
            source_revision="stream-v1",
        )
        second = index.ingest_lines(
            iter(lines),
            snapshot_id="snapshot-stream",
            source_revision="stream-v1",
        )
        self.assertEqual(first.records_seen, 2)
        self.assertEqual(first.partial, 1)
        self.assertEqual(first.terms_ingested, 3)
        self.assertEqual(second.unchanged, 2)
        descriptors = list(index.iter_sequences("snapshot-stream"))
        self.assertEqual([item.parse_status for item in descriptors], ["valid", "partial"])
        self.assertEqual(descriptors[1].term_count, 0)
        self.assertEqual(
            self.catalog.fetch_one(
                "SELECT COUNT(*) AS n FROM term_chunk WHERE revision_id = ?",
                (descriptors[1].ref.revision_id,),
            )["n"],
            0,
        )

    def test_ingestion_streams_term_metadata_and_accepts_arbitrary_precision(self) -> None:
        digits = "9" * 5000
        index = SequenceIndex(self.catalog, self.term_store, chunk_terms=1)
        report = index.ingest_lines(
            iter(
                [
                    "%I A000012 M0000\n",
                    f"%S A000012 {digits},-0\n",
                    "%O A000012 0\n",
                ]
            ),
            snapshot_id="snapshot-large-terms",
            source_revision="large-term-fixture-v1",
        )

        self.assertEqual(report.parsed, 1)
        descriptor = next(index.iter_sequences("snapshot-large-terms"))
        revision = index.get_revision(descriptor.ref)
        self.assertNotIn("%S", json.loads(revision.raw_metadata_json))
        terms = self.term_store.read_range(descriptor.ref, 0, 2)
        expected = 0
        for _ in digits:
            expected = expected * 10 + 9
        self.assertEqual(terms.values, (expected, 0))

    def test_catalog_streams_sequence_refs_in_keyset_pages(self) -> None:
        records = [
            _oeis_record(f"A{index:06d}", [index], offset=0)
            for index in range(1, 4)
        ]
        SequenceIndex(self.catalog, self.term_store, page_size=1).ingest_lines(
            (line for record in records for line in record.splitlines(keepends=True)),
            snapshot_id="snapshot-keyset",
            source_revision="keyset-fixture-v1",
        )

        refs = list(
            self.catalog.iter_sequence_refs("snapshot-keyset", page_size=1)
        )

        self.assertEqual(
            [ref.sequence_id for ref in refs],
            ["A000001", "A000002", "A000003"],
        )
        self.assertEqual({ref.snapshot_id for ref in refs}, {"snapshot-keyset"})

    def test_bridge_finder_streams_large_catalog_seed_and_target_generators(self) -> None:
        SequenceIndex(self.catalog, self.term_store).ingest_lines(
            iter(_oeis_record("A000001", [1]).splitlines(keepends=True)),
            snapshot_id="snapshot-scale",
            source_revision="scale-fixture-v1",
        )
        self.catalog.execute(
            """
            INSERT INTO run (
              run_id, snapshot_id, run_kind, config_digest, code_revision,
              state, manifest_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "run-scale-bridge",
                "snapshot-scale",
                "bridge-test",
                "scale-config",
                "test",
                "running",
                "{}",
            ),
        )

        hash_index = HashIndex(self.catalog, self.term_store, "run-scale-bridge")
        detector = CollisionDetector(
            self.term_store, self.catalog, run_id="run-scale-bridge"
        )
        finder = BridgeFinder(self.term_store, hash_index, detector)
        transform = TransformSpec(
            name="scale-test",
            version="1",
            parameters={"input_term_count": 1},
            input_domain="test",
            output_bound=1,
        )
        pass_counts: list[int] = []
        operation_counts = {"target_fingerprints": 0, "transforms": 0}

        def catalog_refs(snapshot_id: str, *, page_size: int):
            self.assertEqual(snapshot_id, "snapshot-scale")
            self.assertEqual(page_size, 128)
            count = 0
            while count < current_size[0]:
                sequence_id = f"S{count:08d}"
                yield SequenceRef(
                    sequence_id,
                    f"R{count:08d}",
                    snapshot_id,
                )
                count += 1
            pass_counts.append(count)

        current_size = [0]

        def measure_peak(sequence_count: int) -> int:
            current_size[0] = sequence_count
            pass_counts.clear()
            operation_counts.update(target_fingerprints=0, transforms=0)

            def fingerprint_target(
                ref: SequenceRef,
                requested_profile: FingerprintProfile,
            ) -> Fingerprint:
                operation_counts["target_fingerprints"] += 1
                start_index = requested_profile.start_index or 0
                return Fingerprint(
                    ref,
                    requested_profile,
                    start_index,
                    start_index,
                    "target-digest",
                )

            def apply_transform(
                _requested_spec: TransformSpec,
                _values: tuple[int, ...],
            ) -> tuple[int, ...]:
                del _requested_spec, _values
                operation_counts["transforms"] += 1
                return (1,)

            def fingerprint_values(
                ref: SequenceRef,
                requested_profile: FingerprintProfile,
                _values: tuple[int, ...],
                *,
                start_index: int,
            ) -> Fingerprint:
                del _values
                return Fingerprint(
                    ref,
                    requested_profile,
                    start_index,
                    start_index,
                    "source-digest",
                )

            def no_matching_refs(
                _fingerprint: Fingerprint,
            ) -> tuple[SequenceRef, ...]:
                del _fingerprint
                return ()

            def read_range(
                _ref: SequenceRef,
                _start_index: int,
                _count: int,
            ) -> SimpleNamespace:
                del _ref, _start_index, _count

                def value_at(_index: int) -> int:
                    del _index
                    return 1

                return SimpleNamespace(value_at=value_at)

            def has_transform(_spec: TransformSpec) -> bool:
                del _spec
                return True

            with (
                patch.object(
                    self.catalog,
                    "iter_sequence_refs",
                    side_effect=catalog_refs,
                ),
                patch.object(
                    hash_index,
                    "fingerprint",
                    new=fingerprint_target,
                ),
                patch.object(
                    hash_index,
                    "fingerprint_values",
                    new=fingerprint_values,
                ),
                patch.object(
                    hash_index,
                    "find_matching_refs",
                    new=no_matching_refs,
                ),
                patch.object(
                    self.term_store,
                    "read_range",
                    new=read_range,
                ),
                patch.object(detector, "has_transform", new=has_transform),
                patch.object(detector, "apply_transform", new=apply_transform),
            ):
                tracemalloc.start()
                try:
                    for _ in finder.find_snapshot_bridges(
                        "snapshot-scale",
                        [transform],
                        compare_terms=1,
                        page_size=128,
                    ):
                        self.fail("no matching targets should be returned")
                    _, peak = tracemalloc.get_traced_memory()
                finally:
                    tracemalloc.stop()

            self.assertEqual(pass_counts, [sequence_count, sequence_count])
            self.assertEqual(operation_counts["target_fingerprints"], sequence_count)
            self.assertEqual(operation_counts["transforms"], sequence_count)
            return peak

        small_peak = measure_peak(1_000)
        large_peak = measure_peak(12_000)
        self.assertLess(
            large_peak,
            small_peak + 512 * 1024,
            f"peak traced memory scaled with corpus size: "
            f"{small_peak=} bytes, {large_peak=} bytes",
        )

    def _seed_pipeline_snapshot(self, root: Path) -> tuple[Catalog, TermStore]:
        catalog = Catalog(root / "catalog.sqlite")
        self.extra_catalogs.append(catalog)
        term_store = TermStore(root / "terms", catalog)
        lines: list[str] = []
        for index in range(1, 5):
            exponents = [1, 0, 2, 1, 0, index, 0]
            coefficients = euler_transform(exponents, degree=6)
            lines.append(
                _oeis_record(
                    f"A{index:06d}",
                    [int(value) for value in coefficients],
                )
            )
        indexer = SequenceIndex(catalog, term_store, chunk_terms=3)
        report = indexer.ingest_lines(
            (line for record in lines for line in record.splitlines(keepends=True)),
            snapshot_id="snapshot-pipeline",
            source_revision="pipeline-fixture-v1",
        )
        self.assertEqual(report.records_seen, 4)
        self.assertEqual(report.parsed, 4)
        return catalog, term_store

    def test_catalog_defaults_to_wal_with_normal_synchronous_mode(self) -> None:
        journal_mode = self.catalog.fetch_one("PRAGMA journal_mode")
        synchronous = self.catalog.fetch_one("PRAGMA synchronous")
        cache_size = self.catalog.fetch_one("PRAGMA cache_size")

        self.assertEqual(journal_mode["journal_mode"].lower(), "wal")
        self.assertEqual(synchronous["synchronous"], 1)
        self.assertEqual(cache_size["cache_size"], -64_000)

    def test_catalog_mutations_begin_immediate_transactions(self) -> None:
        self.catalog.execute("CREATE TABLE mutation_probe (value INTEGER)")
        statements: list[str] = []
        self.catalog._connection.set_trace_callback(statements.append)
        try:
            self.catalog.execute(
                "INSERT INTO mutation_probe (value) VALUES (?)",
                (1,),
            )
            self.catalog.executemany(
                "INSERT INTO mutation_probe (value) VALUES (?)",
                [(2,), (3,)],
            )
        finally:
            self.catalog._connection.set_trace_callback(None)

        self.assertEqual(
            sum(statement == "BEGIN IMMEDIATE" for statement in statements),
            2,
        )
        self.assertEqual(
            self.catalog.fetch_one("SELECT COUNT(*) AS n FROM mutation_probe")["n"],
            3,
        )

    def test_catalog_has_match_planning_and_run_claim_indexes(self) -> None:
        expected_indexes = {
            "artifact_by_transform_cursor": {
                "transform_key",
                "artifact_id",
            },
            "relation_claim_by_run_evidence": {
                "created_by_run",
                "evidence_level",
            },
        }

        for index_name, expected_columns in expected_indexes.items():
            columns = {
                row["name"]
                for row in self.catalog.fetch_all(
                    f"PRAGMA index_info({index_name})"
                )
            }
            self.assertEqual(columns, expected_columns)

    def test_inverse_euler_adapter_passes_term_series_without_copying(self) -> None:
        series = TermSeries(
            ref=SequenceRef("A000001", "revision", "snapshot"),
            start_index=0,
            values=(1, 1, 2),
            complete_through=2,
        )
        context = TransformerExecutionContext(
            input_series=series,
            spec=TransformSpec(
                name="inverse_euler",
                version="1",
                parameters={},
                input_domain="formal_series_constant_term_one",
                output_bound=2,
            ),
            metadata={"output_start_index": 1},
        )

        with patch(
            "oeis_xray_engine.transformers.builtins.inverse_euler_transform",
            return_value=[1, 2],
        ) as transform:
            result = InverseEulerTransformer().transform(context)

        self.assertIs(transform.call_args.args[0], series)
        self.assertEqual(result.values, (1, 2))

    def test_inverse_euler_pre_filter_keeps_standard_fingerprint_tasks(self) -> None:
        indexer = SequenceIndex(self.catalog, self.term_store, chunk_terms=2)
        records = (
            _oeis_record("A000101", [1, 1, 2, 3, 5, 8, 13]),
            _oeis_record("A000102", [0, 1, 2, 3, 5, 8, 13]),
            _oeis_record("A000103", [1, 1, 2, 3, 5, 8, 13], offset=1),
        )
        indexer.ingest_lines(
            (line for record in records for line in record.splitlines(keepends=True)),
            snapshot_id="snapshot-domain-filter",
            source_revision="domain-filter-fixture",
        )
        base_spec = self._pipeline_spec("run-domain-filter")
        spec = replace(
            base_spec,
            snapshot_id="snapshot-domain-filter",
            stages=("transform", "fingerprint"),
        )
        coordinator = PipelineCoordinator(self.catalog, self.term_store)
        coordinator.plan(spec)

        tasks = self.catalog.fetch_all(
            """
            SELECT task_key, task_manifest_json FROM task WHERE run_id = ?
            """,
            (spec.run_id,),
        )
        transform_ids = {
            json.loads(task["task_manifest_json"])["ref"]["sequence_id"]
            for task in tasks
            if task["task_key"].startswith("00:transform:")
        }
        fingerprint_ids = {
            json.loads(task["task_manifest_json"])["ref"]["sequence_id"]
            for task in tasks
            if task["task_key"].startswith("01:fingerprint:")
        }

        self.assertEqual(transform_ids, {"A000101"})
        self.assertEqual(
            fingerprint_ids,
            {"A000101", "A000102", "A000103"},
        )

    def test_match_planning_skips_oversized_fingerprint_buckets(self) -> None:
        indexer = SequenceIndex(self.catalog, self.term_store, chunk_terms=3)
        records = tuple(
            _oeis_record(f"A{sequence_number:06d}", [1, 2, 3])
            for sequence_number in range(201, 204)
        )
        indexer.ingest_lines(
            (line for record in records for line in record.splitlines(keepends=True)),
            snapshot_id="snapshot-large-bucket",
            source_revision="large-bucket-fixture",
        )
        base_spec = self._pipeline_spec("run-large-bucket")
        spec = replace(
            base_spec,
            snapshot_id="snapshot-large-bucket",
            stages=("fingerprint", "match"),
            transform_specs=(),
        )
        coordinator = PipelineCoordinator(
            self.catalog,
            self.term_store,
            task_batch_size=2,
        )

        with (
            patch("oeis_xray_engine.pipeline._MAX_MATCH_BUCKET_SIZE", 2),
            self.assertLogs("oeis_xray_engine.pipeline", level="INFO") as logs,
        ):
            report = coordinator.execute(
                coordinator.plan(spec),
                workers=1,
            )

        self.assertEqual(report.state, "complete")
        self.assertTrue(
            any("Starting candidate match planning" in message for message in logs.output)
        )
        self.assertEqual(
            self.catalog.fetch_one(
                "SELECT COUNT(*) AS n FROM task "
                "WHERE run_id = ? AND task_key LIKE '02:match:%'",
                (spec.run_id,),
            )["n"],
            0,
        )
        self.assertEqual(
            self.catalog.fetch_one(
                "SELECT COUNT(*) AS n FROM relation_claim WHERE created_by_run = ?",
                (spec.run_id,),
            )["n"],
            0,
        )

    def test_match_planner_batches_candidate_persistence(self) -> None:
        indexer = SequenceIndex(self.catalog, self.term_store, chunk_terms=3)
        records = tuple(
            _oeis_record(f"A{sequence_number:06d}", [1, 2, 3])
            for sequence_number in range(211, 214)
        )
        indexer.ingest_lines(
            (line for record in records for line in record.splitlines(keepends=True)),
            snapshot_id="snapshot-match-batch",
            source_revision="match-batch-fixture",
        )
        base_spec = self._pipeline_spec("run-match-batch")
        spec = replace(
            base_spec,
            snapshot_id="snapshot-match-batch",
            stages=("fingerprint", "match"),
            transform_specs=(),
        )
        coordinator = PipelineCoordinator(
            self.catalog,
            self.term_store,
            task_batch_size=500,
        )

        batch_sizes: list[int] = []
        persist_match_task_batch = coordinator._persist_match_task_batch

        def record_batch_size(items, run_id):
            batch_sizes.append(len(items))
            persist_match_task_batch(items, run_id)

        with patch.object(
            coordinator,
            "_persist_match_task_batch",
            side_effect=record_batch_size,
        ):
            report = coordinator.execute(
                coordinator.plan(spec),
                workers=1,
            )

        self.assertEqual(report.state, "complete")
        self.assertEqual(batch_sizes, [3])
        self.assertEqual(
            self.catalog.fetch_one(
                "SELECT COUNT(*) AS n FROM relation_claim "
                "WHERE created_by_run = ? AND evidence_level = 'E0'",
                (spec.run_id,),
            )["n"],
            3,
        )

    def test_worker_results_commit_in_batches_for_large_run(self) -> None:
        coordinator = PipelineCoordinator(self.catalog, self.term_store)
        results = [
            (
                SimpleNamespace(task_id=str(index), lease_owner="test-owner"),
                SimpleNamespace(),
            )
            for index in range(10_000)
        ]
        transaction_count = 0
        original_transaction = self.catalog.transaction

        @contextmanager
        def counted_transaction(*, immediate: bool = True):
            nonlocal transaction_count
            if getattr(self.catalog._local, "transaction_depth", 0) == 0:
                transaction_count += 1
            with original_transaction(immediate=immediate):
                yield self.catalog

        with (
            patch.object(
                self.catalog,
                "transaction",
                side_effect=counted_transaction,
            ),
            patch.object(coordinator, "_commit_worker_result"),
        ):
            committed = coordinator._commit_worker_results(
                results,
                max_attempts=3,
            )

        self.assertEqual(committed, 10_000)
        self.assertEqual(transaction_count, 50)

    def test_stale_worker_heartbeat_releases_unexpired_lease(self) -> None:
        catalog, term_store = self._seed_pipeline_snapshot(
            self.root / "stale-heartbeat"
        )
        coordinator = PipelineCoordinator(catalog, term_store)
        run_id = coordinator.plan(self._pipeline_spec("run-stale-heartbeat"))
        lease = coordinator.checkpoints.lease_tasks(
            run_id,
            owner="heartbeat-worker",
            limit=1,
            lease_seconds=3_600,
        )[0]

        heartbeat_at = datetime.now(timezone.utc)
        self.assertTrue(
            coordinator.checkpoints.heartbeat_task(
                lease.task_id,
                lease.lease_owner,
                lease_seconds=3_600,
                now=heartbeat_at.isoformat(),
            )
        )
        stored_lease = catalog.fetch_one(
            "SELECT lease_until FROM task WHERE task_id = ?",
            (lease.task_id,),
        )
        self.assertGreater(
            datetime.fromisoformat(stored_lease["lease_until"]),
            heartbeat_at,
        )

        stale_at = (heartbeat_at - timedelta(minutes=5, seconds=1)).isoformat()
        catalog.execute(
            """
            UPDATE task SET last_heartbeat = ?, lease_until = ?
            WHERE task_id = ?
            """,
            (
                stale_at,
                (heartbeat_at + timedelta(hours=1)).isoformat(),
                lease.task_id,
            ),
        )
        recovered = coordinator.checkpoints.recover_expired_leases(
            run_id,
            now=heartbeat_at.isoformat(),
            max_attempts=3,
        )

        self.assertEqual(recovered, 1)
        task = catalog.fetch_one(
            "SELECT state, last_heartbeat FROM task WHERE task_id = ?",
            (lease.task_id,),
        )
        self.assertEqual(task["state"], "retryable_error")
        self.assertIsNone(task["last_heartbeat"])

    def test_heartbeat_listener_updates_lease_in_background(self) -> None:
        catalog, term_store = self._seed_pipeline_snapshot(
            self.root / "background-heartbeat"
        )
        coordinator = PipelineCoordinator(catalog, term_store)
        run_id = coordinator.plan(
            self._pipeline_spec("run-background-heartbeat")
        )
        lease = coordinator.checkpoints.lease_tasks(
            run_id,
            owner="background-heartbeat-worker",
            limit=1,
            lease_seconds=3_600,
        )[0]
        heartbeat_queue: Queue[tuple[str, str]] = Queue()

        with _heartbeat_listener(
            coordinator,
            heartbeat_queue,
            lease_seconds=3_600,
        ):
            heartbeat_queue.put((lease.task_id, lease.lease_owner))
            time.sleep(0.05)

        task = catalog.fetch_one(
            "SELECT last_heartbeat FROM task WHERE task_id = ?",
            (lease.task_id,),
        )
        self.assertIsNotNone(task["last_heartbeat"])

    @staticmethod
    def _pipeline_spec(run_id: str) -> PipelineSpec:
        transform = TransformSpec(
            name="inverse_euler",
            version="1",
            parameters={
                "input_start_index": 0,
                "input_term_count": 7,
                "output_start_index": 1,
                "output_term_count": 6,
            },
            input_domain="formal_series_constant_term_one",
            output_bound=6,
        )
        profile = FingerprintProfile(
            name="pipeline-prefix",
            version="1",
            mode="exact",
            term_count=5,
            start_index=0,
        )
        return PipelineSpec(
            run_id=run_id,
            snapshot_id="snapshot-pipeline",
            stages=("transform", "fingerprint", "match"),
            transform_specs=(transform,),
            fingerprint_profiles=(profile,),
            page_size=2,
            lease_seconds=60,
            max_attempts=3,
            code_revision="test-code",
        )

    def test_multiprocess_pipeline_transform_fingerprint_match_and_recovery(self) -> None:
        catalog, term_store = self._seed_pipeline_snapshot(self.root / "recovery")
        coordinator = PipelineCoordinator(catalog, term_store, task_batch_size=2)
        run_id = coordinator.plan(self._pipeline_spec("run-recovery"))

        checkpoints = CheckpointStore(catalog)
        leases = checkpoints.lease_tasks(
            run_id,
            owner="simulated-crashed-worker",
            limit=1,
            lease_seconds=1,
        )
        self.assertEqual(len(leases), 1)
        recovered = checkpoints.recover_expired_leases(
            run_id,
            now="9999-12-31T23:59:59+00:00",
            max_attempts=3,
        )
        self.assertEqual(recovered, 1)
        self.assertEqual(
            checkpoints.task_counts(run_id).retryable_error,
            1,
        )
        # A crashed task has no committed fingerprint/artifact output.
        self.assertEqual(
            catalog.fetch_one(
                "SELECT COUNT(*) AS n FROM derived_artifact WHERE created_by_run = ?",
                (run_id,),
            )["n"],
            0,
        )
        self.assertEqual(
            catalog.fetch_one(
                "SELECT COUNT(*) AS n FROM fingerprint WHERE created_by_run = ?",
                (run_id,),
            )["n"],
            0,
        )

        interrupted = coordinator.execute(
            run_id,
            workers=4,
            max_completed_tasks=1,
        )
        self.assertEqual(interrupted.state, "paused")
        resumed = coordinator.resume(run_id, workers=4)
        self.assertEqual(resumed.state, "complete")
        self.assertGreater(resumed.counts.complete, 0)
        self.assertEqual(resumed.counts.pending, 0)
        self.assertEqual(resumed.counts.retryable_error, 0)
        self.assertEqual(
            catalog.fetch_one(
                "SELECT COUNT(*) AS n FROM relation_claim WHERE created_by_run = ? AND evidence_level = 'E1'",
                (run_id,),
            )["n"],
            6,
        )
        recovery_manifest = coordinator.result_manifest(run_id)

        clean_root = self.root / "clean"
        clean_catalog, clean_store = self._seed_pipeline_snapshot(clean_root)
        clean_coordinator = PipelineCoordinator(
            clean_catalog,
            clean_store,
            task_batch_size=2,
        )
        clean_run = clean_coordinator.plan(self._pipeline_spec("run-clean"))
        clean_report = clean_coordinator.execute(clean_run, workers=1)
        self.assertEqual(clean_report.state, "complete")
        self.assertEqual(
            recovery_manifest,
            clean_coordinator.result_manifest(clean_run),
        )

    def test_transform_replay_accepts_same_artifact_in_new_bundle_location(self) -> None:
        catalog, term_store = self._seed_pipeline_snapshot(self.root / "artifact-replay")
        coordinator = PipelineCoordinator(catalog, term_store, task_batch_size=2)
        transform_only = replace(
            self._pipeline_spec("run-artifact-first"),
            stages=("transform",),
            fingerprint_profiles=(),
        )
        first_run = coordinator.plan(transform_only)
        first_report = coordinator.execute(first_run, workers=1)
        self.assertEqual(first_report.state, "complete")

        second_run = coordinator.plan(
            replace(transform_only, run_id="run-artifact-second")
        )
        second_report = coordinator.execute(second_run, workers=1)

        self.assertEqual(second_report.state, "complete")
        self.assertEqual(second_report.counts.fatal_error, 0)
        self.assertEqual(second_report.counts.pending, 0)

    def test_mixed_domain_rejections_do_not_skip_derived_matches(self) -> None:
        root = self.root / "derived-match"
        catalog = Catalog(root / "catalog.sqlite")
        self.extra_catalogs.append(catalog)
        term_store = TermStore(root / "terms", catalog)
        exponents = [1, 0, 2, 1, 0, 1, 0]
        coefficients = [int(value) for value in euler_transform(exponents, degree=6)]
        records = [
            _oeis_record("A000001", coefficients),
            _oeis_record("A000002", [1, *exponents]),
            _oeis_record("A000003", [2, *exponents]),
        ]
        SequenceIndex(catalog, term_store, chunk_terms=4).ingest_lines(
            (line for record in records for line in record.splitlines(keepends=True)),
            snapshot_id="snapshot-derived",
            source_revision="derived-fixture-v1",
        )
        transform = TransformSpec(
            name="inverse_euler",
            version="1",
            parameters={
                "input_start_index": 0,
                "input_term_count": 7,
                "output_start_index": 1,
                "output_term_count": 6,
            },
            input_domain="formal_series_constant_term_one",
            output_bound=6,
        )
        profile = FingerprintProfile(
            name="derived-prefix",
            version="1",
            mode="exact",
            term_count=5,
            start_index=1,
        )
        spec = PipelineSpec(
            run_id="run-derived-match",
            snapshot_id="snapshot-derived",
            stages=("transform", "fingerprint", "match"),
            transform_specs=(transform,),
            fingerprint_profiles=(profile,),
            page_size=2,
            lease_seconds=60,
            max_attempts=3,
            code_revision="test-code",
        )

        coordinator = PipelineCoordinator(catalog, term_store, task_batch_size=2)
        run_id = coordinator.plan(spec)
        paused = coordinator.execute(
            run_id,
            workers=2,
            max_completed_tasks=7,
        )
        self.assertEqual(paused.state, "paused")
        coordinator.checkpoints.set_run_state(run_id, "failed")

        reverse_artifact = catalog.fetch_one(
            """
            SELECT af.fingerprint_id
            FROM artifact_fingerprint AS af
            JOIN derived_artifact AS a ON a.artifact_id = af.artifact_id
            JOIN sequence_revision AS source
              ON source.revision_id = a.revision_id
            JOIN sequence AS source_sequence
              ON source_sequence.sequence_id = source.sequence_id
            WHERE source_sequence.sequence_id = 'A000002'
            LIMIT 1
            """
        )
        self.assertIsNotNone(reverse_artifact)
        target_fingerprint = catalog.fetch_one(
            """
            SELECT f.digest
            FROM fingerprint AS f
            JOIN sequence_revision AS target ON target.revision_id = f.revision_id
            JOIN sequence AS target_sequence
              ON target_sequence.sequence_id = target.sequence_id
            WHERE target_sequence.sequence_id = 'A000001'
            LIMIT 1
            """
        )
        self.assertIsNotNone(target_fingerprint)
        catalog.execute(
            "UPDATE artifact_fingerprint SET digest = ? WHERE fingerprint_id = ?",
            (target_fingerprint["digest"], reverse_artifact["fingerprint_id"]),
        )

        report = coordinator.resume(run_id, workers=2)

        self.assertEqual(report.state, "complete")
        self.assertEqual(report.counts.fatal_error, 0)
        self.assertEqual(
            catalog.fetch_one(
                "SELECT COUNT(*) AS n FROM artifact_fingerprint WHERE created_by_run = ?",
                (run_id,),
            )["n"],
            2,
        )
        self.assertGreater(
            catalog.fetch_one(
                """
                SELECT COUNT(*) AS n FROM relation_claim
                WHERE created_by_run = ?
                  AND relation_kind = 'transform-fingerprint-range'
                  AND evidence_level = 'E0'
                """,
                (run_id,),
            )["n"],
            0,
        )
        claim = catalog.fetch_one(
            """
            SELECT rc.relation_kind, rc.source_revision_id, rc.target_revision_id,
                   rc.transform_key, rc.evidence_level, rc.compared_start,
                   rc.compared_end
            FROM relation_claim AS rc
            JOIN sequence_revision AS source
              ON source.revision_id = rc.source_revision_id
            JOIN sequence AS source_sequence
              ON source_sequence.sequence_id = source.sequence_id
            JOIN sequence_revision AS target
              ON target.revision_id = rc.target_revision_id
            JOIN sequence AS target_sequence
              ON target_sequence.sequence_id = target.sequence_id
            WHERE rc.created_by_run = ?
              AND source_sequence.sequence_id = 'A000001'
              AND target_sequence.sequence_id = 'A000002'
              AND rc.evidence_level = 'E1'
            """,
            (run_id,),
        )
        self.assertIsNotNone(claim)
        self.assertEqual(claim["relation_kind"], "transform-fingerprint-range")
        self.assertIsNotNone(claim["transform_key"])
        self.assertEqual(claim["evidence_level"], "E1")
        self.assertEqual((claim["compared_start"], claim["compared_end"]), (1, 5))
        false_e0 = catalog.fetch_one(
            """
            SELECT rc.evidence_level
            FROM relation_claim AS rc
            JOIN sequence_revision AS source
              ON source.revision_id = rc.source_revision_id
            JOIN sequence AS source_sequence
              ON source_sequence.sequence_id = source.sequence_id
            JOIN sequence_revision AS target
              ON target.revision_id = rc.target_revision_id
            JOIN sequence AS target_sequence
              ON target_sequence.sequence_id = target.sequence_id
            WHERE rc.created_by_run = ?
              AND source_sequence.sequence_id = 'A000002'
              AND target_sequence.sequence_id = 'A000001'
              AND rc.relation_kind = 'transform-fingerprint-range'
              AND rc.evidence_level = 'E0'
            """,
            (run_id,),
        )
        self.assertIsNotNone(false_e0)
        false_candidate = catalog.fetch_one(
            """
            SELECT rc.evidence_level
            FROM relation_claim AS rc
            JOIN sequence_revision AS source
              ON source.revision_id = rc.source_revision_id
            JOIN sequence AS source_sequence
              ON source_sequence.sequence_id = source.sequence_id
            JOIN sequence_revision AS target
              ON target.revision_id = rc.target_revision_id
            JOIN sequence AS target_sequence
              ON target_sequence.sequence_id = target.sequence_id
            WHERE rc.created_by_run = ?
              AND source_sequence.sequence_id = 'A000002'
              AND target_sequence.sequence_id = 'A000001'
              AND rc.relation_kind = 'transform-fingerprint-range'
              AND rc.evidence_level = 'E1'
            """,
            (run_id,),
        )
        self.assertIsNone(false_candidate)

    def test_denumerant_artifact_matches_original_sequence(self) -> None:
        root = self.root / "denumerant-derived-match"
        catalog = Catalog(root / "catalog.sqlite")
        self.extra_catalogs.append(catalog)
        term_store = TermStore(root / "terms", catalog)
        records = [
            _oeis_record("A000012", [1] * 20),
            _oeis_record("A000009", [1] * 20),
        ]
        SequenceIndex(catalog, term_store, chunk_terms=8).ingest_lines(
            (line for record in records for line in record.splitlines(keepends=True)),
            snapshot_id="snapshot-denumerant-derived",
            source_revision="denumerant-derived-fixture-v1",
        )
        transform = TransformSpec(
            name="denumerant",
            version="1",
            parameters={
                "input_start_index": 0,
                "term_cap": 20,
                "confidence_window": 5,
                "output_start_index": 1,
            },
            input_domain="finite_integer_sequence_pure_bosonic_euler_product",
            output_bound=20,
        )
        profile = FingerprintProfile(
            name="derived-denumerant",
            version="1",
            mode="exact",
            term_count=1,
            start_index=1,
        )
        spec = PipelineSpec(
            run_id="run-denumerant-derived-match",
            snapshot_id="snapshot-denumerant-derived",
            stages=("transform", "fingerprint", "match"),
            transform_specs=(transform,),
            fingerprint_profiles=(profile,),
            page_size=2,
            lease_seconds=60,
            max_attempts=3,
            code_revision="test-code",
        )

        coordinator = PipelineCoordinator(catalog, term_store, task_batch_size=2)
        report = coordinator.execute(coordinator.plan(spec), workers=2)

        self.assertEqual(report.state, "complete")
        claim = catalog.fetch_one(
            """
            SELECT rc.evidence_level, transform.name
            FROM relation_claim AS rc
            JOIN sequence_revision AS source
              ON source.revision_id = rc.source_revision_id
            JOIN sequence AS source_sequence
              ON source_sequence.sequence_id = source.sequence_id
            JOIN sequence_revision AS target
              ON target.revision_id = rc.target_revision_id
            JOIN sequence AS target_sequence
              ON target_sequence.sequence_id = target.sequence_id
            JOIN transform_spec AS transform
              ON transform.transform_key = rc.transform_key
            WHERE rc.created_by_run = ?
              AND source_sequence.sequence_id = 'A000012'
              AND target_sequence.sequence_id = 'A000009'
              AND rc.evidence_level = 'E1'
            """,
            (spec.run_id,),
        )
        self.assertIsNotNone(claim)
        self.assertEqual(claim["evidence_level"], "E1")
        self.assertEqual(claim["name"], "denumerant")

    def test_incomplete_catalog_schema_migrates_task_manifest_column(self) -> None:
        self.catalog.execute("DROP TABLE task")
        self.catalog.execute(
            """
            CREATE TABLE task (
              task_id TEXT PRIMARY KEY,
              run_id TEXT NOT NULL REFERENCES run(run_id),
              task_key TEXT NOT NULL,
              partition_key TEXT NOT NULL,
              state TEXT NOT NULL,
              lease_owner TEXT,
              lease_until TEXT,
              attempt_count INTEGER NOT NULL DEFAULT 0,
              input_digest TEXT NOT NULL,
              output_digest TEXT,
              error_json TEXT,
              updated_at TEXT NOT NULL,
              UNIQUE(run_id, task_key)
            )
            """
        )
        for column in ("bundle_id", "byte_offset", "byte_length"):
            self.catalog.execute(f"ALTER TABLE term_chunk DROP COLUMN {column}")
        self.catalog.initialize_schema()
        columns = {
            row["name"] for row in self.catalog.fetch_all("PRAGMA table_info(task)")
        }
        self.assertIn("task_manifest_json", columns)
        self.assertIn("last_heartbeat", columns)
        chunk_columns = {
            row["name"]
            for row in self.catalog.fetch_all("PRAGMA table_info(term_chunk)")
        }
        self.assertTrue(
            {"bundle_id", "byte_offset", "byte_length"}.issubset(chunk_columns)
        )


if __name__ == "__main__":
    unittest.main()
