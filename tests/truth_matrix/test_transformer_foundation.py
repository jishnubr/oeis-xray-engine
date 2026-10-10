"""Truth-matrix checks for the transformer plugin foundation."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from oeis_xray_engine.indexing.sequence_index import SequenceIndex
from oeis_xray_engine.models import ChunkRef, SequenceRef, TransformSpec
from oeis_xray_engine.pipeline import PipelineCoordinator, PipelineSpec
from oeis_xray_engine.storage.catalog import Catalog
from oeis_xray_engine.storage.term_store import TermStore
from oeis_xray_engine.transformers import (
    TransformerContract,
    TransformerExecutionContext,
    TransformerRegistry,
    TransformerResult,
)


class DummyTransformer(TransformerContract):
    """Test-only, exact transform proving registry-to-worker dispatch."""

    name = "dummy_add"
    version = "1"

    def transform(
        self,
        context: TransformerExecutionContext,
    ) -> TransformerResult:
        self.validate(context)
        increment = context.spec.parameters["increment"]
        if type(increment) is not int:
            raise TypeError("increment must be an integer")
        return TransformerResult(
            start_index=int(context.metadata["output_start_index"]),
            values=tuple(
                value + increment for value in context.input_series.values
            ),
            metadata={"implementation": "dummy", "increment": increment},
        )


class TransformerFoundationTests(unittest.TestCase):
    """Exercise contract immutability, registration, and pipeline execution."""

    def test_registry_resolves_versioned_transformer(self) -> None:
        registry = TransformerRegistry()
        registry.register(DummyTransformer)

        transformer = registry.resolve("dummy_add", "1")

        self.assertIsInstance(transformer, DummyTransformer)
        self.assertEqual(
            registry.worker_reference("dummy_add", "1"),
            "tests.truth_matrix.test_transformer_foundation:DummyTransformer",
        )

    def test_pipeline_runs_registered_transformer_in_worker(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            catalog = Catalog(root / "catalog.sqlite")
            try:
                store = TermStore(root / "terms", catalog)
                index = SequenceIndex(catalog, store, chunk_terms=2)
                ingest = index.ingest_lines(
                    iter(
                        [
                            "%I A000001 M0000\n",
                            "%S A000001 0,1,1\n",
                            "%O A000001 0\n",
                        ]
                    ),
                    snapshot_id="truth-matrix",
                    source_revision="dummy-fixture-v1",
                )
                self.assertEqual(ingest.parsed, 1)

                registry = TransformerRegistry()
                registry.register(DummyTransformer)
                coordinator = PipelineCoordinator(
                    catalog,
                    store,
                    task_batch_size=1,
                    transformer_registry=registry,
                )
                transform = TransformSpec(
                    name="dummy_add",
                    version="1",
                    parameters={
                        "increment": 7,
                        "input_start_index": 0,
                        "input_term_count": 3,
                        "output_start_index": 0,
                        "output_term_count": 3,
                    },
                    input_domain="integer_sequence",
                    output_bound=3,
                )
                spec = PipelineSpec(
                    run_id="truth-matrix-dummy",
                    snapshot_id="truth-matrix",
                    stages=("transform",),
                    transform_specs=(transform,),
                    page_size=1,
                )

                run_id = coordinator.plan(spec)
                report = coordinator.execute(run_id, workers=2)

                self.assertEqual(report.state, "complete")
                self.assertEqual(report.counts.complete, 1)
                row = catalog.fetch_one(
                    """
                    SELECT revision_id, transform_key, output_bound,
                           term_manifest_json
                    FROM derived_artifact WHERE created_by_run = ?
                    """,
                    (run_id,),
                )
                self.assertIsNotNone(row)
                manifest = json.loads(row["term_manifest_json"])
                self.assertEqual(
                    manifest["_transformer_reference"],
                    "tests.truth_matrix.test_transformer_foundation:DummyTransformer",
                )
                self.assertEqual(
                    manifest["_transformer_metadata"],
                    {"implementation": "dummy", "increment": 7},
                )

                derived_revision = hashlib.sha256(
                    json.dumps(
                        [
                            row["revision_id"],
                            row["transform_key"],
                            row["output_bound"],
                        ],
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest()
                derived_ref = SequenceRef(
                    f"XRAY-{derived_revision[:24].upper()}",
                    derived_revision,
                    "truth-matrix",
                )
                chunk = ChunkRef(
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
                )
                output = TermStore(root / "terms").read_chunk(chunk, derived_ref)
                self.assertEqual(output.values, (7, 8, 8))
            finally:
                catalog.close()


if __name__ == "__main__":
    unittest.main()
