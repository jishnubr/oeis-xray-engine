"""Integration coverage for the X-Ray command-line interface."""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from oeis_xray_engine.cli import main
from oeis_xray_engine.storage.catalog import Catalog


class XRayCliTests(unittest.TestCase):
    @staticmethod
    def _ingest_all_ones(root: Path, term_count: int = 20) -> tuple[Path, Path]:
        source = root / "seq"
        source.mkdir()
        terms = ",".join(["1"] * term_count)
        (source / "A000012.seq").write_text(
            f"%I A000012\n%S A000012 {terms}\n%O A000012 0\n",
            encoding="utf-8",
        )
        catalog = root / "catalog.sqlite"
        term_store = root / "terms"
        with contextlib.redirect_stdout(io.StringIO()):
            result = main(
                [
                    "ingest",
                    "--source",
                    str(source),
                    "--snapshot",
                    "pilot-v1",
                    "--catalog",
                    str(catalog),
                    "--term-store",
                    str(term_store),
                ]
            )
        if result != 0:
            raise AssertionError("fixture ingestion failed")
        return catalog, term_store

    def test_ingest_plan_run_and_report(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "seq"
            source.mkdir()
            for sequence_id in ("A000101", "A000102"):
                (source / f"{sequence_id}.seq").write_text(
                    f"%I {sequence_id} M0000\n"
                    f"%S {sequence_id} 1,2,3\n"
                    f"%O {sequence_id} 0\n",
                    encoding="utf-8",
                )
            catalog = root / "catalog.sqlite"
            terms = root / "terms"

            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(
                    main(
                        [
                            "ingest",
                            "--source",
                            str(source),
                            "--snapshot",
                            "pilot-v1",
                            "--catalog",
                            str(catalog),
                            "--term-store",
                            str(terms),
                        ]
                    ),
                    0,
                )

            plan_output = io.StringIO()
            with contextlib.redirect_stdout(plan_output):
                self.assertEqual(
                    main(
                        [
                            "plan",
                            "--snapshot",
                            "pilot-v1",
                            "--max-terms",
                            "3",
                            "--stages",
                            "fingerprint",
                            "match",
                            "--run-id",
                            "pilot-run",
                            "--catalog",
                            str(catalog),
                            "--term-store",
                            str(terms),
                        ]
                    ),
                    0,
                )
            self.assertIn("pilot-run", plan_output.getvalue())

            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(
                    main(
                        [
                            "run",
                            "--run-id",
                            "pilot-run",
                            "--workers",
                            "1",
                            "--catalog",
                            str(catalog),
                            "--term-store",
                            str(terms),
                        ]
                    ),
                    0,
                )

            report_output = io.StringIO()
            with contextlib.redirect_stdout(report_output):
                self.assertEqual(
                    main(
                        [
                            "report",
                            "--run-id",
                            "pilot-run",
                            "--catalog",
                            str(catalog),
                            "--term-store",
                            str(terms),
                        ]
                    ),
                    0,
                )
            report = report_output.getvalue()
            self.assertIn("Tasks completed", report)
            self.assertIn("E0 candidates found", report)
            self.assertIn("E1 matches verified", report)
            self.assertIn("E2 reconstructions verified", report)
            self.assertIn("E3 symbolic identities verified", report)
            self.assertIn("E4 human reviews recorded", report)
            self.assertIn("Errors encountered", report)
            self.assertIn("pilot-run", report)

    def test_export_gems_outputs_json_and_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "seq"
            source.mkdir()
            (source / "A000123.seq").write_text(
                "%I A000123 M0000\n"
                "%S A000123 1,2,3\n"
                "%O A000123 0\n",
                encoding="utf-8",
            )
            (source / "A000456.seq").write_text(
                "%I A000456 M0000\n"
                "%S A000456 1,2,3\n"
                "%O A000456 0\n",
                encoding="utf-8",
            )
            catalog_path = root / "catalog.sqlite"
            terms_path = root / "terms"
            command_options = [
                "--catalog",
                str(catalog_path),
                "--term-store",
                str(terms_path),
            ]

            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(
                    main(
                        [
                            "ingest",
                            "--source",
                            str(source),
                            "--snapshot",
                            "gem-export-snapshot",
                            *command_options,
                        ]
                    ),
                    0,
                )
                self.assertEqual(
                    main(
                        [
                            "plan",
                            "--snapshot",
                            "gem-export-snapshot",
                            "--max-terms",
                            "3",
                            "--stages",
                            "fingerprint",
                            "match",
                            "--run-id",
                            "gem-export-run",
                            *command_options,
                        ]
                    ),
                    0,
                )
                run_output = io.StringIO()
                with contextlib.redirect_stdout(run_output):
                    run_result = main(
                        [
                            "run",
                            "--run-id",
                            "gem-export-run",
                            "--workers",
                            "1",
                            *command_options,
                        ]
                    )
                self.assertEqual(run_result, 0, run_output.getvalue())

            with Catalog(catalog_path) as catalog:
                claim = catalog.fetch_one(
                    """
                    SELECT claim_id FROM relation_claim
                    WHERE created_by_run = ? AND evidence_level = 'E1'
                    LIMIT 1
                    """,
                    ("gem-export-run",),
                )
                self.assertIsNotNone(claim)
                catalog.execute(
                    """
                    INSERT INTO transform_spec (
                      transform_key, name, implementation_version,
                      canonical_parameters, domain_spec
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        "export-test-transform",
                        "denumerant",
                        "1",
                        json.dumps({"parameters": {"term_cap": 100}}),
                        "test-domain",
                    ),
                )
                catalog.execute(
                    """
                    UPDATE relation_claim
                    SET relation_kind = 'transform-fingerprint-range',
                        transform_key = ?
                    WHERE claim_id = ?
                    """,
                    ("export-test-transform", claim["claim_id"]),
                )

            json_output = io.StringIO()
            with contextlib.redirect_stdout(json_output):
                self.assertEqual(
                    main(
                        [
                            "export-gems",
                            "--run-id",
                            "gem-export-run",
                            "--format",
                            "json",
                            *command_options,
                        ]
                    ),
                    0,
                )
            exported = json.loads(json_output.getvalue())
            self.assertTrue(exported)
            transformed_claim = next(
                item
                for item in exported
                if item["transform"] is not None
                and item["evidence_level"] == "E1"
            )
            self.assertEqual(transformed_claim["transform"]["name"], "denumerant")
            self.assertEqual(transformed_claim["transform"]["output_bound"], 100)
            self.assertEqual(transformed_claim["compared_terms"], 3)

            all_output = io.StringIO()
            with contextlib.redirect_stdout(all_output):
                self.assertEqual(
                    main(
                        [
                            "export-gems",
                            "--format",
                            "json",
                            *command_options,
                        ]
                    ),
                    0,
                )
            self.assertTrue(json.loads(all_output.getvalue()))

            markdown_output = io.StringIO()
            with contextlib.redirect_stdout(markdown_output):
                self.assertEqual(
                    main(
                        [
                            "export-gems",
                            "--run-id",
                            "gem-export-run",
                            "--format",
                            "markdown",
                            *command_options,
                        ]
                    ),
                    0,
                )
            markdown = markdown_output.getvalue()
            self.assertIn(
                f"denumerant({transformed_claim['source']}) = "
                f"{transformed_claim['target']}",
                markdown,
            )
            self.assertIn("denumerant v1 (cap: 100)", markdown)
            self.assertIn("E1 (Exact match for 3 terms)", markdown)

    def test_denumerant_sweep_runs_transform_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            catalog_path, term_store = self._ingest_all_ones(root)

            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(
                    main(
                        [
                            "plan",
                            "--snapshot",
                            "pilot-v1",
                            "--sweep",
                            "denumerant",
                            "--max-terms",
                            "20",
                            "--stages",
                            "transform",
                            "--run-id",
                            "denumerant-run",
                            "--catalog",
                            str(catalog_path),
                            "--term-store",
                            str(term_store),
                        ]
                    ),
                    0,
                )
                self.assertEqual(
                    main(
                        [
                            "run",
                            "--run-id",
                            "denumerant-run",
                            "--workers",
                            "2",
                            "--catalog",
                            str(catalog_path),
                            "--term-store",
                            str(term_store),
                        ]
                    ),
                    0,
                )

            with Catalog(catalog_path) as catalog:
                artifact = catalog.fetch_one(
                    "SELECT artifact_id FROM derived_artifact WHERE created_by_run = ?",
                    ("denumerant-run",),
                )
                self.assertIsNotNone(artifact)

    def test_ramanujan_sweep_spec_is_persisted_in_run_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            catalog_path, term_store = self._ingest_all_ones(root)
            search_spec = {
                "constants": ["1"],
                "lhs_forms": [{"numerator": [1], "denominator": [1]}],
                "rhs_series": [{"a_terms": [1], "b_terms": [0]}],
                "key_scale": 1000,
                "initial_dps": 20,
                "verify_dps": 40,
                "verify_digits": 20,
                "max_pairs": 1,
                "max_results": 1,
            }
            spec_path = root / "ramanujan.json"
            spec_path.write_text(json.dumps(search_spec), encoding="utf-8")

            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(
                    main(
                        [
                            "plan",
                            "--snapshot",
                            "pilot-v1",
                            "--sweep",
                            "ramanujan",
                            "--ramanujan-spec",
                            str(spec_path),
                            "--max-terms",
                            "20",
                            "--run-id",
                            "ramanujan-run",
                            "--catalog",
                            str(catalog_path),
                            "--term-store",
                            str(term_store),
                        ]
                    ),
                    0,
                )

            with Catalog(catalog_path) as catalog:
                run = catalog.fetch_one(
                    "SELECT manifest_json FROM run WHERE run_id = ?",
                    ("ramanujan-run",),
                )
                self.assertIsNotNone(run)
                manifest = json.loads(run["manifest_json"])
                spec = manifest["pipeline_spec"]
                self.assertEqual(spec["stages"], ["fingerprint", "match"])
                self.assertEqual(
                    spec["ramanujan_search_specs"], [search_spec]
                )


if __name__ == "__main__":
    unittest.main()
