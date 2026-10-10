"""Captured legacy truth vectors and pipeline tests for the Ramanujan port."""

from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from unittest.mock import patch

import mpmath

from oeis_xray_engine.indexing.hash_index import FingerprintProfile
from oeis_xray_engine.indexing.sequence_index import SequenceIndex
from oeis_xray_engine.pipeline import PipelineCoordinator, PipelineSpec
from oeis_xray_engine.storage.catalog import Catalog
from oeis_xray_engine.storage.term_store import TermStore
from oeis_xray_engine.transformers.ramanujan import (
    BloomKeyIndex,
    CFSearchSpec,
    EfficientGCF,
    GCFSeries,
    GeneralizedContinuedFraction,
    LinearForm,
    MobiusTransform,
    efficient_gcf_fraction,
    search_gcf_candidates,
)


class RamanujanGCFTruthMatrixTests(unittest.TestCase):
    """Check the exact paths against values captured from Project A."""

    def test_efficient_gcf_matches_project_a_truth_vectors(self) -> None:
        cases = (
            (
                (1, 2, 2, 2, 2),
                (0, 1, 1, 1, 1),
                Fraction(41, 29),
            ),
            ((2, 1, 3, 2), (9, 1, 4, 2), Fraction(5, 2)),
            ((0, 1, 1, 1), (0, 1, 1, 1), Fraction(2, 3)),
        )
        for a_terms, b_terms, expected in cases:
            with self.subTest(a_terms=a_terms, b_terms=b_terms):
                self.assertEqual(efficient_gcf_fraction(a_terms, b_terms), expected)
                self.assertEqual(
                    mpmath.nstr(EfficientGCF(a_terms, b_terms).evaluate(), 15),
                    mpmath.nstr(
                        mpmath.mpf(expected.numerator) / expected.denominator,
                        15,
                    ),
                )

    def test_numpy_mobius_composition_matches_project_a_matrix(self) -> None:
        left = MobiusTransform(((0, 1), (1, 2)))
        right = MobiusTransform(((0, 3), (1, 4)))
        self.assertEqual((left * right).data.tolist(), [[1, 4], [2, 11]])

    def test_generalized_continued_fraction_matches_legacy_convergent(self) -> None:
        continued_fraction = GeneralizedContinuedFraction(
            (1, 2, 2, 2, 2),
            (1, 1, 1, 1),
        )
        self.assertEqual(continued_fraction.evaluate(), Fraction(41, 29))

    def test_bloom_index_has_no_false_negatives_for_inserted_keys(self) -> None:
        index = BloomKeyIndex(expected_items=16)
        inserted = (-1, 0, 2, 10**40)
        for key in inserted:
            index.add(key)
        self.assertTrue(all(key in index for key in inserted))

    def test_bloom_positive_without_exact_bucket_is_not_a_candidate(self) -> None:
        spec = replace(
            _rational_gcf_search_spec(),
            rhs_series=(GCFSeries((2, 2), (0, 1)),),
        )
        with patch(
            "oeis_xray_engine.transformers.ramanujan.search.BloomKeyIndex.__contains__",
            return_value=True,
        ):
            self.assertEqual(search_gcf_candidates(spec), ())

    def test_bounded_search_refines_candidate_to_requested_digits(self) -> None:
        candidates = search_gcf_candidates(_rational_gcf_search_spec())
        self.assertEqual(len(candidates), 1)
        candidate = candidates[0]
        self.assertEqual(candidate.lhs_index, 0)
        self.assertEqual(candidate.rhs_index, 0)
        self.assertEqual(candidate.compared_digits, 20)
        self.assertEqual(candidate.lhs_value, candidate.rhs_value)

    def test_key_collision_is_rejected_by_high_precision_refinement(self) -> None:
        spec = replace(
            _rational_gcf_search_spec(),
            constants=("1.4138", "1"),
            key_scale=100,
        )
        self.assertEqual(search_gcf_candidates(spec), ())

    def test_pipeline_match_stage_persists_candidates_not_relation_claims(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            catalog = Catalog(root / "catalog.sqlite")
            try:
                store = TermStore(root / "terms", catalog)
                index = SequenceIndex(catalog, store)
                ingested = index.ingest_lines(
                    iter(
                        (
                            "%I A000001 M0000\n",
                            "%S A000001 1,2,3,4\n",
                            "%O A000001 0\n",
                        )
                    ),
                    snapshot_id="ramanujan-truth",
                    source_revision="legacy-vector-v1",
                )
                self.assertEqual(ingested.parsed, 1)

                coordinator = PipelineCoordinator(catalog, store)
                spec = PipelineSpec(
                    run_id="ramanujan-truth-run",
                    snapshot_id="ramanujan-truth",
                    stages=("fingerprint", "match"),
                    fingerprint_profiles=(
                        FingerprintProfile(
                            name="exact-prefix",
                            version="1",
                            mode="exact",
                            term_count=3,
                        ),
                    ),
                    ramanujan_search_specs=(_rational_gcf_search_spec(),),
                    page_size=1,
                )
                report = coordinator.execute(
                    coordinator.plan(spec),
                    workers=2,
                )
                self.assertEqual(report.state, "complete")
                stored = catalog.fetch_all(
                    """
                    SELECT lhs_key, lhs_value, rhs_value, compared_digits
                    FROM cf_search_candidate
                    WHERE run_id = ?
                    """,
                    (spec.run_id,),
                )
                self.assertEqual(len(stored), 1)
                self.assertEqual(stored[0]["lhs_value"], stored[0]["rhs_value"])
                self.assertEqual(stored[0]["compared_digits"], 20)
                self.assertEqual(
                    catalog.fetch_all(
                        "SELECT claim_id FROM relation_claim WHERE created_by_run = ?",
                        (spec.run_id,),
                    ),
                    [],
                )
            finally:
                catalog.close()


def _rational_gcf_search_spec() -> CFSearchSpec:
    return CFSearchSpec(
        constants=(
            "1.4137931034482758620689655172413793103448275862068965517241",
            "1",
        ),
        lhs_forms=(LinearForm((1, 0), (0, 1)),),
        rhs_series=(
            GCFSeries((1, 2, 2, 2, 2), (0, 1, 1, 1, 1)),
        ),
        key_scale=100_000,
        initial_dps=50,
        verify_dps=50,
        verify_digits=20,
    )


if __name__ == "__main__":
    unittest.main()
