"""Parametric Tower & Master Array Detector.

Identifies parametric families of generating functions from extracted denumerants,
detects uncataloged "Missing Teeth" across the entire 399k OEIS catalog,
and synthesizes 2D Master Arrays flattened by antidiagonals.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import logging
from pathlib import Path
from typing import Sequence

from ..models import SequenceRef, TermSeries
from ..indexing.hash_index import FingerprintProfile, HashIndex
from ..pipeline import _read_artifact_series
from ..storage.catalog import Catalog
from ..storage.term_store import TermStore
from .results_store import ResultsDatabase

_LOGGER = logging.getLogger("oeis_xray_engine.discovery.tower_detector")


class ParametricTemplate:
    """Base class for identifying parametric families of weight multisets."""

    signature: str = ""

    def match(self, weights: list[int]) -> tuple[bool, int | None, str]:
        """Return (is_match, parameter_k, template_signature)."""
        raise NotImplementedError

    def reconstruct_weights(self, k: int) -> list[int]:
        """Reconstruct the expected weight multiset for parameter k."""
        raise NotImplementedError


class LinearTowerTemplate(ParametricTemplate):
    """Detects families with a fixed base and a variable parameter k with multiplicity m.

    Example: base=[1, 1], m=2 -> [1, 1, k, k]
    Matches A000292 (k=1), A006918 (k=2), A144677 (k=3), etc.
    """

    def __init__(
        self,
        base_weights: list[int],
        var_multiplicity: int = 1,
        name: str | None = None,
    ) -> None:
        self.base_weights = sorted(base_weights)
        self.var_multiplicity = var_multiplicity
        base_str = ",".join(str(w) for w in self.base_weights)
        k_str = f"k^{var_multiplicity}" if var_multiplicity > 1 else "k"
        self.signature = name or f"Tower_[{base_str},{k_str}]"

    def match(self, weights: list[int]) -> tuple[bool, int | None, str]:
        freq = Counter(weights)
        base_freq = Counter(self.base_weights)

        for val, count in base_freq.items():
            if freq[val] < count:
                return False, None, ""
            freq[val] -= count
            if freq[val] == 0:
                del freq[val]

        if len(freq) == 1:
            k_val, k_count = list(freq.items())[0]
            if k_count == self.var_multiplicity:
                return True, k_val, self.signature

        return False, None, ""

    def reconstruct_weights(self, k: int) -> list[int]:
        return sorted(self.base_weights + [k] * self.var_multiplicity)


class GraphTopologyTemplate(ParametricTemplate):
    """Detects complete graph exterior gradings: [1]*k + [2]*C(k, 2)."""

    def __init__(self) -> None:
        self.signature = "Graph_Complete_K_k"

    def match(self, weights: list[int]) -> tuple[bool, int | None, str]:
        freq = Counter(weights)
        if set(freq.keys()) == {1, 2}:
            k = freq[1]
            edges = freq[2]
            if k >= 2 and edges == (k * (k - 1)) // 2:
                return True, k, self.signature
        return False, None, ""

    def reconstruct_weights(self, k: int) -> list[int]:
        edges = (k * (k - 1)) // 2
        return [1] * k + [2] * edges


def get_default_templates() -> list[ParametricTemplate]:
    """Return standard library of mathematical parametric templates."""
    return [
        LinearTowerTemplate(base_weights=[1], var_multiplicity=1, name="[1, k] (2-Part Frobenius / Floor Division)"),
        LinearTowerTemplate(base_weights=[1, 1], var_multiplicity=2, name="[1, 1, k, k] (A400322 Family)"),
        LinearTowerTemplate(base_weights=[1, 2], var_multiplicity=1, name="[1, 2, k] (Quarter-Square Slices)"),
        LinearTowerTemplate(base_weights=[1, 1, 1], var_multiplicity=1, name="[1, 1, 1, k] (Tetrahedral Extensions)"),
        LinearTowerTemplate(base_weights=[1, 1], var_multiplicity=1, name="[1, 1, k] (Triangular Slices)"),
        LinearTowerTemplate(base_weights=[1], var_multiplicity=2, name="[1, k, k] (Dual Partitions)"),
        LinearTowerTemplate(base_weights=[1, 2, 3], var_multiplicity=1, name="[1, 2, 3, k] (Euler Slices)"),
        LinearTowerTemplate(base_weights=[2, 2], var_multiplicity=2, name="[2, 2, k, k] (Even Anisotropic)"),
        LinearTowerTemplate(base_weights=[1, 3], var_multiplicity=1, name="[1, 3, k] (Ternary Step Partitions)"),
        LinearTowerTemplate(base_weights=[1, 2, 4], var_multiplicity=1, name="[1, 2, 4, k] (Binary Expansion Towers)"),
        GraphTopologyTemplate(),
    ]


def expand_gf_denumerant(weights: list[int], max_terms: int = 32) -> list[int]:
    """Compute the first max_terms of the generating function Product 1/(1 - x^w)."""
    dp = [0] * max_terms
    dp[0] = 1
    for w in sorted(weights):
        if w <= 0:
            continue
        for i in range(w, max_terms):
            dp[i] += dp[i - w]
    return dp


def probe_global_oeis(
    catalog: Catalog,
    weights: list[int],
    profile: FingerprintProfile,
) -> list[str]:
    """Check if the given weight generating function exists in the full 399k catalog."""
    terms = expand_gf_denumerant(weights, max_terms=profile.term_count)
    if len(terms) < profile.term_count:
        return []

    dummy_ref = SequenceRef("SEARCH", "search_rev", "full-corpus-v1")
    series = TermSeries(
        ref=dummy_ref,
        start_index=0,
        values=tuple(terms),
        complete_through=profile.term_count - 1,
    )
    digest = HashIndex.digest_series(series, profile)

    rows = catalog.fetch_all(
        """
        SELECT s.sequence_id
        FROM fingerprint f
        JOIN sequence_revision r ON r.revision_id = f.revision_id
        JOIN sequence s ON s.sequence_id = r.sequence_id
        WHERE f.profile_key = ? AND f.digest = ?
        ORDER BY s.sequence_id
        """,
        (profile.profile_key, digest),
    )
    return [r["sequence_id"] for r in rows]


def synthesize_master_array_antidiagonals(
    template: ParametricTemplate,
    max_s: int = 15,
) -> list[tuple[int, int, int]]:
    """Flatten 2D grid A(n, k) along antidiagonals s = n + k.

    Returns a list of (n, k, A(n, k)).
    """
    results: list[tuple[int, int, int]] = []
    cache: dict[int, list[int]] = {}

    for s in range(2, max_s + 1):
        for k in range(1, s):
            n = s - k
            if k not in cache:
                k_weights = template.reconstruct_weights(k)
                cache[k] = expand_gf_denumerant(k_weights, max_terms=max_s + 1)
            val = cache[k][n] if n < len(cache[k]) else 0
            results.append((n, k, val))
    return results


def harvest_artifacts_into_results_db(
    catalog: Catalog,
    term_store: TermStore,
    results_db: ResultsDatabase,
    run_id: str | None = None,
) -> int:
    """Harvest certified denumerant weights from catalog into discoveries.sqlite.

    Returns the count of newly certified sequences added.
    """
    run_filter = "AND a.created_by_run = ?" if run_id is not None else ""
    params = (run_id,) if run_id is not None else ()

    rows = catalog.fetch_all(
        f"""
        SELECT a.artifact_id, s.sequence_id, r.snapshot_id, a.term_manifest_json
        FROM derived_artifact a
        JOIN sequence_revision r ON r.revision_id = a.revision_id
        JOIN sequence s ON s.sequence_id = r.sequence_id
        JOIN transform_spec ts ON ts.transform_key = a.transform_key
        WHERE ts.name = 'denumerant'
        {run_filter}
        ORDER BY s.sequence_id
        """,
        params,
    )

    new_count = 0
    for r in rows:
        seq_id = str(r["sequence_id"])
        if results_db.is_sequence_certified(seq_id):
            continue
        series = _read_artifact_series(catalog, term_store, r["artifact_id"], r["snapshot_id"])
        weights = list(series.values)
        if not weights:
            continue
        try:
            meta = json.loads(r["term_manifest_json"]).get("_transformer_metadata", {})
            confidence = int(meta.get("confidence_terms", len(weights)))
        except Exception:
            confidence = len(weights)
        is_new = results_db.record_certified_denumerant(
            seq_id, weights, confidence_terms=confidence
        )
        if is_new:
            new_count += 1
    return new_count


def detect_towers_and_missing_teeth(
    results_db: ResultsDatabase,
    catalog: Catalog,
    templates: Sequence[ParametricTemplate] | None = None,
    profile: FingerprintProfile | None = None,
    max_k: int = 25,
) -> dict[str, int]:
    """Scan certified denumerants in discoveries.sqlite, identify parametric towers,
    probe for missing teeth against the catalog, and synthesize master arrays.

    Returns summary dictionary of discovery counts.
    """
    if templates is None:
        templates = get_default_templates()
    if profile is None:
        profile = FingerprintProfile(
            name="exact-prefix",
            version="1",
            mode="exact",
            term_count=32,
        )

    certified = results_db.get_certified_denumerants()
    if not certified:
        return {
            "towers": 0,
            "new_towers": 0,
            "missing_teeth": 0,
            "master_arrays": 0,
            "cataloged_entries": 0,
        }

    towers: dict[str, tuple[ParametricTemplate, list[tuple[int, str, list[int]]]]] = {}

    for item in certified:
        seq_id = item["sequence_id"]
        weights = item["weights"]
        for tmpl in templates:
            is_match, k, sig = tmpl.match(weights)
            if is_match and k is not None:
                if sig not in towers:
                    towers[sig] = (tmpl, [])
                towers[sig][1].append((k, seq_id, weights))
                break

    new_towers = 0
    new_missing = 0
    new_arrays = 0
    new_entries = 0

    for sig, (template, members) in sorted(towers.items()):
        base_weights = getattr(template, "base_weights", [])
        var_multiplicity = getattr(template, "var_multiplicity", 1)
        if results_db.record_tower(
            sig, template.__class__.__name__, base_weights, var_multiplicity
        ):
            new_towers += 1

        by_k: dict[int, list[str]] = defaultdict(list)
        for k, seq_id, _ in members:
            by_k[k].append(seq_id)

        known_ks = sorted(by_k.keys())
        for k in known_ks:
            seq_list = sorted(set(by_k[k]))
            expected_weights = template.reconstruct_weights(k)
            terms = expand_gf_denumerant(expected_weights, max_terms=32)
            gf_str = f"Product 1/(1-x^w) for w in {expected_weights}"
            for seq_id in seq_list:
                if results_db.record_tower_entry(
                    sig, k, "cataloged", seq_id, terms, gf_str
                ):
                    new_entries += 1

        probe_limit = max(max(known_ks) + 2, max_k)
        missing_ks = [k for k in range(1, probe_limit + 1) if k not in by_k]

        has_missing_teeth = False
        for k in missing_ks:
            expected_weights = template.reconstruct_weights(k)
            terms = expand_gf_denumerant(expected_weights, max_terms=32)
            gf_str = f"Product 1/(1-x^w) for w in {expected_weights}"
            global_matches = probe_global_oeis(catalog, expected_weights, profile)

            if global_matches:
                for match_seq in sorted(set(global_matches)):
                    results_db.record_tower_entry(
                        sig, k, "cataloged", match_seq, terms, gf_str
                    )
            else:
                if results_db.record_tower_entry(
                    sig, k, "missing_tooth", None, terms, gf_str
                ):
                    new_missing += 1
                has_missing_teeth = True

        if len(known_ks) >= 2 or has_missing_teeth:
            array_id = f"MA_{template.signature.replace(' ', '_').replace('/', '_')}"
            title = f"Master Array for {template.signature}"
            antidiags = synthesize_master_array_antidiagonals(template, max_s=15)
            antidiag_terms = [val for _, _, val in antidiags]
            crossrefs = sorted(set(seq for seqs in by_k.values() for seq in seqs))
            gf_formula = f"G_k(x) = Product 1/(1-x^w) for family {template.signature}"
            if results_db.record_master_array(
                array_id, title, gf_formula, None, antidiag_terms, crossrefs
            ):
                new_arrays += 1

    return {
        "towers": len(towers),
        "new_towers": new_towers,
        "missing_teeth": new_missing,
        "master_arrays": new_arrays,
        "cataloged_entries": new_entries,
    }


def run_tower_detection(
    catalog_path: str = "oeis_xray_catalog.sqlite",
    terms_dir: str = "oeis_xray_terms",
    run_id: str = "all",
    results_db_path: str = "discoveries.sqlite",
    max_k_probe: int = 15,
) -> None:
    """Scan extracted denumerants, group into towers, and detect missing teeth."""
    catalog = Catalog(catalog_path, initialize=False, read_only=True)
    term_store = TermStore(terms_dir, catalog)
    results_db = ResultsDatabase(results_db_path)

    print(f"\n[1/3] Harvesting denumerant artifacts ({run_id!r}) into {results_db_path}...")
    run_arg = None if run_id == "all" else run_id
    new_certified = harvest_artifacts_into_results_db(catalog, term_store, results_db, run_id=run_arg)
    print(f"       Harvested {new_certified} new certified denumerants.")

    print(f"\n[2/3] Detecting parametric towers and probing missing teeth up to k={max_k_probe}...")
    summary = detect_towers_and_missing_teeth(
        results_db, catalog, max_k=max_k_probe
    )
    print(f"       Towers active: {summary['towers']}")
    print(f"       Missing teeth uncataloged: {summary['missing_teeth']}")
    print(f"       Master arrays synthesized: {summary['master_arrays']}")

    print(f"\n[3/3] Results Database Status ({results_db_path}):")
    print(results_db.summary_report())

    results_db.close()
    catalog.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Parametric Tower & Master Array Detector")
    parser.add_argument("--catalog", default="oeis_xray_catalog.sqlite", help="Path to SQLite catalog")
    parser.add_argument("--terms", default="oeis_xray_terms", help="Path to terms store directory")
    parser.add_argument("--run-id", default="all", help="Run ID of denumerant artifacts (or 'all' for all)")
    parser.add_argument("--results-db", default="discoveries.sqlite", help="Path to discoveries database")
    parser.add_argument("--max-k", type=int, default=15, help="Maximum k to probe for missing teeth")
    args = parser.parse_args()

    run_tower_detection(
        catalog_path=args.catalog,
        terms_dir=args.terms,
        run_id=args.run_id,
        results_db_path=args.results_db,
        max_k_probe=args.max_k,
    )


if __name__ == "__main__":
    main()
