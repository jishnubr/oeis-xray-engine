"""Full-corpus denumerant sweep runner with results database decoupling and task pruning.

Orchestrates sequential 10k-sequence slices across up to 400,000 OEIS sequences:
  1. Plans each slice with mathematical prefiltering (pure bosonic Euler products).
  2. Executes with multiprocessing workers.
  3. Extracts certified denumerant weights into discoveries.sqlite.
  4. Detects parametric towers, probes the 399k corpus for missing teeth, and synthesizes master arrays.
  5. Prunes completed task manifests from the operational catalog to keep it lean (< 870 MB).
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import sqlite3
import sys
import time
from typing import Any, Sequence

from ..cli import _run_pipeline
from ..models import TransformSpec
from ..pipeline import PipelineCoordinator, PipelineSpec, PipelineStage
from ..storage.catalog import Catalog
from ..storage.term_store import TermStore
from .results_store import ResultsDatabase
from .tower_detector import (
    detect_towers_and_missing_teeth,
    harvest_artifacts_into_results_db,
)

_LOGGER = logging.getLogger("oeis_xray_engine.discovery.corpus_sweep")


def cleanup_slice_tasks(catalog: Catalog, run_id: str) -> int:
    """Delete completed task manifest rows from catalog to reclaim space.

    Only deletes rows from the 'task' table for the specified run_id.
    Preserves the 'run' record, 'derived_artifact' records, and catalog integrity.
    Returns the number of deleted task rows.
    """
    with catalog.transaction():
        # Check if task rows exist
        count_row = catalog.fetch_one(
            "SELECT COUNT(*) AS n FROM task WHERE run_id = ?",
            (run_id,),
        )
        task_count = count_row["n"] if count_row else 0
        if task_count > 0:
            catalog.execute("DELETE FROM task WHERE run_id = ?", (run_id,))
    return task_count


def run_full_corpus_sweep(
    start_slice: int = 4,
    end_slice: int = 40,
    slice_size: int = 10000,
    total_sequences: int = 400000,
    workers: int = 4,
    max_terms: int = 32,
    max_k: int = 25,
    snapshot_id: str = "full-corpus-v1",
    catalog_path: str | Path = "oeis_xray_catalog.sqlite",
    terms_dir: str | Path = "oeis_xray_terms",
    results_db_path: str | Path = "discoveries.sqlite",
    prune_tasks: bool = True,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Execute the full-corpus automated denumerant sweep."""
    catalog_file = Path(catalog_path)
    terms_path = Path(terms_dir)
    results_file = Path(results_db_path)

    if not catalog_file.exists():
        raise FileNotFoundError(f"Catalog file not found: {catalog_file}")

    print("=" * 76)
    print("      OEIS X-RAY FULL-CORPUS PARAMETRIC DISCOVERY SWEEP (to 400k)     ")
    print("=" * 76)
    print(f"  Operational Catalog : {catalog_file} ({catalog_file.stat().st_size / (1024*1024):.1f} MB)")
    print(f"  Discoveries DB      : {results_file}")
    print(f"  Slice Range         : Slices {start_slice:02d} to {end_slice:02d} (of {slice_size:,} sequences each)")
    print(f"  Workers             : {workers}")
    print(f"  Task Pruning        : {'Enabled (reclaims task manifests)' if prune_tasks else 'Disabled'}")
    print("=" * 76)

    catalog = Catalog(catalog_file)
    term_store = TermStore(terms_path, catalog)
    results_db = ResultsDatabase(results_file)

    if dry_run:
        print("\n[!] Dry run requested. Slices to process:")
        for slice_idx in range(start_slice, end_slice + 1):
            start_num = (slice_idx - 1) * slice_size + 1
            end_num = min(slice_idx * slice_size, total_sequences)
            print(f"    Slice {slice_idx:02d}: A{start_num:06d} -> A{end_num:06d} (run: denumerant-slice-v{slice_idx})")
        results_db.close()
        catalog.close()
        return {"status": "dry_run"}

    # Initial harvesting pass on any existing runs
    print("\n[*] Synchronizing existing completed artifacts into discoveries.sqlite...")
    existing_harvested = harvest_artifacts_into_results_db(catalog, term_store, results_db)
    if existing_harvested > 0:
        print(f"    [+] Ingested {existing_harvested} certified denumerants from existing catalog runs.")
        detect_towers_and_missing_teeth(results_db, catalog, max_k=max_k)

    counts = results_db.get_counts()
    print(f"    Current Discoveries: {counts['certified_denumerants']} certified | "
          f"{counts['parametric_towers']} towers | {counts['missing_teeth']} missing teeth | "
          f"{counts['master_arrays']} master arrays")

    coordinator = PipelineCoordinator(catalog, term_store)
    transform_spec = TransformSpec(
        name="denumerant",
        version="1",
        parameters={
            "input_term_count": max_terms,
            "term_cap": max_terms,
            "confidence_window": 5,
        },
        input_domain="finite_integer_sequence_pure_bosonic_euler_product",
        output_bound=100000,
    )

    total_slices = end_slice - start_slice + 1
    sweep_start = time.monotonic()
    completed_slices = 0

    try:
        for slice_idx in range(start_slice, end_slice + 1):
            start_num = (slice_idx - 1) * slice_size + 1
            end_num = min(slice_idx * slice_size, total_sequences)
            start_id = f"A{start_num:06d}"
            end_id = f"A{end_num:06d}"
            run_id = f"denumerant-slice-v{slice_idx}"

            slice_start_time = time.monotonic()
            print(f"\n" + "-" * 76)
            print(f"  SLICE {slice_idx:02d}/{end_slice:02d}: {start_id} -> {end_id}  (Run: {run_id})")
            print("-" * 76)

            # Check if this run is already completed in catalog
            existing_run = catalog.fetch_one(
                "SELECT state FROM run WHERE run_id = ?",
                (run_id,),
            )
            already_done = existing_run and existing_run["state"] == "complete"

            if already_done:
                print(f"  [*] Run {run_id} is already complete in catalog. Harvesting artifacts...")
                new_certified = harvest_artifacts_into_results_db(catalog, term_store, results_db, run_id=run_id)
            else:
                # 1. Plan slice
                print(f"  [1/4] Planning {start_id}..{end_id} with denumerant prefilters...")
                spec = PipelineSpec(
                    run_id=run_id,
                    snapshot_id=snapshot_id,
                    stages=("transform",),
                    transform_specs=(transform_spec,),
                    fingerprint_profiles=(),
                    ramanujan_search_specs=(),
                    start_sequence_id=start_id,
                    end_sequence_id=end_id,
                )
                t_plan_0 = time.monotonic()
                coordinator.plan(spec)
                t_plan = time.monotonic() - t_plan_0
                print(f"        Planned in {t_plan:.2f}s.")

                # 2. Execute slice with worker pool
                print(f"  [2/4] Executing with {workers} multiprocessing workers...")
                t_exec_0 = time.monotonic()
                result = _run_pipeline(coordinator, run_id, workers, resume=False)
                t_exec = time.monotonic() - t_exec_0
                print(f"        Executed in {t_exec:.2f}s — state: {result.state}, tasks completed: {result.newly_completed}.")

                # 3. Harvest discoveries into discoveries.sqlite
                print(f"  [3/4] Harvesting certified denumerants into {results_file.name}...")
                new_certified = harvest_artifacts_into_results_db(catalog, term_store, results_db, run_id=run_id)

            # 4. Detect towers, missing teeth & synthesize master arrays
            detection_summary = detect_towers_and_missing_teeth(results_db, catalog, max_k=max_k)

            # 5. Prune task rows to keep catalog lean
            pruned_count = 0
            if prune_tasks:
                pruned_count = cleanup_slice_tasks(catalog, run_id)

            slice_elapsed = time.monotonic() - slice_start_time
            completed_slices += 1
            counts = results_db.get_counts()
            cat_mb = catalog_file.stat().st_size / (1024 * 1024)

            print(f"  [+] SLICE {slice_idx:02d} DONE in {slice_elapsed:.1f}s | "
                  f"+{new_certified} new certified | "
                  f"Total: {counts['certified_denumerants']} certified, "
                  f"{counts['missing_teeth']} missing teeth, "
                  f"{counts['master_arrays']} master arrays | "
                  f"Catalog: {cat_mb:.1f} MB (pruned {pruned_count} tasks)")

    except KeyboardInterrupt:
        print("\n\n" + "!" * 76)
        print("  SWEEP INTERRUPTED BY USER (Ctrl+C).")
        print(f"  Completed {completed_slices} slices before interruption.")
        print("  All discoveries made so far are safely persisted in discoveries.sqlite.")
        print(f"  To resume later, run with: --start-slice {slice_idx}")
        print("!" * 76)
        return {"status": "interrupted", "completed_slices": completed_slices}

    finally:
        total_time = time.monotonic() - sweep_start
        print("\n" + "=" * 76)
        print(f"  SWEEP COMPLETE: {completed_slices}/{total_slices} slices in {total_time:.1f}s ({total_time / 60:.1f} min)")
        print("=" * 76)
        print(results_db.summary_report())
        results_db.close()
        catalog.close()

    return {
        "status": "complete",
        "completed_slices": completed_slices,
        "total_time_seconds": total_time,
    }
