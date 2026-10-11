"""Discovery module for parametric towers, master array synthesis, and results storage."""

from .results_store import ResultsDatabase
from .tower_detector import (
    GraphTopologyTemplate,
    LinearTowerTemplate,
    ParametricTemplate,
    detect_towers_and_missing_teeth,
    expand_gf_denumerant,
    harvest_artifacts_into_results_db,
    probe_global_oeis,
    synthesize_master_array_antidiagonals,
)

__all__ = [
    "ResultsDatabase",
    "ParametricTemplate",
    "LinearTowerTemplate",
    "GraphTopologyTemplate",
    "expand_gf_denumerant",
    "probe_global_oeis",
    "synthesize_master_array_antidiagonals",
    "detect_towers_and_missing_teeth",
    "harvest_artifacts_into_results_db",
]
