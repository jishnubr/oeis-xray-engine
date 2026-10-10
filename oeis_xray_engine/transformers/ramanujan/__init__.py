"""Ramanujan Machine generalized continued-fraction search components.

The search functions in this package produce numerical conjecture candidates.
They do not establish an infinite identity and must not be promoted to an
X-Ray E1 relation without a separate exact verifier.
"""

from .gcf import (
    EfficientGCF,
    GeneralizedContinuedFraction,
    MobiusTransform,
    efficient_gcf_fraction,
    efficient_gcf_numerator_denominator,
)
from .search import (
    BloomKeyIndex,
    CFSearchCandidate,
    CFSearchSpec,
    GCFSeries,
    LinearForm,
    search_gcf_candidates,
)

__all__ = [
    "BloomKeyIndex",
    "CFSearchCandidate",
    "CFSearchSpec",
    "EfficientGCF",
    "GCFSeries",
    "GeneralizedContinuedFraction",
    "LinearForm",
    "MobiusTransform",
    "efficient_gcf_fraction",
    "efficient_gcf_numerator_denominator",
    "search_gcf_candidates",
]
