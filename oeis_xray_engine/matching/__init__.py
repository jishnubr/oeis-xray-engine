"""Exact verification and bounded transform-bridge discovery."""

from .bridge_finder import BridgeFinder, BridgeResult
from .collision_detector import (
    CollisionDetector,
    RelationSpec,
    VerificationResult,
)

__all__ = [
    "BridgeFinder",
    "BridgeResult",
    "CollisionDetector",
    "RelationSpec",
    "VerificationResult",
]
