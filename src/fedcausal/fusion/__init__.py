"""Structural fusion of causal graphs learned across sites (Puerta et al., 2021).

Every input is transformed into one consensus order
(:func:`~fedcausal.fusion.alpha.alpha_order`,
:func:`~fedcausal.fusion.beta.beta_to_alpha`) and the transformed parent sets
are unioned (:func:`consensus_union`). The budgeted variants add only the most
agreed edges; :func:`intersection_fusion` keeps only what every input shares.
"""

from fedcausal.fusion.alpha import alpha_order
from fedcausal.fusion.base import FusionOperator
from fedcausal.fusion.beta import beta_to_alpha
from fedcausal.fusion.consensus import (
    Consensus,
    ConsensusUnion,
    MaxEdges,
    MaxFrequency,
    MaxParents,
    consensus_pct,
    consensus_union,
    edge_frequency,
    max_edges,
    max_frequency,
    max_parents,
)
from fedcausal.fusion.intersection import Intersection, intersection_fusion

__all__ = [
    "FusionOperator",
    "alpha_order",
    "beta_to_alpha",
    "consensus_union",
    "ConsensusUnion",
    "edge_frequency",
    "max_edges",
    "max_frequency",
    "consensus_pct",
    "max_parents",
    "MaxEdges",
    "MaxFrequency",
    "Consensus",
    "MaxParents",
    "intersection_fusion",
    "Intersection",
]
