"""Structural fusion of causal graphs learned across sites.

Ports the reference implementation's canonical structural fusion (Puerta et al., Information Fusion
2021): a consensus topological order (:func:`~fedcausal.fusion.alpha.alpha_order`),
a minimal-I-map transform of each DAG into that order
(:func:`~fedcausal.fusion.beta.beta_to_alpha`), and the union of the transformed
parent sets (:func:`consensus_union`), plus four limited-budget post-filters.
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
