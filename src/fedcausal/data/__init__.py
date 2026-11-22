"""Datasets, their encoding, and the splits that simulate sites."""

from __future__ import annotations

from fedcausal.data.continuous import ContinuousDataset, as_continuous, as_continuous_sites
from fedcausal.data.dataset import DiscreteDataset, as_discrete, as_discrete_sites
from fedcausal.data.partition import contiguous_split, dirichlet_split

__all__ = [
    "ContinuousDataset",
    "DiscreteDataset",
    "as_continuous",
    "as_continuous_sites",
    "as_discrete",
    "as_discrete_sites",
    "contiguous_split",
    "dirichlet_split",
]
