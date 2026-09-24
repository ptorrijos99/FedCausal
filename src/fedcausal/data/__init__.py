"""Datasets and the splits that simulate sites."""

from __future__ import annotations

from fedcausal.data.continuous import ContinuousDataset
from fedcausal.data.dataset import DiscreteDataset
from fedcausal.data.partition import contiguous_split, dirichlet_split

__all__ = ["ContinuousDataset", "DiscreteDataset", "contiguous_split", "dirichlet_split"]
