"""Data handling utilities for causal structure learning."""

from __future__ import annotations

from fedcausal.data.dataset import DiscreteDataset
from fedcausal.data.partition import contiguous_split, dirichlet_split

__all__ = ["DiscreteDataset", "contiguous_split", "dirichlet_split"]
