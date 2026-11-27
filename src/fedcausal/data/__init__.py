"""The splits that simulate sites.

The datasets themselves and their encoding are causalbase's
(:mod:`causalbase.data`).
"""

from __future__ import annotations

from fedcausal.data.partition import contiguous_split, dirichlet_split

__all__ = ["contiguous_split", "dirichlet_split"]
