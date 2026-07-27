"""Input/output utilities for graphs and datasets."""

from __future__ import annotations

from fedcausal.io.sampling import forward_sample
from fedcausal.io.xbif import DiscreteBN, load_xbif

__all__ = ["DiscreteBN", "forward_sample", "load_xbif"]
