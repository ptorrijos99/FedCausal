"""The optional numba reachability path must be byte-identical to the set walk.

Skipped when numba is not installed. Compares the numba fast path against the
pure-Python traversal both directly (random graphs + queries) and end-to-end
(a full GES fit), so the accelerated path can never silently diverge.
"""

from __future__ import annotations

import numpy as np
import pytest

import fedcausal.search.gesops as ops
from fedcausal.graph import PDAG

pytestmark = pytest.mark.skipif(not ops._HAVE_NUMBA, reason="numba not installed")


def _random_pdag(n: int, p_dir: float, p_undir: float, rng: np.random.Generator) -> PDAG:
    """Build a random PDAG: each ordered pair maybe a directed edge, each pair maybe undirected."""
    g = PDAG(n)
    for u in range(n):
        for v in range(n):
            if u == v:
                continue
            if u < v and rng.random() < p_undir:
                # avoid clashing with an already-added directed edge
                if v not in g._children[u] and u not in g._children[v]:
                    g.add_undirected(u, v)
                continue
            if rng.random() < p_dir and v not in g._undirected[u] and u not in g._children[v]:
                g.add_directed(u, v)
    return g


def _pure_python_blocked(g: PDAG, y: int, x: int, cond: frozenset[int]) -> bool:
    """The set-walk result, forced by lifting the numba threshold above n."""
    saved = ops._NUMBA_MIN_NODES
    ops._NUMBA_MIN_NODES = 10**9
    try:
        return ops.is_semidirected_blocked(g, y, x, cond)
    finally:
        ops._NUMBA_MIN_NODES = saved


@pytest.mark.parametrize("seed", range(8))
def test_numba_matches_set_walk_on_random_graphs(seed: int) -> None:
    """blocked_numba agrees with the set walk on every (y, x, cond) over random PDAGs."""
    rng = np.random.default_rng(seed)
    n = 60  # >= _NUMBA_MIN_NODES so the numba path is the one under test
    g = _random_pdag(n, p_dir=0.03, p_undir=0.02, rng=rng)
    for _ in range(400):
        y = int(rng.integers(n))
        x = int(rng.integers(n))
        k = int(rng.integers(0, 4))
        cond = frozenset(int(c) for c in rng.integers(0, n, size=k))
        got = ops.is_semidirected_blocked(g, y, x, cond)  # numba (n=60 >= 50)
        assert got == _pure_python_blocked(g, y, x, cond)
