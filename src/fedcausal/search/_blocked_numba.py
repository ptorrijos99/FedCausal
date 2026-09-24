"""Semi-directed reachability compiled with numba, for large graphs.

Used by :func:`fedcausal.search.gesops.is_semidirected_blocked` when numba is
installed (the ``accel`` extra); it returns the same answer as the pure-Python
walk, about 4x faster at a few hundred variables.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from fedcausal.graph.pdag import PDAG

try:
    from numba import njit

    HAVE_NUMBA = True
except ImportError:  # pragma: no cover - exercised only without the accel extra
    HAVE_NUMBA = False

__all__ = ["HAVE_NUMBA", "blocked_numba"]

_EMPTY = np.zeros(0, dtype=np.int32)


if HAVE_NUMBA:

    @njit(cache=True)
    def _kernel(indptr, indices, n, y, x, cond_arr):  # type: ignore[no-untyped-def]  # noqa: ANN001, ANN202
        """Return whether every path from `y` to `x` in the successor CSR hits `cond`."""
        visited = np.zeros(n, dtype=np.bool_)
        for i in range(cond_arr.shape[0]):
            visited[cond_arr[i]] = True
        if visited[y]:
            return True
        if y == x:
            return False
        stack = np.empty(n, dtype=np.int32)
        sp = 0
        stack[sp] = y
        sp += 1
        visited[y] = True
        while sp > 0:
            sp -= 1
            v = stack[sp]
            for k in range(indptr[v], indptr[v + 1]):
                w = indices[k]
                if visited[w]:
                    continue
                if w == x:
                    return False
                visited[w] = True
                stack[sp] = w
                sp += 1
        return True

    def blocked_numba(pdag: PDAG, y: int, x: int, cond: frozenset[int]) -> bool:
        """Return :func:`fedcausal.search.gesops.is_semidirected_blocked`'s answer."""
        cond_arr = np.fromiter(cond, dtype=np.int32, count=len(cond)) if cond else _EMPTY
        indptr, indices = pdag.successor_csr()
        return bool(_kernel(indptr, indices, pdag.num_nodes, y, x, cond_arr))

else:  # pragma: no cover - only when the accel extra is not installed

    def blocked_numba(pdag: PDAG, y: int, x: int, cond: frozenset[int]) -> bool:
        """Raise: numba is not installed. Check :data:`HAVE_NUMBA` first."""
        raise RuntimeError("blocked_numba requires the optional 'numba' extra (fedcausal[accel])")
