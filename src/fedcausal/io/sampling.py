"""Forward (ancestral) sampling from a loaded discrete Bayesian network.

Draws a synthetic :class:`~fedcausal.data.DiscreteDataset` from a
:class:`~fedcausal.io.xbif.DiscreteBN` by sampling each variable, in
topological order, conditional on its already-sampled parents. Sampling is
vectorized over rows: for a given variable, all `n` draws are produced by
one batch of `numpy` operations rather than a Python loop over rows. The
only Python-level loop is over the (small, fixed) number of variables in
the network, and it never recurses, so there is no recursion-depth limit
tied to network size.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from fedcausal.data import DiscreteDataset

if TYPE_CHECKING:
    from fedcausal.io.xbif import DiscreteBN

__all__ = ["forward_sample"]


def _parent_row_indices(
    codes: np.ndarray, parents: tuple[int, ...], cardinalities: list[int]
) -> np.ndarray:
    """Map each row's parent codes to a CPT row index.

    Parameters
    ----------
    codes : np.ndarray
        Sampled outcome codes so far, shape ``(n, num_nodes)``, in XBIF
        outcome order (column `p` holds variable `p`'s sampled code).
    parents : tuple[int, ...]
        Parent node indices, in `GIVEN` declaration order (matches
        `DiscreteBN.parent_order[j]`).
    cardinalities : list[int]
        Outcome counts for every node, indexed by node id.

    Returns
    -------
    np.ndarray
        Row index into the child's CPT for every sample, shape ``(n,)``,
        computed as the row-major (first parent slowest) flattening of the
        parent codes -- matching the layout `load_xbif` uses when it
        reshapes each `<TABLE>` to ``(q, r)``.
    """
    n = codes.shape[0]
    if not parents:
        return np.zeros(n, dtype=np.int64)
    dims = tuple(cardinalities[p] for p in parents)
    multi_index = tuple(codes[:, p] for p in parents)
    return np.ravel_multi_index(multi_index, dims).astype(np.int64)


def forward_sample(bn: DiscreteBN, n: int, seed: int) -> DiscreteDataset:
    """Draw `n` i.i.d. samples from `bn` by ancestral (forward) sampling.

    Parameters
    ----------
    bn : DiscreteBN
        The network to sample from, as returned by
        :func:`fedcausal.io.xbif.load_xbif`.
    n : int
        Number of rows to sample.
    seed : int
        Seed for the random number generator. The same `seed` always
        produces bit-identical output for a given `bn` and `n`.

    Returns
    -------
    DiscreteDataset
        The sampled dataset. Its `names` are `bn.names`, in the same
        node order. Its `categories` are `bn`'s outcome labels **sorted
        lexicographically per variable** (not the XBIF declaration
        order), with `values` re-encoded to match -- so the result
        follows the same level-encoding convention as
        `DiscreteDataset.from_csv`/`from_pandas`.

    Notes
    -----
    Each variable is sampled once, in one topological pass, using a
    single vectorized batch of `n` draws (inverse-CDF sampling against
    the variable's CPT rows selected by each sample's already-sampled
    parent codes). No per-row Python loop is used; the only Python loop
    is over the network's variables, in topological order, which is
    iterative (`DAG.topological_order` uses Kahn's algorithm) and so has
    no recursion-depth limit.
    """
    rng = np.random.default_rng(seed)
    num_nodes = len(bn.names)
    cardinalities = [len(outcomes) for outcomes in bn.outcomes]

    # xbif-order codes: codes[:, j] indexes into bn.outcomes[j].
    codes = np.empty((n, num_nodes), dtype=np.int64)

    for j in bn.dag.topological_order():
        cpt = bn.cpts[j]
        row_idx = _parent_row_indices(codes, bn.parent_order[j], cardinalities)
        cum = np.cumsum(cpt, axis=1)
        cum_rows = cum[row_idx]
        # Guard against floating-point drift so the last outcome is always
        # reachable even if a row's cumulative sum is fractionally below 1.
        cum_rows[:, -1] = 1.0
        u = rng.random(n)
        codes[:, j] = np.sum(u[:, None] >= cum_rows[:, :-1], axis=1)

    # Re-encode from XBIF declaration order to the sorted-categories
    # convention used elsewhere in the package.
    values = np.empty((n, num_nodes), dtype=np.int32)
    sorted_categories: list[list[str]] = []
    for j in range(num_nodes):
        xbif_outcomes = bn.outcomes[j]
        sorted_outcomes = sorted(xbif_outcomes)
        sorted_categories.append(sorted_outcomes)
        lut = np.array([sorted_outcomes.index(label) for label in xbif_outcomes], dtype=np.int32)
        values[:, j] = lut[codes[:, j]]

    return DiscreteDataset.from_arrays(
        values=values,
        cardinalities=np.array(cardinalities, dtype=np.int32),
        names=bn.names,
        categories=sorted_categories,
    )
