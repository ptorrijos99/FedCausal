"""Compiled counting kernels behind the discrete scores."""

from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

__all__ = ["family_extensions", "family_score"]

_MAX_CODE = 1 << 40
"""Largest mixed-radix code built before configurations are relabelled densely."""


@njit(cache=True, nogil=True)
def _relabel(codes: np.ndarray) -> int:
    """Replace `codes` by dense labels ``0..k-1`` in the same order; return ``k``."""
    order = np.argsort(codes, kind="mergesort")
    label = -1
    previous = -1
    for i in order:
        if label < 0 or codes[i] != previous:
            previous = codes[i]
            label += 1
        codes[i] = label
    return label + 1


@njit(cache=True, nogil=True)
def _extend(codes: np.ndarray, code_range: int, column: np.ndarray, radix: int) -> int:
    """Append one more variable to the configuration codes; return the new range."""
    if code_range * radix > _MAX_CODE:
        code_range = _relabel(codes)
    for i in range(codes.shape[0]):
        codes[i] = codes[i] * radix + column[i]
    return code_range * radix


@njit(cache=True, nogil=True, inline="always")
def _cell(c: int, a: float, lg_a: float, mle: bool) -> float:
    """One count's term: ``log Gamma(a + c) - log Gamma(a)``, or ``c log c`` for the MLE."""
    if mle:
        return c * math.log(c)
    return math.lgamma(a + c) - lg_a


@njit(cache=True, nogil=True)
def _score_codes(
    codes: np.ndarray,
    code_range: int,
    child: np.ndarray,
    q: float,
    r: int,
    ess: float,
    scratch: np.ndarray,
) -> float:
    """Return the log likelihood of `child` given parent configurations `codes`.

    The BDeu marginal likelihood with equivalent sample size `ess`, or the
    maximised likelihood when `ess` is zero.

    `codes` lie in ``[0, code_range)`` and are overwritten. When the table of
    ``code_range * r`` cells fits in `scratch` the rows are counted into it;
    otherwise they are sorted by cell, which visits only the configurations
    that occur. Configurations that never occur contribute exactly zero, so
    both paths agree. The pseudocounts use the true number of configurations
    `q`, which may exceed `code_range` after relabelling.
    """
    n = codes.shape[0]
    mle = ess == 0.0
    a_j = ess / q
    a_jk = ess / (q * r)
    lg_aj = 0.0 if mle else math.lgamma(a_j)
    lg_ajk = 0.0 if mle else math.lgamma(a_jk)
    total = 0.0
    if code_range * r <= scratch.shape[0]:
        cells = code_range * r
        counts = scratch[:cells]
        counts[:] = 0
        for i in range(n):
            counts[codes[i] * r + child[i]] += 1
        for j in range(code_range):
            n_j = 0
            for k in range(r):
                c = counts[j * r + k]
                if c > 0:
                    total += _cell(c, a_jk, lg_ajk, mle)
                    n_j += c
            if n_j > 0:
                total -= _cell(n_j, a_j, lg_aj, mle)
        return total

    if code_range > _MAX_CODE // r:
        _relabel(codes)
    for i in range(n):
        codes[i] = codes[i] * r + child[i]
    codes.sort()
    i = 0
    while i < n:
        config = codes[i] // r
        n_j = 0
        while i < n and codes[i] // r == config:
            cell = codes[i]
            c = 0
            while i < n and codes[i] == cell:
                c += 1
                i += 1
            total += _cell(c, a_jk, lg_ajk, mle)
            n_j += c
        total -= _cell(n_j, a_j, lg_aj, mle)
    return total


@njit(cache=True, nogil=True)
def _encode(
    values: np.ndarray, cards: np.ndarray, parents: np.ndarray
) -> tuple[np.ndarray, int, float]:
    """Return the parent configuration codes, their range, and the true count ``q``."""
    codes = np.zeros(values.shape[0], np.int64)
    code_range = 1
    q = 1.0
    for p in parents:
        code_range = _extend(codes, code_range, values[:, p], cards[p])
        q *= cards[p]
    return codes, code_range, q


@njit(cache=True, nogil=True)
def family_score(
    values: np.ndarray,
    cards: np.ndarray,
    node: int,
    parents: np.ndarray,
    ess: float,
    scratch: np.ndarray,
) -> float:
    """Return the log likelihood of `node` given `parents`: BDeu, or the MLE when `ess` is 0.

    Parameters
    ----------
    values : np.ndarray
        Integer-coded data, shape ``(n_rows, n_vars)``, column-major so that
        each column is contiguous.
    cards : np.ndarray
        Cardinality of each variable, ``int64``.
    node : int
        The child.
    parents : np.ndarray
        The parents, ``int64``.
    ess : float
        Equivalent sample size; 0 gives the maximised likelihood.
    scratch : np.ndarray
        ``int64`` work buffer for the contingency table.
    """
    codes, code_range, q = _encode(values, cards, parents)
    return _score_codes(codes, code_range, values[:, node], q, cards[node], ess, scratch)


@njit(cache=True, nogil=True, parallel=True)
def family_extensions(
    values: np.ndarray,
    cards: np.ndarray,
    node: int,
    parents: np.ndarray,
    extra: np.ndarray,
    ess: float,
    table_size: int,
) -> np.ndarray:
    """Return :func:`family_score` of `node` given ``parents + [x]``, for each `x` in `extra`.

    The configurations of `parents` are encoded once and shared, and the
    families are scored in parallel, each with its own buffers of at most
    `table_size` cells.
    """
    base, base_range, q_base = _encode(values, cards, parents)
    child = values[:, node]
    r = cards[node]
    out = np.empty(extra.shape[0])
    for k in prange(extra.shape[0]):
        x = extra[k]
        codes = base.copy()
        code_range = _extend(codes, base_range, values[:, x], cards[x])
        scratch = np.empty(min(table_size, code_range * r), np.int64)
        out[k] = _score_codes(codes, code_range, child, q_base * cards[x], r, ess, scratch)
    return out
