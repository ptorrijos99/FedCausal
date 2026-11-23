"""The discrete scores: BDeu and BIC."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
from numba import get_num_threads

from fedcausal.scores._family import FamilyScore
from fedcausal.scores._kernels import (
    extension_buffers,
    family_extensions,
    family_score,
    value_bits,
)

if TYPE_CHECKING:
    import numpy.typing as npt

    from fedcausal.data import DiscreteDataset

__all__ = ["BDeuScore", "BICScore"]


class _DiscreteScore(FamilyScore):
    """Counts families of a discrete dataset with the compiled kernels."""

    def __init__(self, data: DiscreteDataset, expected_parents: float | None) -> None:
        empty = [name for name, r in zip(data.names, data.cardinalities, strict=True) if r < 1]
        if empty:
            raise ValueError(f"variables with no levels cannot be scored: {empty}")
        super().__init__(data.n_vars, expected_parents)
        self.data = data
        self._values = np.asfortranarray(data.values)
        self._cards = data.cardinalities.astype(np.int64)
        self._scratch = np.empty(max(4 * data.n_rows, 64), dtype=np.int64)
        # Work buffers of the batch kernel, one row per thread, made on first use.
        self._buffers: tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]] | None = None
        # Row bitsets per value, for counting small tables by popcount; skipped when large.
        bit_bytes = int(self._cards.sum()) * ((data.n_rows + 63) // 64) * 8
        if bit_bytes <= 256 * 2**20:
            self._bits, self._offsets = value_bits(self._values, self._cards)
        else:
            self._bits, self._offsets = (
                np.zeros((0, 1), np.uint64),
                np.zeros(data.n_vars + 1, np.int64),
            )

    def _count(self, node: int, parents: npt.NDArray[np.int64], ess: float) -> float:
        return float(family_score(self._values, self._cards, node, parents, ess, self._scratch))

    def _count_extensions(
        self, node: int, parents: npt.NDArray[np.int64], extra: npt.NDArray[np.int64], ess: float
    ) -> npt.NDArray[np.float64]:
        threads = get_num_threads()
        if self._buffers is None or len(self._buffers[0]) < threads:
            self._buffers = extension_buffers(
                self._values.shape[0], self._cards, len(self._scratch), threads
            )
        ints, floats = self._buffers
        out: npt.NDArray[np.float64] = family_extensions(
            self._values,
            self._cards,
            node,
            parents,
            extra,
            ess,
            len(self._scratch),
            self._bits,
            self._offsets,
            ints[:threads],
            floats[:threads],
        )
        return out


class BDeuScore(_DiscreteScore):
    """BDeu local score over a discrete dataset, memoised per family.

    The local score of a family is its log marginal likelihood under a BDeu
    prior with equivalent sample size `ess`, plus the structure prior of
    :class:`~fedcausal.scores.FamilyScore`.

    Parameters
    ----------
    data : DiscreteDataset
        The data to score against.
    ess : float, default=1.0
        Equivalent sample size of the BDeu prior.
    expected_parents : float or None, default=1.0
        Expected number of parents per node under the structure prior. Smaller
        values penalise edges more. ``None`` gives a uniform prior over
        structures, the plain BDeu score. One expected parent recovers
        markedly better structures than the uniform prior once networks reach
        a few dozen variables, and about the same below that.

    Notes
    -----
    Not safe to share between threads: the score keeps per-thread work
    buffers for its parallel kernel and a cache of families, and two threads
    calling it at once would overwrite each other's. Use one score per thread,
    or processes.
    """

    def __init__(
        self,
        data: DiscreteDataset,
        ess: float = 1.0,
        expected_parents: float | None = 1.0,
    ) -> None:
        if not 0 < ess < math.inf:
            raise ValueError(f"ess must be positive and finite, got {ess}")
        super().__init__(data, expected_parents)
        self.ess = float(ess)

    def _family(self, node: int, parents: npt.NDArray[np.int64]) -> float:
        return self._count(node, parents, self.ess)

    def _extensions(
        self, node: int, parents: npt.NDArray[np.int64], extra: npt.NDArray[np.int64]
    ) -> npt.NDArray[np.float64]:
        return self._count_extensions(node, parents, extra, self.ess)


class BICScore(_DiscreteScore):
    """BIC local score over a discrete dataset, memoised per family.

    The maximised log likelihood of a family less ``penalty_discount * log(n) / 2``
    per free parameter, ``q * (r - 1)`` for a child of ``r`` levels whose
    parents take ``q`` joint configurations, plus the structure prior of
    :class:`~fedcausal.scores.FamilyScore`.

    Parameters
    ----------
    data : DiscreteDataset
        The data to score against.
    penalty_discount : float, default=1.0
        Multiplies the penalty; larger values give sparser structures.
    expected_parents : float or None, default=None
        Expected number of parents per node under the structure prior.
        ``None``, the default, is the plain BIC.

    Notes
    -----
    Not safe to share between threads, for the reasons given in
    :class:`BDeuScore`.
    """

    def __init__(
        self,
        data: DiscreteDataset,
        penalty_discount: float = 1.0,
        expected_parents: float | None = None,
    ) -> None:
        if not 0 < penalty_discount < math.inf:
            raise ValueError(
                f"penalty_discount must be positive and finite, got {penalty_discount}"
            )
        super().__init__(data, expected_parents)
        self.penalty_discount = float(penalty_discount)
        self._per_parameter = penalty_discount * math.log(max(data.n_rows, 1)) / 2

    def _family(self, node: int, parents: npt.NDArray[np.int64]) -> float:
        q = float(np.prod(self._cards[parents].astype(np.float64)))
        free = q * (self._cards[node] - 1)
        return self._count(node, parents, 0.0) - self._per_parameter * float(free)

    def _extensions(
        self, node: int, parents: npt.NDArray[np.int64], extra: npt.NDArray[np.int64]
    ) -> npt.NDArray[np.float64]:
        q = float(np.prod(self._cards[parents].astype(np.float64)))
        free = q * self._cards[extra].astype(np.float64) * float(self._cards[node] - 1)
        out: npt.NDArray[np.float64] = (
            self._count_extensions(node, parents, extra, 0.0) - self._per_parameter * free
        )
        return out
