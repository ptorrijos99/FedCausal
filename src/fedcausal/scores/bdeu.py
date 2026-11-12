"""The discrete scores: BDeu and BIC."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from fedcausal.scores._family import FamilyScore
from fedcausal.scores._kernels import family_extensions, family_score

if TYPE_CHECKING:
    import numpy.typing as npt

    from fedcausal.data import DiscreteDataset

__all__ = ["BDeuScore", "BICScore"]


class _DiscreteScore(FamilyScore):
    """Counts families of a discrete dataset with the compiled kernels."""

    def __init__(self, data: DiscreteDataset, expected_parents: float | None) -> None:
        super().__init__(data.n_vars, expected_parents)
        self.data = data
        self._values = np.asfortranarray(data.values)
        self._cards = data.cardinalities.astype(np.int64)
        self._scratch = np.empty(max(4 * data.n_rows, 64), dtype=np.int64)

    def _count(self, node: int, parents: npt.NDArray[np.int64], ess: float) -> float:
        return float(family_score(self._values, self._cards, node, parents, ess, self._scratch))

    def _count_extensions(
        self, node: int, parents: npt.NDArray[np.int64], extra: npt.NDArray[np.int64], ess: float
    ) -> npt.NDArray[np.float64]:
        out: npt.NDArray[np.float64] = family_extensions(
            self._values, self._cards, node, parents, extra, ess, len(self._scratch)
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
    """

    def __init__(
        self,
        data: DiscreteDataset,
        ess: float = 1.0,
        expected_parents: float | None = 1.0,
    ) -> None:
        if ess <= 0:
            raise ValueError(f"ess must be positive, got {ess}")
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
    """

    def __init__(
        self,
        data: DiscreteDataset,
        penalty_discount: float = 1.0,
        expected_parents: float | None = None,
    ) -> None:
        if penalty_discount <= 0:
            raise ValueError(f"penalty_discount must be positive, got {penalty_discount}")
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
