"""Scoring functions for candidate causal structures."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from fedcausal.scores._family import FamilyScore
from fedcausal.scores.base import Score
from fedcausal.scores.bdeu import BDeuScore, BICScore
from fedcausal.scores.gaussian import GaussianBICScore
from fedcausal.scores.likelihood import log_likelihood

if TYPE_CHECKING:
    from fedcausal.data import ContinuousDataset, DiscreteDataset

__all__ = [
    "SCORE_NAMES",
    "BDeuScore",
    "BICScore",
    "FamilyScore",
    "GaussianBICScore",
    "Score",
    "ScoreName",
    "log_likelihood",
    "make_score",
]

ScoreName = Literal["bdeu", "bic", "gaussian_bic"]
"""The scores the estimators take by name: two discrete, one continuous."""

SCORE_NAMES: tuple[ScoreName, ...] = ("bdeu", "bic", "gaussian_bic")


def make_score(
    name: ScoreName,
    data: DiscreteDataset | ContinuousDataset,
    *,
    ess: float = 1.0,
    expected_parents: float | None = 1.0,
    penalty_discount: float = 1.0,
) -> FamilyScore:
    """Return the score called `name` over `data`.

    Parameters
    ----------
    name : {"bdeu", "bic", "gaussian_bic"}
        :class:`BDeuScore` and :class:`BICScore` take a ``DiscreteDataset``,
        :class:`GaussianBICScore` a ``ContinuousDataset``.
    data : DiscreteDataset or ContinuousDataset
        The data to score against.
    ess : float, default=1.0
        Equivalent sample size, for BDeu only.
    expected_parents : float or None, default=1.0
        Structure prior, for every score; see :class:`FamilyScore`.
    penalty_discount : float, default=1.0
        Multiplier of the BIC penalty, for the BIC scores only.

    Raises
    ------
    ValueError
        If `name` is not one of the three.
    """
    if name == "bdeu":
        return BDeuScore(data, ess, expected_parents)  # type: ignore[arg-type]
    if name == "bic":
        return BICScore(data, penalty_discount, expected_parents)  # type: ignore[arg-type]
    if name == "gaussian_bic":
        return GaussianBICScore(data, penalty_discount, expected_parents)  # type: ignore[arg-type]
    raise ValueError(f"score must be one of {SCORE_NAMES}, got {name!r}")
