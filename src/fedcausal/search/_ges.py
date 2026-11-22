"""Greedy Equivalence Search (Chickering, 2002)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from fedcausal._base import Estimator, named_adjacency
from fedcausal.data import ContinuousDataset, DiscreteDataset, as_continuous, as_discrete
from fedcausal.graph import DAG, PDAG
from fedcausal.graph.cpdag import dag_to_cpdag
from fedcausal.scores import SCORE_NAMES, log_likelihood, make_score
from fedcausal.search._engine import backward_phase, forward_phase
from fedcausal.search._knowledge import Knowledge, PairLike

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy.typing as npt
    import pandas as pd
    from typing_extensions import Self

    from fedcausal.data.dataset import DiscreteLike
    from fedcausal.scores import Score, ScoreName

__all__ = ["GES", "ges"]


def ges(
    score: Score,
    n_vars: int,
    initial_graph: DAG | PDAG | None = None,
    *,
    candidates: frozenset[tuple[int, int]] | None = None,
    max_t_size: int | None = 2,
    knowledge: Knowledge | None = None,
    max_parents: int | None = None,
    forward: bool = True,
    backward: bool = True,
) -> PDAG:
    """Run GES with `score` and return the learned CPDAG.

    Parameters
    ----------
    score : Score
        Decomposable score over `n_vars` variables.
    n_vars : int
        Number of variables.
    initial_graph : DAG or PDAG, optional
        Where the search starts: the equivalence class of a DAG, or of a
        consistent extension of a PDAG. The empty graph by default.
    candidates : frozenset of (int, int), optional
        Restrict both phases to these ordered pairs ``(x, y)``.
    max_t_size : int or None, default=2
        Largest conditioning subset tried by an Insert. ``None`` tries all of
        them, which is exact but exponential in the degree; 2 has given the
        same structures on every benchmark network.
    knowledge : Knowledge, optional
        Adjacencies the result must have or must not have. The search starts
        from `initial_graph` with them imposed.
    max_parents : int or None, optional
        No Insert gives its child more parents than this. ``None`` is no limit.
    forward, backward : bool, default=True
        Which phases to run.

    Returns
    -------
    PDAG
        The CPDAG of the learned equivalence class.
    """
    if max_t_size is not None and max_t_size < 0:
        raise ValueError(f"max_t_size must be non-negative or None, got {max_t_size}")
    if max_parents is not None and max_parents < 0:
        raise ValueError(f"max_parents must be non-negative or None, got {max_parents}")
    if initial_graph is not None and initial_graph.num_nodes != n_vars:
        raise ValueError(f"initial_graph has {initial_graph.num_nodes} nodes, expected {n_vars}")
    if initial_graph is None:
        start = DAG(n_vars)
    elif isinstance(initial_graph, DAG):
        start = initial_graph
    else:
        start = initial_graph.to_dag()
    cpdag = dag_to_cpdag(knowledge.impose(start) if knowledge else start)
    if forward:
        cpdag = forward_phase(score, cpdag, candidates, max_t_size, knowledge, max_parents)
    if backward:
        cpdag = backward_phase(score, cpdag, candidates, knowledge)
    return cpdag


class GES(Estimator):
    """Learn a causal structure with Greedy Equivalence Search.

    The search adds edges while the score improves, then removes them while it
    improves, moving between Markov equivalence classes.

    Parameters
    ----------
    criterion : {"bdeu", "bic", "gaussian_bic"}, default="bdeu"
        The scoring criterion: BDeu or BIC over categorical data, or the BIC of
        a linear Gaussian model over continuous data.
    ess : float, default=1.0
        Equivalent sample size of the BDeu score. Also the prior of the
        parameters behind :meth:`score` on categorical data.
    expected_parents : float or None, default=1.0
        Expected in-degree under the structure prior, for every score; see
        :class:`~fedcausal.scores.FamilyScore`. ``None`` is a uniform prior.
    penalty_discount : float, default=1.0
        Multiplier of the BIC penalty; larger values give sparser structures.
    max_parents : int or None, default=None
        Most parents an Insert may give a node. ``None`` is no limit.
    max_t_size : int or None, default=2
        Largest conditioning subset tried by an Insert; ``None`` tries all.
    required, forbidden : sequence of pairs, optional
        Pairs of variables, by name or column index, that must be adjacent or
        must not be. The direction of a required edge is still learned.

    Attributes
    ----------
    cpdag_ : PDAG
        The learned equivalence class.
    dag_ : DAG
        One DAG of that class.
    score_ : float
        The score of `dag_`.
    adjacency_ : pandas.DataFrame
        The CPDAG with variable names: ``A[i, j] = 1`` and ``A[j, i] = 0`` is
        ``i -> j``, and both set is an undirected edge.
    n_features_in_ : int
        Number of variables.
    feature_names_in_ : ndarray of str
        Variable names.

    Examples
    --------
    >>> from fedcausal import GES
    >>> from fedcausal.datasets import load_asia
    >>> model = GES().fit(load_asia().data)
    >>> model.cpdag_
    PDAG(num_nodes=8, directed=..., undirected=...)
    """

    def __init__(
        self,
        criterion: ScoreName = "bdeu",
        ess: float = 1.0,
        expected_parents: float | None = 1.0,
        penalty_discount: float = 1.0,
        max_parents: int | None = None,
        max_t_size: int | None = 2,
        required: Sequence[PairLike] | None = None,
        forbidden: Sequence[PairLike] | None = None,
    ) -> None:
        self.criterion = criterion
        self.ess = ess
        self.expected_parents = expected_parents
        self.penalty_discount = penalty_discount
        self.max_parents = max_parents
        self.max_t_size = max_t_size
        self.required = required
        self.forbidden = forbidden

    def _encode(
        self, X: DiscreteLike | ContinuousDataset | npt.ArrayLike
    ) -> DiscreteDataset | ContinuousDataset:
        if self.criterion not in SCORE_NAMES:
            raise ValueError(f"criterion must be one of {SCORE_NAMES}, got {self.criterion!r}")
        if self.criterion == "gaussian_bic":
            return as_continuous(X)  # type: ignore[arg-type]
        return as_discrete(X)  # type: ignore[arg-type]

    def fit(
        self,
        X: DiscreteLike | ContinuousDataset | npt.ArrayLike,
        y: None = None,
        initial_graph: DAG | None = None,
    ) -> Self:
        """Learn the structure of `X`.

        Parameters
        ----------
        X : DiscreteDataset, ContinuousDataset, pandas.DataFrame or array-like
            Of shape (n_rows, n_vars). For the discrete scores every distinct
            value of a column is a level; for ``"gaussian_bic"`` the columns
            are real-valued.
        y : None
            Ignored.
        initial_graph : DAG, optional
            Where the search starts. The empty graph by default.

        Returns
        -------
        self
        """
        data = self._encode(X)
        local = make_score(
            self.criterion,
            data,
            ess=self.ess,
            expected_parents=self.expected_parents,
            penalty_discount=self.penalty_discount,
        )
        knowledge = Knowledge.from_pairs(self.required or (), self.forbidden or (), data.names)
        self.cpdag_ = ges(
            local,
            data.n_vars,
            initial_graph,
            max_t_size=self.max_t_size,
            knowledge=knowledge,
            max_parents=self.max_parents,
        )
        self.dag_ = self.cpdag_.to_dag()
        self.score_ = local.score_dag(self.dag_)
        self.n_features_in_ = data.n_vars
        self.feature_names_in_ = np.asarray(data.names, dtype=object)
        self._data = data
        return self

    def score(self, X: DiscreteLike | ContinuousDataset | npt.ArrayLike, y: None = None) -> float:
        """Return the mean log likelihood per row of `X` under the learned structure.

        The parameters are learned from the training data: by a BDeu prior of
        equivalent sample size `ess` for categorical data, by maximum
        likelihood for continuous data. Higher is better, so a scikit-learn
        parameter search can select the hyperparameters on held-out rows.

        Parameters
        ----------
        X : DiscreteDataset, ContinuousDataset, pandas.DataFrame or array-like
            Rows over the training variables, with no level the training data
            lacks.
        y : None
            Ignored.

        Returns
        -------
        float

        Raises
        ------
        ValueError
            If `X` has other variables, or a level the training data lacks.
        """
        self._check_fitted("dag_")
        if isinstance(self._data, DiscreteDataset):
            test: DiscreteDataset | ContinuousDataset = as_discrete(
                X,  # type: ignore[arg-type]
                self._data.categories,
            )
        else:
            test = as_continuous(X)  # type: ignore[arg-type]
        return log_likelihood(self.dag_, self._data, test, ess=self.ess) / max(test.n_rows, 1)

    @property
    def adjacency_(self) -> pd.DataFrame:
        """pandas.DataFrame: The learned CPDAG as a named adjacency matrix."""
        self._check_fitted("cpdag_")
        return named_adjacency(self.cpdag_, self.feature_names_in_)
