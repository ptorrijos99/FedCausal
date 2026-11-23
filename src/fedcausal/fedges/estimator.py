"""The FedGES estimator."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import numpy as np

from fedcausal._base import Estimator, check_sites, named_adjacency
from fedcausal.data import as_continuous_sites, as_discrete_sites
from fedcausal.exceptions import ConvergedBeforeFusionError
from fedcausal.fedges.convergence import default_convergence
from fedcausal.fedges.rounds import SitePool, run_rounds
from fedcausal.fedges.site import Site
from fedcausal.fusion import Consensus
from fedcausal.graph.cpdag import dag_to_cpdag
from fedcausal.scores import SCORE_NAMES
from fedcausal.search import Knowledge

if TYPE_CHECKING:
    from collections.abc import Sequence

    import pandas as pd
    from typing_extensions import Self

    from fedcausal.data.dataset import DiscreteLike
    from fedcausal.fedges.convergence import Convergence
    from fedcausal.fedges.model import BNModel
    from fedcausal.fedges.site import Refinement
    from fedcausal.fusion import FusionOperator
    from fedcausal.scores import ScoreName
    from fedcausal.search._knowledge import PairLike

__all__ = ["FedGES"]


class FedGES(Estimator):
    """Learn one causal structure from data held by several sites.

    Each round every site runs GES on its own data, starting from its current
    model; a server fuses the sites' structures into a global one and sends it
    back; each site fuses that into its own model. Only structures and scores
    leave the sites.

    Torrijos, Gámez & Puerta, "FedGES: A Federated Learning Approach for
    Bayesian Network Structure Learning", *Machine Learning* 115, 9 (2026).

    Parameters
    ----------
    n_rounds : int, default=10
        Most rounds to run.
    criterion : {"bdeu", "bic", "gaussian_bic"}, default="bdeu"
        Each site's scoring criterion: BDeu or BIC over categorical data, or
        the BIC of a linear Gaussian model over continuous data.
    ess : float, default=1.0
        Equivalent sample size of the BDeu score.
    expected_parents : float or None, default=1.0
        Structure prior of each site's score; see
        :class:`~fedcausal.scores.FamilyScore`.
    penalty_discount : float, default=1.0
        Multiplier of the BIC penalty.
    max_parents : int or None, default=None
        Most parents an Insert may give a node at a site.
    max_t_size : int or None, default=2
        Largest conditioning subset tried by an Insert.
    global_fusion : FusionOperator, optional
        How the server combines the sites' structures. Default
        ``Consensus(50)``: the edges at least half the sites agree on.
    local_fusion : FusionOperator, optional
        How a site combines the global structure with its own. Default
        ``ConsensusUnion()``.
    refinement : {"forward", "backward", "both"} or None, default=None
        What a site does after fusing; see :class:`~fedcausal.fedges.Site`.
    convergence : Convergence, optional
        When to stop before `n_rounds`. By default, once the mean local score
        stops improving or every site repeats an earlier model.
    required, forbidden : sequence of pairs, optional
        Pairs of variables, by name or column index, that must be adjacent or
        must not be, in every site's model and in the global one. The
        direction of a required edge is still learned.
    n_jobs : int or None, default=None
        Processes to spread the sites over. ``None`` or 1 runs them one after
        another in this process; -1, or any negative value, uses one process
        per site up to the number of CPUs.

    Attributes
    ----------
    dag_ : DAG
        The global structure after the last round.
    cpdag_ : PDAG
        Its equivalence class.
    adjacency_ : pandas.DataFrame
        `dag_` as a named adjacency matrix, ``A[i, j] = 1`` for ``i -> j``.
    site_models_ : list of BNModel
        Each site's own model when the rounds stopped.
    n_rounds_ : int
        Rounds run.
    history_ : pandas.DataFrame
        One row per model built: ``round``, ``site`` (-1 for the global
        model), ``edges`` and ``score`` (the site's own score; NaN for the
        global model).
    n_features_in_ : int
    feature_names_in_ : ndarray of str

    Examples
    --------
    >>> from fedcausal import FedGES
    >>> from fedcausal.data import contiguous_split
    >>> from fedcausal.datasets import load_alarm
    >>> sites = contiguous_split(load_alarm().data, 4)
    >>> model = FedGES(n_rounds=5).fit(sites)
    >>> model.dag_
    DAG(num_nodes=37, num_edges=...)
    """

    def __init__(
        self,
        n_rounds: int = 10,
        criterion: ScoreName = "bdeu",
        ess: float = 1.0,
        expected_parents: float | None = 1.0,
        penalty_discount: float = 1.0,
        max_parents: int | None = None,
        max_t_size: int | None = 2,
        global_fusion: FusionOperator | None = None,
        local_fusion: FusionOperator | None = None,
        refinement: Refinement | None = None,
        convergence: Convergence | None = None,
        required: Sequence[PairLike] | None = None,
        forbidden: Sequence[PairLike] | None = None,
        n_jobs: int | None = None,
    ) -> None:
        self.n_rounds = n_rounds
        self.criterion = criterion
        self.ess = ess
        self.expected_parents = expected_parents
        self.penalty_discount = penalty_discount
        self.max_parents = max_parents
        self.max_t_size = max_t_size
        self.global_fusion = global_fusion
        self.local_fusion = local_fusion
        self.refinement = refinement
        self.convergence = convergence
        self.required = required
        self.forbidden = forbidden
        self.n_jobs = n_jobs

    def fit(self, X: Sequence[DiscreteLike], y: None = None) -> Self:
        """Learn the structure from the sites' data.

        Parameters
        ----------
        X : sequence of DiscreteDataset, ContinuousDataset, pandas.DataFrame or array-like
            One table per site, all over the same variables. For the discrete
            criteria, levels are pooled across sites, so a level seen at one
            site exists at all of them.
        y : None
            Ignored.

        Returns
        -------
        self

        Raises
        ------
        TypeError
            If `X` is a single table rather than one per site.
        ValueError
            If the sites hold no rows between them, disagree on their
            variables, or a hyperparameter is out of range.
        ConvergedBeforeFusionError
            If the rounds stopped before the first fusion.
        """
        if self.n_rounds < 1:
            raise ValueError(f"n_rounds must be at least 1, got {self.n_rounds}")
        if self.n_jobs == 0:
            raise ValueError("n_jobs must be a positive or negative integer, or None; got 0")
        if self.criterion not in SCORE_NAMES:
            raise ValueError(f"criterion must be one of {SCORE_NAMES}, got {self.criterion!r}")
        check_sites(X)
        datasets = (
            as_continuous_sites(X)  # type: ignore[arg-type]
            if self.criterion == "gaussian_bic"
            else as_discrete_sites(X)
        )
        if sum(data.n_rows for data in datasets) == 0:
            raise ValueError("the sites hold no rows")
        knowledge = Knowledge.from_pairs(
            self.required or (), self.forbidden or (), datasets[0].names
        )
        sites = [
            Site(
                data,
                criterion=self.criterion,
                ess=self.ess,
                expected_parents=self.expected_parents,
                penalty_discount=self.penalty_discount,
                max_parents=self.max_parents,
                max_t_size=self.max_t_size,
                local_fusion=self.local_fusion,
                refinement=self.refinement,
                knowledge=knowledge,
            )
            for data in datasets
        ]
        rows: list[tuple[int, int, int, float]] = []

        def record(stage: str, round_: int, site: int, model: BNModel) -> None:
            score = model.score if stage == "local" else float("nan")
            rows.append((round_, site, model.dag.num_edges, score))

        # Criteria keep state between rounds, so each fit gets its own copy.
        convergence = (
            copy.deepcopy(self.convergence)
            if self.convergence is not None
            else default_convergence()
        )
        fusion = self.global_fusion if self.global_fusion is not None else Consensus(50)
        with SitePool(sites, self.n_jobs) as pool:
            global_dag, rounds = run_rounds(
                pool.learn, pool.merge, fusion, convergence, self.n_rounds, record
            )
            site_models = pool.models()
        if global_dag is None:
            raise ConvergedBeforeFusionError("the rounds stopped before the first fusion")

        # The server's fusion knows nothing of the constraints, so they are
        # imposed on its output as each site imposes them on its own.
        self.dag_ = knowledge.impose(global_dag)
        self.cpdag_ = dag_to_cpdag(self.dag_)
        self.site_models_ = site_models
        self.n_rounds_ = rounds
        import pandas as pd

        self.history_ = pd.DataFrame(rows, columns=["round", "site", "edges", "score"])
        self.n_features_in_ = datasets[0].n_vars
        self.feature_names_in_ = np.asarray(datasets[0].names, dtype=object)
        return self

    @property
    def adjacency_(self) -> pd.DataFrame:
        """pandas.DataFrame: The global DAG as a named adjacency matrix."""
        self._check_fitted("dag_")
        return named_adjacency(self.dag_, self.feature_names_in_)
