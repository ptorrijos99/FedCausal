"""What one FedGES site does with the data it holds."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from fedcausal.fedges.model import BNModel
from fedcausal.fusion import ConsensusUnion
from fedcausal.graph.cpdag import dag_to_cpdag
from fedcausal.scores import make_score
from fedcausal.search import Knowledge, ges, xges

if TYPE_CHECKING:
    from fedcausal.data import ContinuousDataset, DiscreteDataset
    from fedcausal.fusion import FusionOperator
    from fedcausal.graph import DAG
    from fedcausal.scores import FamilyScore, ScoreName

__all__ = ["Refinement", "Search", "Site"]

Refinement = Literal["forward", "backward", "both"]
"""How a site refines the fused model; see :class:`Site`."""

Search = Literal["ges", "xges0", "xges"]
"""The search a site learns with; see :class:`Site`."""

_REFINEMENTS = ("forward", "backward", "both")
_SEARCHES = ("ges", "xges0", "xges")


def _new_adjacencies(have: DAG, base: DAG) -> frozenset[tuple[int, int]]:
    """Return, in both orientations, the adjacencies of `have` that `base` lacks."""
    base_skeleton = {frozenset(e) for e in base.edges()}
    pairs: set[tuple[int, int]] = set()
    for u, v in have.edges():
        if frozenset((u, v)) not in base_skeleton:
            pairs.update(((u, v), (v, u)))
    return frozenset(pairs)


class Site:
    """A FedGES participant: it learns from its own data and merges in the global model.

    Parameters
    ----------
    data : DiscreteDataset or ContinuousDataset
        The site's data. It never leaves the site.
    criterion : {"bdeu", "bic", "gaussian_bic"}, default="bdeu"
        The scoring criterion; see :func:`~fedcausal.scores.make_score`.
    ess : float, default=1.0
        Equivalent sample size of the BDeu score.
    expected_parents : float or None, default=1.0
        Structure prior; see :class:`~fedcausal.scores.FamilyScore`.
    penalty_discount : float, default=1.0
        Multiplier of the BIC penalty.
    max_parents : int or None, default=None
        Most parents an Insert may give a node.
    max_t_size : int or None, default=None
        Largest conditioning subset tried by an Insert.
    local_fusion : FusionOperator, optional
        Fuses the site's model with the global one. The consensus union by
        default.
    refinement : {"forward", "backward", "both"} or None, default=None
        What to do after fusing. ``None`` keeps the fused model. The others
        start again from the site's own model and let GES's forward phase add
        the edges the fused model brought in, the backward phase remove the
        ones it dropped, or both, wherever that improves the local score.
    knowledge : Knowledge, optional
        Adjacencies every model must have or must not have. The search keeps
        to them, and they are imposed again after each fusion.
    search : {"ges", "xges0", "xges"}, default="ges"
        What :meth:`learn` runs: GES, XGES-0, or XGES with its forced
        deletions; see :class:`~fedcausal.XGES`. Refinement always uses GES's
        phases, restricted to the edges the fusion changed.

    Attributes
    ----------
    model : BNModel or None
        The site's current structure and its local score.
    """

    def __init__(
        self,
        data: DiscreteDataset | ContinuousDataset,
        *,
        criterion: ScoreName = "bdeu",
        ess: float = 1.0,
        expected_parents: float | None = 1.0,
        penalty_discount: float = 1.0,
        max_parents: int | None = None,
        max_t_size: int | None = None,
        local_fusion: FusionOperator | None = None,
        refinement: Refinement | None = None,
        knowledge: Knowledge | None = None,
        search: Search = "ges",
    ) -> None:
        if refinement is not None and refinement not in _REFINEMENTS:
            raise ValueError(
                f"refinement must be one of {_REFINEMENTS} or None, got {refinement!r}"
            )
        if search not in _SEARCHES:
            raise ValueError(f"search must be one of {_SEARCHES}, got {search!r}")
        self.data = data
        self.criterion = criterion
        self.ess = ess
        self.expected_parents = expected_parents
        self.penalty_discount = penalty_discount
        self.max_parents = max_parents
        self.max_t_size = max_t_size
        self.local_fusion = local_fusion if local_fusion is not None else ConsensusUnion()
        self.refinement = refinement
        self.knowledge = knowledge if knowledge is not None else Knowledge()
        self.search = search
        self.model: BNModel | None = None
        self._score: FamilyScore | None = None

    @property
    def score(self) -> FamilyScore:
        """FamilyScore: The site's score, kept across rounds for its cache."""
        if self._score is None:
            self._score = make_score(
                self.criterion,
                self.data,
                ess=self.ess,
                expected_parents=self.expected_parents,
                penalty_discount=self.penalty_discount,
            )
        return self._score

    def _model(self, dag: DAG) -> BNModel:
        return BNModel(dag, self.score.score_dag(dag))

    def learn(self) -> BNModel:
        """Run the search from the current model (or the empty graph) and keep the result."""
        start = self.model.dag if self.model is not None else None
        if self.search == "ges":
            cpdag = ges(
                self.score,
                self.data.n_vars,
                start,
                max_t_size=self.max_t_size,
                knowledge=self.knowledge,
                max_parents=self.max_parents,
            )
        else:
            cpdag = xges(
                self.score,
                self.data.n_vars,
                start,
                extended=self.search == "xges",
                max_t_size=self.max_t_size,
                knowledge=self.knowledge,
                max_parents=self.max_parents,
            )
        self.model = self._model(cpdag.to_dag())
        return self.model

    def merge(self, global_dag: DAG) -> BNModel:
        """Fuse `global_dag` into the current model, refine, and keep the result.

        Raises
        ------
        RuntimeError
            If the site has not learned a model yet.
        """
        if self.model is None:
            raise RuntimeError("merge() needs a model; call learn() first")
        own = self.model.dag
        fused = self.knowledge.impose(self.local_fusion.fuse([own, global_dag]))
        if self.refinement is None:
            self.model = self._model(fused)
            return self.model

        cpdag = dag_to_cpdag(own)
        n = self.data.n_vars
        if self.refinement in ("forward", "both"):
            cpdag = ges(
                self.score,
                n,
                cpdag,
                candidates=_new_adjacencies(fused, own),
                max_t_size=self.max_t_size,
                knowledge=self.knowledge,
                max_parents=self.max_parents,
                backward=False,
            )
        if self.refinement in ("backward", "both"):
            cpdag = ges(
                self.score,
                n,
                cpdag,
                candidates=_new_adjacencies(own, fused),
                knowledge=self.knowledge,
                forward=False,
            )
        self.model = self._model(cpdag.to_dag())
        return self.model
