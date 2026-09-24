"""The local learner each FedGES site runs.

A site learns with GES, starting from its previous model. After fusing in the
global model it can re-run the forward or backward phase over just the edges
the two disagree on, which is much cheaper than a full search.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Protocol, get_args, runtime_checkable

from fedcausal.exceptions import ConfigurationError
from fedcausal.fedges.model import BNModel
from fedcausal.graph.meek import dag_to_cpdag
from fedcausal.scores import BDeuScore
from fedcausal.search import GES, HPool
from fedcausal.search.incremental import run_bes_incremental, run_fes_incremental

if TYPE_CHECKING:
    from fedcausal.data import DiscreteDataset
    from fedcausal.graph import DAG

__all__ = ["LocalLearner", "GESLearner", "Refinement"]

Refinement = Literal["None", "FES", "BES", "GES"]
"""Post-fusion refinement stage a :class:`GESLearner` can run.

``"None"`` keeps the fused model as-is. The others re-run the named phase over
the edges the fused and pre-fusion models disagree on.
"""

_REFINEMENTS: tuple[Refinement, ...] = get_args(Refinement)


@runtime_checkable
class LocalLearner(Protocol):
    """Builds and refines a client's local model."""

    @property
    def refinement_name(self) -> str:
        """Name of the refinement stage (``"None"`` disables it)."""
        ...

    def build_local_model(self, previous: BNModel | None, data: DiscreteDataset) -> BNModel:
        """Learn the local model, optionally warm-starting from `previous`."""
        ...

    def refine_local_model(self, old: BNModel, fused: BNModel, data: DiscreteDataset) -> BNModel:
        """Refine the fused model against `old` on their edge differences."""
        ...

    def score_dag(self, dag: DAG, data: DiscreteDataset) -> float:
        """Return the local score of `dag` on `data`."""
        ...


def _diff_candidates(have: DAG, base: DAG) -> frozenset[tuple[int, int]]:
    """Return both orientations of every adjacency `have` has and `base` lacks."""
    base_skel = {frozenset(e) for e in base.edges()}
    out: set[tuple[int, int]] = set()
    for u, v in have.edges():
        if frozenset((u, v)) in base_skel:
            continue
        out.add((u, v))
        out.add((v, u))
    return frozenset(out)


class GESLearner:
    """GES-based local learner with edge-difference refinement.

    Parameters
    ----------
    sample_prior, structure_prior : float, optional
        BDeu priors. Pass ``structure_prior=0.0`` to reproduce published
        FedGES numbers exactly; see :class:`~fedcausal.scores.BDeuScore`.
    refinement : {"None", "FES", "BES", "GES"}, optional
        Post-fusion refinement stage. Default ``"None"``.
    max_t_size : int or None, optional
        FES ``|T|`` cap forwarded to :class:`~fedcausal.search.GES`.
    empty_t_gate : bool, optional
        ``T = {}`` pre-gate forwarded to GES; see ``docs/algorithms.md``.
    h_pool : str, optional
        BES candidate pool forwarded to GES.

    Raises
    ------
    ConfigurationError
        If `refinement` is not one of the four recognised stages.
    """

    def __init__(
        self,
        sample_prior: float = 1.0,
        structure_prior: float = 1.0,
        refinement: Refinement = "None",
        max_t_size: int | None = 2,
        empty_t_gate: bool = False,
        h_pool: HPool = "nayx",
    ) -> None:
        if refinement not in _REFINEMENTS:
            expected = ", ".join(repr(r) for r in _REFINEMENTS)
            raise ConfigurationError(
                f"Unknown refinement: {refinement!r}; expected one of {expected}."
            )
        self.sample_prior = sample_prior
        self.structure_prior = structure_prior
        self._refinement: Refinement = refinement
        self.max_t_size = max_t_size
        self.empty_t_gate = empty_t_gate
        self.h_pool: HPool = h_pool
        self._score_cache: tuple[DiscreteDataset, BDeuScore] | None = None

    @property
    def refinement_name(self) -> str:
        """Name of the refinement stage."""
        return self._refinement

    def _score(self, data: DiscreteDataset) -> BDeuScore:
        """Return a `BDeuScore` over `data`, reusing the last one if `data` is the same object.

        Keeping it across rounds keeps its cache, where the search spends most
        of its time.
        """
        if self._score_cache is not None and self._score_cache[0] is data:
            return self._score_cache[1]
        score = BDeuScore(data, self.sample_prior, self.structure_prior)
        self._score_cache = (data, score)
        return score

    def score_dag(self, dag: DAG, data: DiscreteDataset) -> float:
        """Return the BDeu score of `dag` on `data`."""
        return self._score(data).score_dag(dag)

    def build_local_model(self, previous: BNModel | None, data: DiscreteDataset) -> BNModel:
        """Learn the local model with GES, warm-started from `previous`."""
        score = self._score(data)
        ges = GES(
            score,
            h_pool=self.h_pool,
            max_t_size=self.max_t_size,
            empty_t_gate=self.empty_t_gate,
        )
        initial = previous.dag if previous is not None else None
        cpdag = ges.fit(data, initial_graph=initial)
        dag = cpdag.to_dag()
        return BNModel(dag, score.score_dag(dag))

    def refine_local_model(self, old: BNModel, fused: BNModel, data: DiscreteDataset) -> BNModel:
        """Refine `fused` against `old` on their edge differences."""
        if self._refinement == "None":
            return fused
        score = self._score(data)
        if self._refinement == "FES":
            dag = self._refine_fes(score, old.dag, fused.dag)
        elif self._refinement == "BES":
            dag = self._refine_bes(score, old.dag, fused.dag)
        else:
            fes_dag = self._refine_fes(score, old.dag, fused.dag)
            dag = self._refine_bes(score, fes_dag, fused.dag)
        return BNModel(dag, score.score_dag(dag))

    def _refine_fes(self, score: BDeuScore, old: DAG, target: DAG) -> DAG:
        candidates = _diff_candidates(target, old)
        # The full scan visits every pair on every step even when restricted.
        pdag, _ = run_fes_incremental(
            score,
            dag_to_cpdag(old),
            candidate_edges=candidates,
            max_t_size=self.max_t_size,
            empty_t_gate=self.empty_t_gate,
        )
        return pdag.to_dag()

    def _refine_bes(self, score: BDeuScore, old: DAG, target: DAG) -> DAG:
        candidates = _diff_candidates(old, target)
        pdag, _ = run_bes_incremental(
            score,
            dag_to_cpdag(old),
            candidate_edges=candidates,
            h_pool=self.h_pool,
        )
        return pdag.to_dag()
