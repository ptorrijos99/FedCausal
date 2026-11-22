"""The FedGES round loop, independent of how the sites are reached.

Every round each site rebuilds its model, the server asks the stopping rule,
fuses the sites' models and sends the result back for each site to merge in.
The server holds no data: scores come from the sites. The in-process estimator
and the Flower transport both run this loop.
"""

from __future__ import annotations

import contextlib
import multiprocessing as mp
import os
import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from fedcausal.exceptions import DenseFusionWarning
from fedcausal.fedges.model import BNModel

if TYPE_CHECKING:
    from collections.abc import Sequence
    from multiprocessing.connection import Connection

    from fedcausal.fedges.convergence import Convergence
    from fedcausal.fedges.site import Site
    from fedcausal.fusion import FusionOperator
    from fedcausal.graph import DAG

__all__ = ["SitePool", "run_rounds"]

RoundCallback = Callable[[str, int, int, BNModel], None]
"""Called as ``(stage, round, site, model)``, `stage` being ``"local"`` or ``"global"``.

The global model has site ``-1`` and score 0, since the server cannot score it.
"""


def run_rounds(
    learn: Callable[[], Sequence[BNModel]],
    merge: Callable[[DAG], object],
    fusion: FusionOperator,
    convergence: Convergence | None,
    n_rounds: int,
    on_model: RoundCallback | None = None,
) -> tuple[DAG | None, int]:
    """Run up to `n_rounds` rounds and return the last global DAG and the rounds run.

    Parameters
    ----------
    learn : callable
        Makes every site rebuild its model; returns them in site order.
    merge : callable
        Sends a global DAG for every site to merge in.
    fusion : FusionOperator
        Combines the sites' DAGs into the global one.
    convergence : Convergence or None
        Asked after every build; ``None`` runs every round.
    n_rounds : int
        Most rounds to run.
    on_model : callable, optional
        Receives every local and global model as the loop runs.

    Returns
    -------
    global_dag : DAG or None
        ``None`` if the loop stopped before its first fusion.
    rounds : int
        Rounds in which the sites built a model.
    """
    global_dag: DAG | None = None
    for round_ in range(1, n_rounds + 1):
        local = learn()
        if on_model is not None:
            for site, model in enumerate(local):
                on_model("local", round_, site, model)
        if convergence is not None and convergence.check_convergence(local):
            return global_dag, round_
        global_dag = fusion.fuse([m.dag for m in local])
        _check_density(global_dag, local, round_)
        if on_model is not None:
            on_model("global", round_, -1, BNModel(global_dag))
        merge(global_dag)
    return global_dag, n_rounds


_DENSE_RATIO = 2.0
"""A global model with this many times the sites' mean edge count triggers a warning."""


def _check_density(global_dag: DAG, local: Sequence[BNModel], round_: int) -> None:
    mean_local = sum(m.dag.num_edges for m in local) / len(local)
    if mean_local > 0 and global_dag.num_edges > _DENSE_RATIO * mean_local:
        warnings.warn(
            f"round {round_}: the fused global model has {global_dag.num_edges} edges against "
            f"{mean_local:.0f} per site on average. The sites disagree on their v-structures, "
            "so the fusion's shared order adds edges no site has. A stronger structure prior "
            "(a smaller expected_parents) or fewer, larger sites usually removes it.",
            DenseFusionWarning,
            stacklevel=3,
        )


def _serve(conn: Connection, n_threads: int) -> None:
    """Worker loop: receive this worker's sites, then run each command on them."""
    import numba

    numba.set_num_threads(n_threads)
    sites: list[Site] = conn.recv()
    while True:
        command, argument = conn.recv()
        if command == "stop":
            return
        try:
            if command == "learn":
                result: Any = [site.learn() for site in sites]
            elif command == "merge":
                result = [site.merge(argument) for site in sites]
            else:  # "models"
                result = [site.model for site in sites]
        except Exception as exc:  # noqa: BLE001 - re-raised in the parent
            conn.send(("error", exc))
        else:
            conn.send(("ok", result))


class SitePool:
    """Runs a list of sites, either in this process or spread over worker processes.

    Each worker owns its sites for the whole run, so their score caches survive
    between rounds; only DAGs and scores cross between processes. Use it as a
    context manager so the workers are shut down.

    Parameters
    ----------
    sites : list of Site
    n_jobs : int or None
        Worker processes. ``None`` or 1 runs the sites in this process, and -1
        uses one worker per site up to the number of CPUs.
    """

    def __init__(self, sites: list[Site], n_jobs: int | None = None) -> None:
        self.sites = sites
        n_workers = _n_workers(n_jobs, len(sites))
        self._workers: list[tuple[Any, Any, list[int]]] = []
        if n_workers > 1:
            ctx = mp.get_context("spawn")
            # Share the CPUs between workers rather than oversubscribe them.
            n_threads = max(1, (os.cpu_count() or 1) // n_workers)
            # Start every worker before sending any data, so they import in
            # parallel instead of one after another.
            for w in range(n_workers):
                parent, child = ctx.Pipe()
                process = ctx.Process(target=_serve, args=(child, n_threads), daemon=True)
                process.start()
                child.close()
                self._workers.append((process, parent, list(range(w, len(sites), n_workers))))
            for _, conn, owned in self._workers:
                conn.send([sites[i] for i in owned])

    def _broadcast(self, command: str, argument: object = None) -> list[Any]:
        for _, conn, _ in self._workers:
            conn.send((command, argument))
        results: list[Any] = [None] * len(self.sites)
        error: BaseException | None = None
        for _, conn, owned in self._workers:
            status, payload = conn.recv()
            if status == "error":
                error = payload
                continue
            for i, result in zip(owned, payload, strict=True):
                results[i] = result
        if error is not None:
            raise error
        return results

    def learn(self) -> list[BNModel]:
        """Make every site rebuild its model; return the models in site order."""
        if not self._workers:
            return [site.learn() for site in self.sites]
        return self._broadcast("learn")

    def merge(self, global_dag: DAG) -> list[BNModel]:
        """Make every site merge `global_dag` in; return the models in site order."""
        if not self._workers:
            return [site.merge(global_dag) for site in self.sites]
        return self._broadcast("merge", global_dag)

    def models(self) -> list[BNModel | None]:
        """Return every site's current model, in site order."""
        if not self._workers:
            return [site.model for site in self.sites]
        return self._broadcast("models")

    def close(self) -> None:
        """Shut the workers down, ending any that do not stop within a few seconds."""
        for process, conn, _ in self._workers:
            with contextlib.suppress(OSError):  # The worker may already be gone.
                conn.send(("stop", None))
            conn.close()
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
                process.join()
        self._workers = []

    def __enter__(self) -> SitePool:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _n_workers(n_jobs: int | None, n_sites: int) -> int:
    if n_jobs is None or n_jobs == 1:
        return 1
    cpus = os.cpu_count() or 1
    wanted = cpus if n_jobs < 0 else n_jobs
    return max(1, min(wanted, n_sites))
