"""The FedRCD family over Flower.

The one-shot solvers need one round trip: every site replies with its moment
sums, row count and a fingerprint of its variable names, and the server checks
the fingerprints, aggregates the sums into cumulants and runs the solver.

``fedrcd_exact`` takes ``p - 1`` rounds. Each identifies a source, and every
site deflates its own data by it and replies with fresh sums; the deflated
data stays in the site's ``context.state`` between rounds.

:func:`run_server_loop` and :func:`run_exact_server_loop` take any ``Grid``;
:func:`run_flower_rcd` and :func:`run_flower_rcd_exact` split a dataset and
run it on Flower's simulation engine.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np
from flwr.client import ClientApp
from flwr.common import ArrayRecord, Context, Message, RecordDict
from flwr.server import Grid, ServerApp

from fedcausal.data import ContinuousDataset
from fedcausal.exceptions import ClientFailureError, ConfigurationError
from fedcausal.flower._grid import (
    CTRL_KEY,
    await_nodes,
    broadcast,
    check_schemas,
    ctrl_int,
    required_clients,
    simulation_runner,
    usable_replies,
)
from fedcausal.flower.serde import (
    control_config,
    names_fingerprint,
    record_to_stats,
    stats_to_record,
)
from fedcausal.lingam.cumulants import aggregate_and_compute_cumulants, compute_local_raw_stats
from fedcausal.lingam.ishc import fedhc, fedishc
from fedcausal.lingam.rcd import RCDResult, fedrcd, fedrcd_noniterative, ishc_beta
from fedcausal.lingam.solvers import estimate_b_from_cov, estimate_b_lasso_from_cov
from fedcausal.lingam.tau import tau_matrix_4th

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from fedcausal.flower.loaders import PartitionLoader
    from fedcausal.lingam.cumulants import Cumulants

__all__ = [
    "build_client_app",
    "build_exact_client_app",
    "build_exact_server_app",
    "build_server_app",
    "handle_deflate_message",
    "handle_stats_message",
    "run_exact_server_loop",
    "run_flower_rcd",
    "run_flower_rcd_exact",
    "run_server_loop",
]

_STATS_KEY = "stats"
_BETAS_KEY = "betas"
_WORKING_STATE_KEY = "rcd_working"

_SOLVERS: dict[str, Callable[..., RCDResult]] = {
    "fedrcd": fedrcd,
    "fedrcd_noniterative": fedrcd_noniterative,
    "ishc_beta": ishc_beta,
    "fedishc": fedishc,
    "fedhc": fedhc,
}


def handle_stats_message(
    msg: Message, context: Context, loader: PartitionLoader[ContinuousDataset]
) -> Message:
    """Reply to a ``train`` message with this client's moment sums.

    Parameters
    ----------
    msg : Message
        The incoming message. Carries no model.
    context : Context
        This supernode's context.
    loader : PartitionLoader
        Resolves this supernode's dataset.

    Returns
    -------
    Message
        A reply carrying the nine moment-sum arrays, the local row count, and
        the client's variable-name fingerprint.
    """
    data = loader(context)
    stats = compute_local_raw_stats(data.values, order=4)
    reply = RecordDict(
        {
            _STATS_KEY: stats_to_record(stats),
            CTRL_KEY: control_config(
                n=int(stats["n"]),
                n_vars=data.n_vars,
                schema=names_fingerprint(data.names),
            ),
        }
    )
    return Message(reply, reply_to=msg)


def build_client_app(loader: PartitionLoader[ContinuousDataset]) -> ClientApp:
    """Register :func:`handle_stats_message` as a ``ClientApp``."""
    app = ClientApp()

    @app.train()
    def train(msg: Message, context: Context) -> Message:
        return handle_stats_message(msg, context, loader)

    return app


def run_server_loop(
    grid: Grid,
    n_clients: int,
    solver: str = "fedrcd",
    *,
    min_clients: int | None = None,
    **solver_kwargs: Any,
) -> RCDResult:
    """Run the one-shot FedRCD protocol over `grid` and return the result.

    Parameters
    ----------
    grid : Grid
        Anything that can send and receive Flower messages.
    n_clients : int
        Number of supernodes to wait for before broadcasting.
    solver : {"fedrcd", "fedrcd_noniterative", "ishc_beta", "fedishc", "fedhc"}, optional
        Which one-shot solver to run on the aggregated statistics.
    min_clients : int or None, optional
        How many usable replies the round needs. Default `n_clients`, so a
        failed client stops the run; lower it to tolerate dropouts.
    **solver_kwargs : Any
        Passed through to the selected solver (``use_lasso``, ``lmbda_type``).

    Returns
    -------
    RCDResult
        The causal order and coefficient matrix the solver estimated.

    Raises
    ------
    ConfigurationError
        If `solver` is not one of the supported names, or `min_clients` is not
        between 1 and `n_clients`.
    ClientFailureError
        If too few supernodes register, or too few reply.
    SchemaMismatchError
        If the clients do not agree on their variables.
    """
    solver_fn = _resolve_solver(solver)
    required = required_clients(n_clients, min_clients)

    node_ids = await_nodes(grid, n_clients)
    replies = grid.send_and_receive(
        broadcast(node_ids, RecordDict({CTRL_KEY: control_config(phase="stats")}))
    )
    good = usable_replies(replies, required, "stats")
    check_schemas(good)

    return solver_fn(_aggregate_replies(good), **solver_kwargs)


def build_server_app(
    n_clients: int,
    solver: str = "fedrcd",
    *,
    min_clients: int | None = None,
    results: list[RCDResult] | None = None,
    **solver_kwargs: Any,
) -> ServerApp:
    """Register :func:`run_server_loop` as a ``ServerApp``.

    Takes the same arguments. `results`, if given, receives the final
    `RCDResult`, since a ``ServerApp`` cannot return one.
    """
    app = ServerApp()

    @app.main()
    def main(grid: Grid, context: Context) -> None:
        result = run_server_loop(grid, n_clients, solver, min_clients=min_clients, **solver_kwargs)
        if results is not None:
            results.append(result)

    return app


def run_flower_rcd(
    data: ContinuousDataset,
    n_clients: int,
    solver: str = "fedrcd",
    **solver_kwargs: Any,
) -> RCDResult:
    """Run a one-shot FedRCD solver over a Flower simulation and return the result.

    Splits `data` into `n_clients` contiguous blocks, one per simulated
    supernode. Needs ``fedcausal[flower-simulation]``.

    Parameters
    ----------
    data : ContinuousDataset
        Full dataset to partition across clients.
    n_clients : int
        Number of clients and supernodes.
    solver : {"fedrcd", "fedrcd_noniterative", "ishc_beta", "fedishc", "fedhc"}, optional
        Which one-shot solver to run on the aggregated statistics.
    **solver_kwargs : Any
        Passed through to the selected solver (``use_lasso``, ``lmbda_type``).

    Returns
    -------
    RCDResult
        The causal order and coefficient matrix the solver estimated.

    Raises
    ------
    ConfigurationError
        If Flower's simulation extra is not installed, or `solver` is not one of
        the supported names.
    ClientFailureError
        If too few supernodes register, or too few reply.
    SchemaMismatchError
        If the clients do not agree on their variables.
    """
    results: list[RCDResult] = []
    _simulate(
        build_client_app,
        build_server_app(n_clients, solver, results=results, **solver_kwargs),
        data,
        n_clients,
    )
    if not results:
        raise ClientFailureError("the Flower simulation completed without producing a result")
    return results[0]


def handle_deflate_message(
    msg: Message, context: Context, loader: PartitionLoader[ContinuousDataset]
) -> Message:
    """Handle one round of FedRCD-Exact and reply with fresh moment sums.

    Round 0 starts from the site's data. Every later round deflates the stored
    working matrix by the source and coefficients the server sent
    (``x_j -= beta_j * x_s``) and drops the source column.

    Parameters
    ----------
    msg : Message
        The incoming message. Its control record carries ``round``, and from
        round 1 onward also ``src_local`` and a ``betas`` ``ArrayRecord``.
    context : Context
        This supernode's context. The progressively deflated working matrix
        persists in ``context.state`` between rounds.
    loader : PartitionLoader
        Resolves this supernode's dataset. Only consulted on round 0.

    Returns
    -------
    Message
        The moment sums of the working matrix and the row count. Round 0 also
        carries the fingerprint of the variable names.
    """
    control = msg.content[CTRL_KEY]
    round_idx = int(cast(int, control["round"]))

    if round_idx == 0:
        data = loader(context)
        x_curr = np.ascontiguousarray(data.values, dtype=np.float64)
        schema = names_fingerprint(data.names)
    else:
        src_local = int(cast(int, control["src_local"]))
        betas = cast(ArrayRecord, msg.content[_BETAS_KEY]).to_numpy_ndarrays()[0]
        x_prev = cast(ArrayRecord, context.state[_WORKING_STATE_KEY]).to_numpy_ndarrays()[0]
        x_s = x_prev[:, src_local]
        remaining = [j for j in range(x_prev.shape[1]) if j != src_local]
        x_curr = np.column_stack([x_prev[:, j] - betas[j] * x_s for j in remaining])
        schema = None

    context.state[_WORKING_STATE_KEY] = ArrayRecord([x_curr])
    stats = compute_local_raw_stats(x_curr, order=4)
    ctrl: dict[str, Any] = {"n": int(stats["n"]), "n_vars": x_curr.shape[1]}
    if schema is not None:
        ctrl["schema"] = schema
    reply = RecordDict({_STATS_KEY: stats_to_record(stats), CTRL_KEY: control_config(**ctrl)})
    return Message(reply, reply_to=msg)


def build_exact_client_app(loader: PartitionLoader[ContinuousDataset]) -> ClientApp:
    """Register :func:`handle_deflate_message` as a ``ClientApp``."""
    app = ClientApp()

    @app.train()
    def train(msg: Message, context: Context) -> Message:
        return handle_deflate_message(msg, context, loader)

    return app


def run_exact_server_loop(
    grid: Grid,
    n_clients: int,
    *,
    min_clients: int | None = None,
    use_lasso: bool = False,
    lmbda_type: str = "fixed",
) -> RCDResult:
    """Run the multi-round FedRCD-Exact protocol over `grid` and return the result.

    The server half of :func:`fedcausal.lingam.rcd.fedrcd_exact`, step for step.

    Parameters
    ----------
    grid : Grid
        Anything that can send and receive Flower messages.
    n_clients : int
        Number of supernodes to wait for before broadcasting.
    min_clients : int or None, optional
        How many usable replies each round needs. Defaults to `n_clients`.
    use_lasso : bool, optional
        If ``True``, estimate B with Adaptive Lasso; otherwise OLS.
    lmbda_type : str, optional
        Lambda selection for Lasso: ``"fixed"`` or ``"bic"``.

    Returns
    -------
    RCDResult
        The causal order and coefficient matrix, in original variable indices.

    Raises
    ------
    ConfigurationError
        If `min_clients` is not between 1 and `n_clients`.
    ClientFailureError
        If too few supernodes register, or too few reply in any round.
    SchemaMismatchError
        If the clients do not agree on their variables at round 0.
    """
    required = required_clients(n_clients, min_clients)
    node_ids = await_nodes(grid, n_clients)

    replies = grid.send_and_receive(
        broadcast(node_ids, RecordDict({CTRL_KEY: control_config(round=0)}))
    )
    good = usable_replies(replies, required, "stats")
    check_schemas(good)

    agg = _aggregate_replies(good)
    cov_original = np.asarray(agg["cov"], dtype=np.float64)
    n_total = int(agg["n"])
    n_vars = cov_original.shape[0]

    active_global = list(range(n_vars))
    causal_order: list[int] = []

    for step in range(n_vars - 1):
        n_active = len(active_global)
        tau = tau_matrix_4th(
            np.asarray(agg["c4"], dtype=np.float64),
            np.asarray(agg["c13"], dtype=np.float64),
            np.asarray(agg["c31"], dtype=np.float64),
            np.asarray(agg["c22"], dtype=np.float64),
        )
        src_local = int(np.argmin(np.sum(tau, axis=1)))
        causal_order.append(active_global[src_local])

        cov_curr = np.asarray(agg["cov"], dtype=np.float64)
        var_s = max(float(cov_curr[src_local, src_local]), 1e-15)
        betas = cov_curr[:, src_local] / var_s

        remaining_local = [j for j in range(n_active) if j != src_local]
        active_global = [active_global[j] for j in remaining_local]

        if step < n_vars - 2:
            round_replies = grid.send_and_receive(
                broadcast(
                    node_ids,
                    RecordDict(
                        {
                            CTRL_KEY: control_config(round=step + 1, src_local=src_local),
                            _BETAS_KEY: ArrayRecord([betas]),
                        }
                    ),
                )
            )
            agg = _aggregate_replies(usable_replies(round_replies, required, "stats"))

    if active_global:
        causal_order.append(active_global[0])

    if use_lasso:
        b = estimate_b_lasso_from_cov(
            causal_order, cov_original, n_samples=n_total, lmbda_type=lmbda_type
        )
    else:
        b = estimate_b_from_cov(causal_order, cov_original)

    return RCDResult(tuple(causal_order), b)


def build_exact_server_app(
    n_clients: int,
    *,
    min_clients: int | None = None,
    results: list[RCDResult] | None = None,
    **solver_kwargs: Any,
) -> ServerApp:
    """Register :func:`run_exact_server_loop` as a ``ServerApp``.

    Takes the same arguments. `results`, if given, receives the final `RCDResult`.
    """
    app = ServerApp()

    @app.main()
    def main(grid: Grid, context: Context) -> None:
        result = run_exact_server_loop(grid, n_clients, min_clients=min_clients, **solver_kwargs)
        if results is not None:
            results.append(result)

    return app


def run_flower_rcd_exact(
    data: ContinuousDataset,
    n_clients: int,
    **solver_kwargs: Any,
) -> RCDResult:
    """Run the multi-round FedRCD-Exact solver over a Flower simulation.

    As :func:`run_flower_rcd`, over ``p - 1`` rounds.

    Parameters
    ----------
    data : ContinuousDataset
        Full dataset to partition across clients.
    n_clients : int
        Number of clients and supernodes.
    **solver_kwargs : Any
        Passed through to :func:`run_exact_server_loop` (``use_lasso``,
        ``lmbda_type``).

    Returns
    -------
    RCDResult
        The causal order and coefficient matrix the solver estimated.

    Raises
    ------
    ConfigurationError
        If Flower's simulation extra is not installed.
    ClientFailureError
        If too few supernodes register, or too few reply in any round.
    SchemaMismatchError
        If the clients do not agree on their variables at round 0.
    """
    results: list[RCDResult] = []
    _simulate(
        build_exact_client_app,
        build_exact_server_app(n_clients, results=results, **solver_kwargs),
        data,
        n_clients,
    )
    if not results:
        raise ClientFailureError("the Flower simulation completed without producing a result")
    return results[0]


def _simulate(
    make_client: Callable[[PartitionLoader[ContinuousDataset]], ClientApp],
    server_app: ServerApp,
    data: ContinuousDataset,
    n_clients: int,
) -> None:
    """Partition `data` and drive both apps through Flower's simulation engine."""
    from fedcausal.data import contiguous_split
    from fedcausal.flower.loaders import InMemoryPartitions

    run_simulation = simulation_runner()
    parts = contiguous_split(data, n_clients)
    run_simulation(
        server_app=server_app,
        client_app=make_client(InMemoryPartitions(parts)),
        num_supernodes=n_clients,
    )


def _resolve_solver(solver: str) -> Callable[..., RCDResult]:
    """Look up a one-shot solver by name.

    Raises
    ------
    ConfigurationError
        If `solver` is not one of the names in `_SOLVERS`.
    """
    try:
        return _SOLVERS[solver]
    except KeyError as exc:
        raise ConfigurationError(
            f"unknown solver {solver!r}; use one of {sorted(_SOLVERS)}"
        ) from exc


def _aggregate_replies(replies: Iterable[Message]) -> Cumulants:
    """Unpack a batch of stats replies and aggregate them into global cumulants."""
    stats_list = [
        record_to_stats(cast(ArrayRecord, reply.content[_STATS_KEY]), ctrl_int(reply, "n"))
        for reply in replies
    ]
    return aggregate_and_compute_cumulants(stats_list)
