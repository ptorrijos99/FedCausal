"""The FedRCD family over Flower.

The one-shot solvers need one round trip: every site replies with its moment
sums, row count and a fingerprint of its variable names, and the server checks
the fingerprints, aggregates the sums into cumulants and runs the solver.

``fedrcd_exact`` takes ``p - 1`` rounds. Each identifies a source, and every
site deflates its own data by it and replies with fresh sums; the deflated
data stays in the site's ``context.state`` between rounds.

:func:`run_server_loop` and :func:`run_exact_server_loop` take any ``Grid``;
:func:`run_flower_rcd` and :func:`run_flower_rcd_exact` run a list of sites on
Flower's simulation engine.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np
from flwr.client import ClientApp
from flwr.common import ArrayRecord, ConfigRecord, Context, Message, RecordDict
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
from fedcausal.lingam.rcd import RCDResult, _coefficients, exact_order, fedrcd

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

    import numpy.typing as npt

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
        A reply carrying the mean and moment sums, the local row count, and
        the client's variable-name fingerprint.
    """
    data = loader(context)
    stats = compute_local_raw_stats(data.values, _order(msg))
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
    method: str = "deflation",
    *,
    order: int = 4,
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
    method : {"deflation", "ratio", "sort"}, optional
        Passed to :func:`fedcausal.lingam.fedrcd`.
    order : {3, 4}, optional
        Order of the moment sums the sites send and of the tau discriminator.
    min_clients : int or None, optional
        How many usable replies the round needs. Default `n_clients`, so a
        failed client stops the run; lower it to tolerate dropouts.
    **solver_kwargs : Any
        Passed to :func:`fedcausal.lingam.fedrcd` (``use_lasso``, ``lmbda_type``).

    Returns
    -------
    RCDResult
        The causal order and coefficient matrix the solver estimated.

    Raises
    ------
    ConfigurationError
        If `method` or `order` is not supported, or `min_clients` is not
        between 1 and `n_clients`.
    ClientFailureError
        If too few supernodes register, or too few reply.
    SchemaMismatchError
        If the clients do not agree on their variables.
    """
    _check_method(method, order)
    required = required_clients(n_clients, min_clients)

    node_ids = await_nodes(grid, n_clients)
    replies = grid.send_and_receive(
        broadcast(node_ids, RecordDict({CTRL_KEY: control_config(phase="stats", order=order)}))
    )
    good = usable_replies(replies, required, "stats")
    check_schemas(good)

    stats = _aggregate_replies(good)
    return fedrcd(stats, method=method, order=order, **solver_kwargs)  # type: ignore[arg-type]


def build_server_app(
    n_clients: int,
    method: str = "deflation",
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
        result = run_server_loop(grid, n_clients, method, min_clients=min_clients, **solver_kwargs)
        if results is not None:
            results.append(result)

    return app


def run_flower_rcd(
    sites: Sequence[ContinuousDataset],
    method: str = "deflation",
    **solver_kwargs: Any,
) -> RCDResult:
    """Run a one-shot FedRCD solver over a Flower simulation and return the result.

    One simulated supernode per site. Needs ``fedcausal[flower-simulation]``.

    Parameters
    ----------
    sites : sequence of ContinuousDataset
        Each site's data, over the same variables.
    method : {"deflation", "ratio", "sort"}, optional
        Passed to :func:`fedcausal.lingam.fedrcd`.
    **solver_kwargs : Any
        Passed to :func:`run_server_loop` (``order``, ``use_lasso``, ``lmbda_type``).

    Returns
    -------
    RCDResult
        The causal order and coefficient matrix the solver estimated.

    Raises
    ------
    ConfigurationError
        If Flower's simulation extra is not installed, or `method` or `order`
        is not supported.
    ClientFailureError
        If too few supernodes register, or too few reply.
    SchemaMismatchError
        If the clients do not agree on their variables.
    """
    results: list[RCDResult] = []
    _simulate(
        build_client_app,
        build_server_app(len(sites), method, results=results, **solver_kwargs),
        sites,
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
        keep = np.arange(x_prev.shape[1]) != src_local
        x_curr = x_prev[:, keep] - np.outer(x_prev[:, src_local], betas[keep])
        schema = None

    context.state[_WORKING_STATE_KEY] = ArrayRecord([x_curr])
    stats = compute_local_raw_stats(x_curr, _order(msg))
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
    order: int = 4,
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
    order : {3, 4}, optional
        Order of the moment sums the sites send and of the tau discriminator.
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
        If `order` is not 3 or 4, or `min_clients` is not between 1 and `n_clients`.
    ClientFailureError
        If too few supernodes register, or too few reply in any round.
    SchemaMismatchError
        If the clients do not agree on their variables at round 0.
    """
    _check_method("deflation", order)
    required = required_clients(n_clients, min_clients)
    node_ids = await_nodes(grid, n_clients)

    replies = grid.send_and_receive(
        broadcast(node_ids, RecordDict({CTRL_KEY: control_config(round=0, order=order)}))
    )
    good = usable_replies(replies, required, "stats")
    check_schemas(good)

    first = _aggregate_replies(good)
    step = 0

    def deflate(src: int, betas: npt.NDArray[np.float64]) -> Cumulants:
        nonlocal step
        step += 1
        replies = grid.send_and_receive(
            broadcast(
                node_ids,
                RecordDict(
                    {
                        CTRL_KEY: control_config(round=step, src_local=src, order=order),
                        _BETAS_KEY: ArrayRecord([betas]),
                    }
                ),
            )
        )
        return _aggregate_replies(usable_replies(replies, required, "stats"))

    causal_order = exact_order(first, deflate, order)  # type: ignore[arg-type]
    cov = np.asarray(first["cov"], dtype=np.float64)
    b = _coefficients(causal_order, cov, int(first["n"]), use_lasso, lmbda_type)

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
    sites: Sequence[ContinuousDataset],
    **solver_kwargs: Any,
) -> RCDResult:
    """Run the multi-round FedRCD-Exact solver over a Flower simulation.

    As :func:`run_flower_rcd`, over ``p - 1`` rounds.

    Parameters
    ----------
    sites : sequence of ContinuousDataset
        Each site's data, over the same variables.
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
        build_exact_server_app(len(sites), results=results, **solver_kwargs),
        sites,
    )
    if not results:
        raise ClientFailureError("the Flower simulation completed without producing a result")
    return results[0]


def _simulate(
    make_client: Callable[[PartitionLoader[ContinuousDataset]], ClientApp],
    server_app: ServerApp,
    sites: Sequence[ContinuousDataset],
) -> None:
    """Drive both apps through Flower's simulation engine, one supernode per site."""
    from fedcausal.flower.loaders import InMemoryPartitions

    run_simulation = simulation_runner()
    run_simulation(
        server_app=server_app,
        client_app=make_client(InMemoryPartitions(list(sites))),
        num_supernodes=len(sites),
    )


def _check_method(method: str, order: int) -> None:
    if method not in ("deflation", "ratio", "sort"):
        raise ConfigurationError(f"unknown method {method!r}; use 'deflation', 'ratio' or 'sort'")
    if order not in (3, 4):
        raise ConfigurationError(f"order must be 3 or 4, got {order!r}")


def _order(msg: Message) -> int:
    """Return the order of moment sums the server asked for, 4 if it did not say."""
    if CTRL_KEY not in msg.content:
        return 4
    return int(cast(int, cast(ConfigRecord, msg.content[CTRL_KEY]).get("order", 4)))


def _aggregate_replies(replies: Iterable[Message]) -> Cumulants:
    """Unpack a batch of stats replies and aggregate them into global cumulants."""
    stats_list = [
        record_to_stats(cast(ArrayRecord, reply.content[_STATS_KEY]), ctrl_int(reply, "n"))
        for reply in replies
    ]
    return aggregate_and_compute_cumulants(stats_list)
