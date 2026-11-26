"""The FedRCD family over Flower.

The one-round methods need one round trip: every site replies with its moment
sums, row count and a fingerprint of its variable names, and the server checks
the fingerprints, aggregates the sums into cumulants and runs the solver.

``method="exact"`` takes ``p - 1`` rounds. Each identifies a source, and every
site deflates its own data by it and replies with fresh sums; the deflated
data stays in the site's ``context.state`` between rounds.

The same functions serve every method, as :class:`fedcausal.FedRCD` does:
:func:`handle_train_message` on the sites, :func:`run_server_loop` on the
server over any ``Grid``, and :func:`run_flower_rcd` for a list of sites on
Flower's simulation engine.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np
from causalbase.data import ContinuousDataset
from flwr.client import ClientApp
from flwr.common import ArrayRecord, ConfigRecord, Context, Message, RecordDict
from flwr.server import Grid, ServerApp

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
    record_to_sums,
    sums_to_record,
)
from fedcausal.lingam.cumulants import cumulants_from_sums, moment_sums
from fedcausal.lingam.rcd import RCDResult, _coefficients, exact_order, fedrcd

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    import numpy.typing as npt

    from fedcausal.flower.loaders import PartitionLoader
    from fedcausal.lingam.cumulants import Cumulants
    from fedcausal.lingam.rcd import LambdaSelection, Pruning

__all__ = [
    "build_client_app",
    "build_server_app",
    "handle_train_message",
    "run_flower_rcd",
    "run_server_loop",
]

_METHODS = ("deflation", "exact", "ratio", "sort")
_STATS_KEY = "stats"
_BETAS_KEY = "betas"
_WORKING_STATE_KEY = "rcd_working"


def handle_train_message(
    msg: Message, context: Context, loader: PartitionLoader[ContinuousDataset]
) -> Message:
    """Do this site's part of a FedRCD round, for any method, and reply.

    A one-round method asks once for the moment sums of the site's data. The
    exact method's messages carry a ``round``: round 0 asks for the same sums,
    and every later round deflates the working matrix kept in
    ``context.state`` by the source and coefficients the server sent
    (``x_j -= beta_j * x_s``), drops the source column and asks for the sums
    of what is left.

    Parameters
    ----------
    msg : Message
        The incoming message.
    context : Context
        This supernode's context.
    loader : PartitionLoader
        Resolves this supernode's dataset.

    Returns
    -------
    Message
        The mean and moment sums, the row count and, on the first round, the
        fingerprint of the variable names.
    """
    control = cast(ConfigRecord, msg.content[CTRL_KEY]) if CTRL_KEY in msg.content else None
    rounds = control is not None and "round" in control
    round_idx = int(cast(int, control["round"])) if control is not None and rounds else 0
    schema = None
    if round_idx == 0:
        data = loader(context)
        x = np.ascontiguousarray(data.values, dtype=np.float64)
        schema = names_fingerprint(data.names)
    else:
        assert control is not None
        src_local = int(cast(int, control["src_local"]))
        betas = cast(ArrayRecord, msg.content[_BETAS_KEY]).to_numpy_ndarrays()[0]
        x_prev = cast(ArrayRecord, context.state[_WORKING_STATE_KEY]).to_numpy_ndarrays()[0]
        keep = np.arange(x_prev.shape[1]) != src_local
        x = x_prev[:, keep] - np.outer(x_prev[:, src_local], betas[keep])
    if rounds:
        context.state[_WORKING_STATE_KEY] = ArrayRecord([x])

    sums = moment_sums(x, _order(msg))
    ctrl: dict[str, Any] = {"n": int(sums["n"]), "n_vars": x.shape[1]}
    if schema is not None:
        ctrl["schema"] = schema
    reply = RecordDict({_STATS_KEY: sums_to_record(sums), CTRL_KEY: control_config(**ctrl)})
    return Message(reply, reply_to=msg)


def build_client_app(loader: PartitionLoader[ContinuousDataset]) -> ClientApp:
    """Register :func:`handle_train_message` as a ``ClientApp``."""
    app = ClientApp()

    @app.train()
    def train(msg: Message, context: Context) -> Message:
        return handle_train_message(msg, context, loader)

    return app


def run_server_loop(
    grid: Grid,
    *,
    n_clients: int,
    method: str = "deflation",
    order: int = 4,
    pruning: Pruning = "adaptive_lasso",
    lambda_selection: LambdaSelection = "bic",
    min_clients: int | None = None,
) -> RCDResult:
    """Run a FedRCD method over `grid` and return the result.

    Parameters
    ----------
    grid : Grid
        Anything that can send and receive Flower messages.
    n_clients : int
        Number of supernodes to wait for before broadcasting.
    method, order, pruning, lambda_selection
        As in :class:`fedcausal.FedRCD`. ``"exact"`` takes ``p - 1`` rounds,
        the others one.
    min_clients : int or None, optional
        How many usable replies each round needs. Default `n_clients`, so a
        failed client stops the run; lower it to tolerate dropouts.

    Returns
    -------
    RCDResult
        The causal order and coefficient matrix, in the original variable
        indices: what the in-process solver gives on the same sites.

    Raises
    ------
    ConfigurationError
        If an option is not supported, or `min_clients` is not between 1 and
        `n_clients`.
    ClientFailureError
        If too few supernodes register, or too few reply in a round.
    SchemaMismatchError
        If the clients do not agree on their variables.
    """
    _check_options(method, order, pruning, lambda_selection)
    required = required_clients(n_clients, min_clients)
    node_ids = await_nodes(grid, n_clients)
    exact = method == "exact"

    first_control = control_config(round=0, order=order) if exact else control_config(order=order)
    replies = grid.send_and_receive(broadcast(node_ids, RecordDict({CTRL_KEY: first_control})))
    good = usable_replies(replies, required, "stats")
    check_schemas(good)
    first = _aggregate_replies(good)
    if not exact:
        return fedrcd(
            first,
            method=method,  # type: ignore[arg-type]
            order=order,  # type: ignore[arg-type]
            pruning=pruning,
            lambda_selection=lambda_selection,
        )

    step = 0

    def deflate(src: int, betas: npt.NDArray[np.float64]) -> Cumulants:
        nonlocal step
        step += 1
        content = RecordDict(
            {
                CTRL_KEY: control_config(round=step, src_local=src, order=order),
                _BETAS_KEY: ArrayRecord([betas]),
            }
        )
        replies = grid.send_and_receive(broadcast(node_ids, content))
        return _aggregate_replies(usable_replies(replies, required, "stats"))

    causal_order = exact_order(first, deflate, order)  # type: ignore[arg-type]
    cov = np.asarray(first["cov"], dtype=np.float64)
    coef = _coefficients(causal_order, cov, int(first["n"]), pruning, lambda_selection)
    return RCDResult(tuple(causal_order), coef)


def build_server_app(
    *,
    n_clients: int,
    method: str = "deflation",
    order: int = 4,
    pruning: Pruning = "adaptive_lasso",
    lambda_selection: LambdaSelection = "bic",
    min_clients: int | None = None,
    results: list[RCDResult] | None = None,
) -> ServerApp:
    """Register :func:`run_server_loop` as a ``ServerApp``; takes the same arguments.

    `results`, if given, receives the final `RCDResult`, since a
    ``ServerApp`` cannot return one.
    """
    app = ServerApp()

    @app.main()
    def main(grid: Grid, context: Context) -> None:
        result = run_server_loop(
            grid,
            n_clients=n_clients,
            method=method,
            order=order,
            pruning=pruning,
            lambda_selection=lambda_selection,
            min_clients=min_clients,
        )
        if results is not None:
            results.append(result)

    return app


def run_flower_rcd(
    sites: Sequence[ContinuousDataset],
    *,
    method: str = "deflation",
    order: int = 4,
    pruning: Pruning = "adaptive_lasso",
    lambda_selection: LambdaSelection = "bic",
) -> RCDResult:
    """Run FedRCD on Flower's simulation engine and return the result.

    The Flower counterpart of :class:`fedcausal.FedRCD`, one simulated
    supernode per site. Needs ``fedcausal[federated-simulation]``.

    Parameters
    ----------
    sites : sequence of ContinuousDataset
        Each site's data, over the same variables.
    method, order, pruning, lambda_selection
        As in :class:`fedcausal.FedRCD`.

    Returns
    -------
    RCDResult
        The causal order and coefficient matrix.

    Raises
    ------
    ConfigurationError
        If Flower's simulation extra is not installed, or an option is not
        supported.
    ClientFailureError
        If too few supernodes register, or too few reply.
    SchemaMismatchError
        If the clients do not agree on their variables.
    """
    from fedcausal.flower.loaders import InMemoryPartitions

    _check_options(method, order, pruning, lambda_selection)
    results: list[RCDResult] = []
    server_app = build_server_app(
        n_clients=len(sites),
        method=method,
        order=order,
        pruning=pruning,
        lambda_selection=lambda_selection,
        results=results,
    )
    run_simulation = simulation_runner()
    run_simulation(
        server_app=server_app,
        client_app=build_client_app(InMemoryPartitions(list(sites))),
        num_supernodes=len(sites),
    )
    if not results:
        raise ClientFailureError("the Flower simulation completed without producing a result")
    return results[0]


def _check_options(method: str, order: int, pruning: object, lambda_selection: object) -> None:
    if method not in _METHODS:
        raise ConfigurationError(f"unknown method {method!r}; use one of {_METHODS}")
    if order not in (3, 4):
        raise ConfigurationError(f"order must be 3 or 4, got {order!r}")
    if pruning not in ("adaptive_lasso", None):
        raise ConfigurationError(f"pruning must be 'adaptive_lasso' or None, got {pruning!r}")
    if lambda_selection not in ("bic", "fixed"):
        raise ConfigurationError(
            f"lambda_selection must be 'bic' or 'fixed', got {lambda_selection!r}"
        )


def _order(msg: Message) -> int:
    """Return the order of moment sums the server asked for, 4 if it did not say."""
    if CTRL_KEY not in msg.content:
        return 4
    return int(cast(int, cast(ConfigRecord, msg.content[CTRL_KEY]).get("order", 4)))


def _aggregate_replies(replies: Iterable[Message]) -> Cumulants:
    """Unpack a batch of replies and aggregate their moment sums into cumulants."""
    sums = [
        record_to_sums(cast(ArrayRecord, reply.content[_STATS_KEY]), ctrl_int(reply, "n"))
        for reply in replies
    ]
    return cumulants_from_sums(sums)
