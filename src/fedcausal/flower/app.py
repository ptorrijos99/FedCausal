"""Flower ``ClientApp`` / ``ServerApp`` running FedGES on the Message API.

The server owns the loop in a low-level ``@app.main()`` that mirrors
:meth:`fedcausal.federated.server.Server.run` step for step, so a Flower run and
the in-process engine produce identical round-by-round models. Each iteration is
two message round-trips:

1. **build** -- every client rebuilds its local model, warm-started from the
   model in its ``context.state``, and replies with its local DAG and score.
2. **fuse** -- the server fuses the local DAGs and broadcasts the global DAG;
   every client fuses it into its own model, refines, and updates its state.

The algorithmic core (``GESLearner``, ``FusionOperator``, ``Convergence``) is
reused verbatim; only the transport differs. Where a client's data comes from is
a seam -- see :mod:`fedcausal.flower.data` -- so the same app runs in simulation
and in a real deployment.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from flwr.client import ClientApp
from flwr.common import ArrayRecord, Context, Message, MessageType, RecordDict
from flwr.server import Grid, ServerApp

from fedcausal.exceptions import ClientFailureError, ConfigurationError, SchemaMismatchError
from fedcausal.federated.learner import GESLearner, Refinement
from fedcausal.federated.model import BNModel
from fedcausal.flower.serde import (
    control_config,
    dag_to_record,
    record_to_dag,
    schema_fingerprint,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from fedcausal.federated.convergence import Convergence
    from fedcausal.federated.observer import RoundObserver
    from fedcausal.flower.data import PartitionLoader
    from fedcausal.fusion import FusionOperator

__all__ = [
    "build_client_app",
    "build_server_app",
    "handle_train_message",
    "run_server_loop",
]

_LOCAL_STATE_KEY = "local_dag"
_GLOBAL_STATE_KEY = "global_dag"
_LOCAL_KEY = "local"
_GLOBAL_KEY = "global"
_CTRL_KEY = "control"


def _load_local(context: Context, num_nodes: int) -> BNModel | None:
    """Return the client's persisted local model, or `None` if it has none yet."""
    if _LOCAL_STATE_KEY not in context.state.array_records:
        return None
    dag = record_to_dag(cast(ArrayRecord, context.state[_LOCAL_STATE_KEY]), num_nodes)
    return BNModel(dag)


def handle_train_message(
    msg: Message,
    context: Context,
    loader: PartitionLoader,
    client_fusion: FusionOperator,
    refinement: Refinement = "None",
) -> Message:
    """Handle one ``train`` message and return the reply.

    The whole client side of a round, with no Flower app machinery around it,
    so it can be driven directly -- by :func:`build_client_app`, by an app that
    reads its settings from ``run_config``, or by a test.

    Parameters
    ----------
    msg : Message
        The incoming message; its control record names the phase.
    context : Context
        This supernode's context. The local model persists in ``context.state``
        between rounds.
    loader : PartitionLoader
        Resolves this supernode's dataset; see :mod:`fedcausal.flower.data`.
    client_fusion : FusionOperator
        Fuses the client's local DAG with the broadcast global DAG.
    refinement : {"None", "FES", "BES", "GES"}, optional
        Refinement stage for the client's :class:`GESLearner`.

    Returns
    -------
    Message
        A reply carrying the client's local DAG, its score, and its schema.

    Raises
    ------
    ClientFailureError
        On an unknown phase, or a fuse message arriving before any build round.
    """
    data = loader(context)
    learner = GESLearner(refinement=refinement)
    previous = _load_local(context, data.n_vars)
    phase = str(msg.content[_CTRL_KEY]["phase"])

    if phase == "build":
        local = learner.build_local_model(previous, data)
    elif phase == "fuse":
        if previous is None:
            raise ClientFailureError("received a fuse message before any build round")
        global_dag = record_to_dag(cast(ArrayRecord, msg.content[_GLOBAL_KEY]), data.n_vars)
        fused_dag = client_fusion.fuse([previous.dag, global_dag])
        fused = BNModel(fused_dag, learner.score_dag(fused_dag, data))
        local = learner.refine_local_model(previous, fused, data)
    else:
        raise ClientFailureError(f"unknown phase {phase!r}")

    context.state[_LOCAL_STATE_KEY] = dag_to_record(local.dag)
    reply = RecordDict(
        {
            _LOCAL_KEY: dag_to_record(local.dag),
            _CTRL_KEY: control_config(
                node_index=int(context.node_config.get("partition-id", -1)),
                score=local.score,
                n_vars=data.n_vars,
                schema=schema_fingerprint(data),
            ),
        }
    )
    return Message(content=reply, reply_to=msg)


def build_client_app(
    loader: PartitionLoader,
    client_fusion: FusionOperator,
    refinement: Refinement = "None",
) -> ClientApp:
    """Build a ``ClientApp`` running the FedGES client logic.

    A thin registration wrapper around :func:`handle_train_message`; see there
    for the parameters.

    Returns
    -------
    ClientApp
        The configured client application.
    """
    app = ClientApp()

    @app.train()
    def train(msg: Message, context: Context) -> Message:
        return handle_train_message(msg, context, loader, client_fusion, refinement)

    return app


def build_server_app(
    global_fusion: FusionOperator,
    convergence: Convergence,
    n_clients: int,
    num_nodes: int,
    n_iterations: int = 5,
    observer: RoundObserver | None = None,
    min_clients: int | None = None,
) -> ServerApp:
    """Build a ``ServerApp`` driving the FedGES round loop.

    Parameters
    ----------
    global_fusion : FusionOperator
        Server-side fusion of the clients' local DAGs.
    convergence : Convergence
        Round-loop stopping criterion.
    n_clients : int
        Number of supernodes to wait for before the first round.
    num_nodes : int
        Variable count, needed to rebuild DAGs from replies.
    n_iterations : int, optional
        Maximum number of rounds (default 5).
    observer : RoundObserver or None, optional
        Receives build and server events as the loop runs.
    min_clients : int or None, optional
        How many usable replies a round needs. Defaults to `n_clients`, so a
        failed client aborts the run rather than silently shrinking the
        ensemble. Lower it to tolerate dropouts.

    Returns
    -------
    ServerApp
        The configured server application. The final global model is stored on
        ``context.state["global_dag"]``.

    Raises
    ------
    ConfigurationError
        If `min_clients` is not between 1 and `n_clients`.
    """
    app = ServerApp()

    @app.main()
    def main(grid: Grid, context: Context) -> None:
        run_server_loop(
            grid,
            context,
            global_fusion=global_fusion,
            convergence=convergence,
            n_clients=n_clients,
            num_nodes=num_nodes,
            n_iterations=n_iterations,
            observer=observer,
            min_clients=min_clients,
        )

    return app


def run_server_loop(
    grid: Grid,
    context: Context,
    *,
    global_fusion: FusionOperator,
    convergence: Convergence,
    n_clients: int,
    num_nodes: int,
    n_iterations: int = 5,
    observer: RoundObserver | None = None,
    min_clients: int | None = None,
) -> BNModel | None:
    """Run the FedGES round loop over `grid` and return the final global model.

    The whole server side, with no Flower app machinery around it, so an app
    that reads its settings from ``run_config`` can call it directly. See
    :func:`build_server_app` for the parameters.

    Returns
    -------
    BNModel or None
        The final global model, also stored on ``context.state["global_dag"]``.
        `None` if the loop stopped before any fusion.

    Raises
    ------
    ConfigurationError
        If `min_clients` is not between 1 and `n_clients`.
    ClientFailureError
        If too few supernodes register, or too few reply in a round.
    SchemaMismatchError
        If the clients do not agree on their variables.
    """
    required = n_clients if min_clients is None else min_clients
    if not 1 <= required <= n_clients:
        raise ConfigurationError(
            f"min_clients must be between 1 and n_clients={n_clients}, got {min_clients}"
        )

    node_ids = _await_nodes(grid, n_clients)
    global_model: BNModel | None = None
    schema: str | None = None

    for iteration in range(1, n_iterations + 1):
        build_replies = grid.send_and_receive(
            _broadcast(node_ids, RecordDict({_CTRL_KEY: control_config(phase="build")}))
        )
        local_models, schema = _collect_locals(build_replies, num_nodes, required, schema)
        if observer is not None:
            for index, model in enumerate(local_models):
                observer.on_round("build", iteration, index, model)

        if convergence.check_convergence(local_models):
            break

        fused_dag = global_fusion.fuse([m.dag for m in local_models])
        global_model = BNModel(fused_dag)
        if observer is not None:
            observer.on_round("server", iteration, -1, global_model)

        fuse_replies = grid.send_and_receive(
            _broadcast(
                node_ids,
                RecordDict(
                    {
                        _CTRL_KEY: control_config(phase="fuse"),
                        _GLOBAL_KEY: dag_to_record(fused_dag),
                    }
                ),
            )
        )
        _check_replies(fuse_replies, required, "fuse")

    if global_model is not None:
        context.state[_GLOBAL_STATE_KEY] = dag_to_record(global_model.dag)
    return global_model


def _await_nodes(grid: Grid, n_clients: int, tries: int = 60, delay: float = 0.3) -> list[int]:
    """Poll until at least `n_clients` supernodes have registered.

    Raises
    ------
    ClientFailureError
        If too few register within ``tries * delay`` seconds.
    """
    import time

    node_ids: list[int] = []
    for _ in range(tries):
        node_ids = list(grid.get_node_ids())
        if len(node_ids) >= n_clients:
            return sorted(node_ids)
        time.sleep(delay)
    raise ClientFailureError(
        f"only {len(node_ids)} of {n_clients} supernodes registered within {tries * delay:.0f}s"
    )


def _broadcast(node_ids: Sequence[int], content: RecordDict) -> list[Message]:
    """Build one identical ``train`` message per node."""
    return [
        Message(content=content, dst_node_id=nid, message_type=MessageType.TRAIN, group_id="0")
        for nid in node_ids
    ]


def _check_replies(replies: Iterable[Message], required: int, phase: str) -> list[Message]:
    """Return the successful replies, or raise if too few came back.

    Failures are never dropped quietly: a client that errors out shrinks the
    ensemble the next fusion sees, and a run that silently learns from four
    sites instead of five is worse than one that stops.

    Raises
    ------
    ClientFailureError
        If fewer than `required` replies succeeded.
    """
    good, errors = [], []
    for reply in replies:
        if reply.has_error():
            errors.append(f"node {reply.metadata.src_node_id}: {reply.error!r}")
        else:
            good.append(reply)
    if len(good) < required:
        detail = "; ".join(errors) if errors else "no error detail reported"
        raise ClientFailureError(
            f"{phase} round got {len(good)} usable replies, needs {required} ({detail})"
        )
    return good


def _collect_locals(
    replies: Iterable[Message],
    num_nodes: int,
    required: int,
    known_schema: str | None,
) -> tuple[list[BNModel], str]:
    """Rebuild the clients' local models, ordered by their reported index.

    Ordering is by the index the client reports rather than by node id, so
    fusion inputs arrive in the same order the in-process engine would produce.

    Raises
    ------
    ClientFailureError
        If too few clients replied successfully.
    SchemaMismatchError
        If the clients do not agree on variable names and cardinalities, or
        disagree with an earlier round.
    """
    good = _check_replies(replies, required, "build")

    schemas = {_ctrl_str(r, "schema") for r in good}
    if len(schemas) > 1:
        sizes = sorted({_ctrl_int(r, "n_vars") for r in good})
        raise SchemaMismatchError(
            f"clients disagree on their variables ({len(schemas)} distinct schemas, "
            f"variable counts {sizes}). Structures travel as bare node indices, so "
            "every client must hold the same variables in the same order."
        )
    schema = schemas.pop()
    if known_schema is not None and schema != known_schema:
        raise SchemaMismatchError("a client's schema changed between rounds")

    by_index: dict[int, BNModel] = {}
    for reply in good:
        dag = record_to_dag(cast(ArrayRecord, reply.content[_LOCAL_KEY]), num_nodes)
        by_index[_ctrl_int(reply, "node_index")] = BNModel(dag, _ctrl_float(reply, "score"))
    return [by_index[i] for i in sorted(by_index)], schema


def _ctrl_str(reply: Message, key: str) -> str:
    """Read a string out of a reply's control record."""
    return str(reply.content[_CTRL_KEY][key])


def _ctrl_int(reply: Message, key: str) -> int:
    """Read an integer out of a reply's control record."""
    return int(cast(int, reply.content[_CTRL_KEY][key]))


def _ctrl_float(reply: Message, key: str) -> float:
    """Read a float out of a reply's control record."""
    return float(cast(float, reply.content[_CTRL_KEY][key]))
