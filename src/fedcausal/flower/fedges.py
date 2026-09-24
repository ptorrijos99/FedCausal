"""FedGES over Flower: a ``ClientApp``, a ``ServerApp``, and the round loop.

The server owns the loop in a low-level ``@app.main()`` that follows
:meth:`fedcausal.fedges.server.Server.run` step for step, so a Flower run and
the in-process engine produce identical models round by round. Each iteration is
two message round trips:

1. **build** -- every client rebuilds its local model, warm-started from the
   model in its ``context.state``, and replies with its local DAG and score.
2. **fuse** -- the server fuses the local DAGs and broadcasts the global DAG;
   every client fuses it into its own model, refines, and updates its state.

``GESLearner``, the fusion operators and the convergence criteria are reused
untouched; only the transport differs. Where a client's data comes from is a
seam (:mod:`fedcausal.flower.loaders`), which is what lets the same app run in
simulation and in a real deployment.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from flwr.client import ClientApp
from flwr.common import ArrayRecord, Context, Message, RecordDict
from flwr.server import Grid, ServerApp

from fedcausal.exceptions import ClientFailureError, SchemaMismatchError
from fedcausal.fedges.learner import GESLearner, Refinement
from fedcausal.fedges.model import BNModel
from fedcausal.flower._grid import (
    CTRL_KEY,
    await_nodes,
    broadcast,
    check_schemas,
    ctrl_float,
    ctrl_int,
    required_clients,
    simulation_runner,
    usable_replies,
)
from fedcausal.flower.serde import (
    control_config,
    dag_to_record,
    record_to_dag,
    schema_fingerprint,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from fedcausal.data import DiscreteDataset
    from fedcausal.fedges.convergence import Convergence
    from fedcausal.fedges.observer import RoundObserver
    from fedcausal.flower.loaders import PartitionLoader
    from fedcausal.fusion import FusionOperator

__all__ = [
    "build_client_app",
    "build_server_app",
    "handle_train_message",
    "run_flower_fedges",
    "run_server_loop",
]

_LOCAL_STATE_KEY = "local_dag"
_GLOBAL_STATE_KEY = "global_dag"
_LOCAL_KEY = "local"
_GLOBAL_KEY = "global"


def handle_train_message(
    msg: Message,
    context: Context,
    loader: PartitionLoader[DiscreteDataset],
    client_fusion: FusionOperator,
    refinement: Refinement = "None",
) -> Message:
    """Handle one ``train`` message and return the reply.

    The whole client side of a round, with no Flower app machinery around it,
    so it can be driven by :func:`build_client_app`, by an app reading its
    settings from ``run_config``, or by a test.

    Parameters
    ----------
    msg : Message
        The incoming message; its control record names the phase.
    context : Context
        This supernode's context. The local model persists in ``context.state``
        between rounds.
    loader : PartitionLoader
        Resolves this supernode's dataset.
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
    phase = str(msg.content[CTRL_KEY]["phase"])

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
            CTRL_KEY: control_config(
                node_index=int(context.node_config.get("partition-id", -1)),
                score=local.score,
                n_vars=data.n_vars,
                schema=schema_fingerprint(data),
            ),
        }
    )
    return Message(content=reply, reply_to=msg)


def build_client_app(
    loader: PartitionLoader[DiscreteDataset],
    client_fusion: FusionOperator,
    refinement: Refinement = "None",
) -> ClientApp:
    """Register :func:`handle_train_message` as a ``ClientApp``."""
    app = ClientApp()

    @app.train()
    def train(msg: Message, context: Context) -> Message:
        return handle_train_message(msg, context, loader, client_fusion, refinement)

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
    reading its settings from ``run_config`` can call it directly and a test can
    drive it over an in-process stub grid with no Ray anywhere.

    Parameters
    ----------
    grid : Grid
        Anything that can send and receive Flower messages.
    context : Context
        The server's context. The final global DAG is stored on
        ``context.state["global_dag"]``.
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
    BNModel or None
        The final global model, or `None` if the loop stopped before any
        fusion.

    Raises
    ------
    ConfigurationError
        If `min_clients` is not between 1 and `n_clients`.
    ClientFailureError
        If too few supernodes register, or too few reply in a round.
    SchemaMismatchError
        If the clients do not agree on their variables.
    """
    required = required_clients(n_clients, min_clients)
    node_ids = await_nodes(grid, n_clients)
    global_model: BNModel | None = None
    schema: str | None = None

    for iteration in range(1, n_iterations + 1):
        build_replies = grid.send_and_receive(
            broadcast(node_ids, RecordDict({CTRL_KEY: control_config(phase="build")}))
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
            broadcast(
                node_ids,
                RecordDict(
                    {
                        CTRL_KEY: control_config(phase="fuse"),
                        _GLOBAL_KEY: dag_to_record(fused_dag),
                    }
                ),
            )
        )
        usable_replies(fuse_replies, required, "fuse")

    if global_model is not None:
        context.state[_GLOBAL_STATE_KEY] = dag_to_record(global_model.dag)
    return global_model


def build_server_app(
    global_fusion: FusionOperator,
    convergence: Convergence,
    n_clients: int,
    num_nodes: int,
    n_iterations: int = 5,
    observer: RoundObserver | None = None,
    min_clients: int | None = None,
) -> ServerApp:
    """Register :func:`run_server_loop` as a ``ServerApp``.

    Takes the same arguments; see there. The final global model ends up on
    ``context.state["global_dag"]``.
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


def run_flower_fedges(
    data: DiscreteDataset,
    n_clients: int,
    n_iterations: int = 5,
    client_fusion: FusionOperator | None = None,
    server_fusion: FusionOperator | None = None,
    convergence: Convergence | None = None,
    refinement: Refinement = "None",
    observer: RoundObserver | None = None,
) -> BNModel | None:
    """Run FedGES over a Flower simulation and return the final global model.

    :func:`fedcausal.fedges.run_fedges` over the Flower transport: splits `data`
    into `n_clients` contiguous partitions and drives the same round loop. Both
    read their defaults from :mod:`fedcausal.fedges.defaults`, so the two
    transports cannot drift apart.

    Simulation runs every site inside one process and needs Flower's Ray-based
    engine (``fedcausal[flower-simulation]``). A deployment across real machines
    needs neither; see ``docs/deployment.md``.

    Parameters
    ----------
    data : DiscreteDataset
        Full dataset to partition across clients.
    n_clients : int
        Number of clients and supernodes.
    n_iterations : int, optional
        Maximum number of rounds (default 5).
    client_fusion, server_fusion : FusionOperator or None, optional
        Fusion operators; default to ``ConsensusUnion()`` / ``Consensus(50)``.
    convergence : Convergence or None, optional
        Stopping criterion; defaults to ``ScoreImprovement or ModelEquality``.
    refinement : {"None", "FES", "BES", "GES"}, optional
        Client refinement stage.
    observer : RoundObserver or None, optional
        Receives build and server events from the server side.

    Returns
    -------
    BNModel or None
        The final global model, or `None` if the run converged before any
        fusion.

    Raises
    ------
    ConfigurationError
        If Flower's simulation extra is not installed.
    """
    from fedcausal.data import contiguous_split
    from fedcausal.fedges.defaults import (
        default_client_fusion,
        default_convergence,
        default_server_fusion,
    )
    from fedcausal.fedges.observer import ModelCollector
    from fedcausal.flower.loaders import InMemoryPartitions

    run_simulation = simulation_runner()
    parts = contiguous_split(data, n_clients)
    collector = ModelCollector()
    fanout = _Fanout([collector] + ([observer] if observer is not None else []))

    client_app = build_client_app(
        InMemoryPartitions(parts),
        client_fusion or default_client_fusion(),
        refinement,
    )
    server_app = build_server_app(
        server_fusion or default_server_fusion(),
        convergence or default_convergence(),
        n_clients,
        data.n_vars,
        n_iterations,
        fanout,
    )
    run_simulation(server_app=server_app, client_app=client_app, num_supernodes=n_clients)
    return collector.globals[-1] if collector.globals else None


def _load_local(context: Context, num_nodes: int) -> BNModel | None:
    """Return the client's persisted local model, or `None` if it has none yet."""
    if _LOCAL_STATE_KEY not in context.state.array_records:
        return None
    dag = record_to_dag(cast(ArrayRecord, context.state[_LOCAL_STATE_KEY]), num_nodes)
    return BNModel(dag)


def _collect_locals(
    replies: Iterable[Message],
    num_nodes: int,
    required: int,
    known_schema: str | None,
) -> tuple[list[BNModel], str]:
    """Rebuild the clients' local models in a stable order.

    Ordered by the partition id each client reports, then by node id: in
    simulation that is the in-process engine's order, and in a deployment,
    where no partition id is set, it is still the same every round.

    Raises
    ------
    ClientFailureError
        If too few clients replied successfully.
    SchemaMismatchError
        If the clients do not agree on their variables, or disagree with an
        earlier round.
    """
    good = usable_replies(replies, required, "build")
    schema = check_schemas(good)
    if known_schema is not None and schema != known_schema:
        raise SchemaMismatchError("a client's schema changed between rounds")

    keyed = []
    for reply in good:
        dag = record_to_dag(cast(ArrayRecord, reply.content[_LOCAL_KEY]), num_nodes)
        key = (ctrl_int(reply, "node_index"), reply.metadata.src_node_id)
        keyed.append((key, BNModel(dag, ctrl_float(reply, "score"))))
    return [model for _, model in sorted(keyed, key=lambda item: item[0])], schema


class _Fanout:
    """Dispatch each round event to several observers."""

    def __init__(self, observers: list[RoundObserver]) -> None:
        self._observers = list(observers)

    def on_round(self, stage: str, iteration: int, client_id: int, model: BNModel) -> None:
        """Forward one round event to every registered observer."""
        for obs in self._observers:
            obs.on_round(stage, iteration, client_id, model)
