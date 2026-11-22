"""FedGES over Flower.

Each round is two round trips. In **build** every site rebuilds its model, from
the one kept in its ``context.state``, and replies with it and its score. In
**fuse** the server sends the global model and every site merges it in. The
server runs :func:`fedcausal.fedges.run_rounds`, the loop
:class:`fedcausal.FedGES` runs in one process, so both give the same models.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, cast

import numpy as np
from flwr.client import ClientApp
from flwr.common import ArrayRecord, Context, Message, RecordDict
from flwr.server import Grid, ServerApp

from fedcausal.data import DiscreteDataset
from fedcausal.exceptions import ClientFailureError, SchemaMismatchError
from fedcausal.fedges.convergence import default_convergence
from fedcausal.fedges.model import BNModel
from fedcausal.fedges.rounds import RoundCallback, run_rounds
from fedcausal.fedges.site import Site
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
from fedcausal.fusion import Consensus

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from fedcausal.fedges.convergence import Convergence
    from fedcausal.flower.loaders import PartitionLoader
    from fedcausal.fusion import FusionOperator
    from fedcausal.graph import DAG

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

SiteFactory = Callable[["DiscreteDataset"], Site]
"""Builds a node's :class:`~fedcausal.fedges.Site` from its data."""

# A node's Site lives across messages so that its score cache does; the model
# itself is always restored from the node's context, which Flower persists. A
# loader that reads a file every round returns a new dataset each time, so the
# site is kept whenever the data is the same, not only the same object.
_SITES: dict[tuple[int, int], Site] = {}


def _same_data(a: object, b: DiscreteDataset) -> bool:
    return a is b or (
        isinstance(a, DiscreteDataset)
        and a.names == b.names
        and a.categories == b.categories
        and a.values.shape == b.values.shape
        and bool(np.array_equal(a.values, b.values))
    )


def handle_train_message(
    msg: Message,
    context: Context,
    loader: PartitionLoader[DiscreteDataset],
    make_site: SiteFactory = Site,
) -> Message:
    """Handle one ``train`` message, a site's whole part of a round, and reply.

    Parameters
    ----------
    msg : Message
        The incoming message; its control record names the phase.
    context : Context
        The node's context. The site's model persists in ``context.state``.
    loader : PartitionLoader
        Resolves the node's data.
    make_site : callable, default=Site
        Builds the node's site from its data, e.g.
        ``functools.partial(Site, refinement="both")``.

    Returns
    -------
    Message
        The site's DAG, its score and its schema.

    Raises
    ------
    ClientFailureError
        On an unknown phase, or a fuse message before any build.
    """
    data = loader(context)
    schema = schema_fingerprint(data)
    key = (int(context.node_id), int(context.node_config.get("partition-id", -1)))
    site = _SITES.get(key)
    if site is None or not _same_data(site.data, data):
        site = _SITES[key] = make_site(data)
    if _LOCAL_STATE_KEY in context.state.array_records:
        dag = record_to_dag(cast(ArrayRecord, context.state[_LOCAL_STATE_KEY]), data.n_vars)
        site.model = BNModel(dag, site.score.score_dag(dag))
    else:
        site.model = None

    phase = str(msg.content[CTRL_KEY]["phase"])
    if phase == "build":
        model = site.learn()
    elif phase == "fuse":
        if site.model is None:
            raise ClientFailureError("received a fuse message before any build round")
        global_dag = record_to_dag(cast(ArrayRecord, msg.content[_GLOBAL_KEY]), data.n_vars)
        model = site.merge(global_dag)
    else:
        raise ClientFailureError(f"unknown phase {phase!r}")

    context.state[_LOCAL_STATE_KEY] = dag_to_record(model.dag)
    reply = RecordDict(
        {
            _LOCAL_KEY: dag_to_record(model.dag),
            CTRL_KEY: control_config(
                node_index=int(context.node_config.get("partition-id", -1)),
                score=model.score,
                n_vars=data.n_vars,
                schema=schema,
            ),
        }
    )
    return Message(content=reply, reply_to=msg)


def build_client_app(
    loader: PartitionLoader[DiscreteDataset], make_site: SiteFactory = Site
) -> ClientApp:
    """Register :func:`handle_train_message` as a ``ClientApp``."""
    app = ClientApp()

    @app.train()
    def train(msg: Message, context: Context) -> Message:
        return handle_train_message(msg, context, loader, make_site)

    return app


def run_server_loop(
    grid: Grid,
    context: Context,
    *,
    n_clients: int,
    n_vars: int,
    n_rounds: int = 10,
    global_fusion: FusionOperator | None = None,
    convergence: Convergence | None = None,
    min_clients: int | None = None,
    on_model: RoundCallback | None = None,
) -> DAG | None:
    """Run the FedGES rounds over `grid` and return the final global DAG.

    Parameters
    ----------
    grid : Grid
        Anything that can send and receive Flower messages.
    context : Context
        The server's context. The final global DAG is also stored on
        ``context.state["global_dag"]``.
    n_clients : int
        Supernodes to wait for before the first round.
    n_vars : int
        Number of variables.
    n_rounds : int, default=10
        Most rounds to run.
    global_fusion : FusionOperator, optional
        Default ``Consensus(50)``.
    convergence : Convergence, optional
        Default: no score improvement, or every model repeated.
    min_clients : int, optional
        Usable replies a round needs. Default `n_clients`, so a failed site
        stops the run; lower it to tolerate dropouts.
    on_model : callable, optional
        Receives every local and global model; see
        :data:`~fedcausal.fedges.rounds.RoundCallback`.

    Returns
    -------
    DAG or None
        ``None`` if the rounds stopped before the first fusion.

    Raises
    ------
    ConfigurationError
        If `min_clients` is not between 1 and `n_clients`.
    ClientFailureError
        If too few supernodes register, or too few reply in a round.
    SchemaMismatchError
        If the sites disagree on their variables.
    """
    required = required_clients(n_clients, min_clients)
    node_ids = await_nodes(grid, n_clients)
    schema: list[str] = []

    def learn() -> list[BNModel]:
        replies = grid.send_and_receive(
            broadcast(node_ids, RecordDict({CTRL_KEY: control_config(phase="build")}))
        )
        models, round_schema = _collect_models(replies, n_vars, required)
        if schema and round_schema != schema[0]:
            raise SchemaMismatchError("a site's schema changed between rounds")
        schema[:] = [round_schema]
        return models

    def merge(global_dag: DAG) -> None:
        content = RecordDict(
            {CTRL_KEY: control_config(phase="fuse"), _GLOBAL_KEY: dag_to_record(global_dag)}
        )
        usable_replies(grid.send_and_receive(broadcast(node_ids, content)), required, "fuse")

    global_dag, _ = run_rounds(
        learn,
        merge,
        global_fusion if global_fusion is not None else Consensus(50),
        convergence if convergence is not None else default_convergence(),
        n_rounds,
        on_model,
    )
    if global_dag is not None:
        context.state[_GLOBAL_STATE_KEY] = dag_to_record(global_dag)
    return global_dag


def build_server_app(
    *,
    n_clients: int,
    n_vars: int,
    n_rounds: int = 10,
    global_fusion: FusionOperator | None = None,
    convergence: Convergence | None = None,
    min_clients: int | None = None,
    on_model: RoundCallback | None = None,
) -> ServerApp:
    """Register :func:`run_server_loop` as a ``ServerApp``; takes the same arguments."""
    app = ServerApp()

    @app.main()
    def main(grid: Grid, context: Context) -> None:
        run_server_loop(
            grid,
            context,
            n_clients=n_clients,
            n_vars=n_vars,
            n_rounds=n_rounds,
            global_fusion=global_fusion,
            convergence=convergence,
            min_clients=min_clients,
            on_model=on_model,
        )

    return app


def run_flower_fedges(
    sites: Sequence[DiscreteDataset],
    *,
    n_rounds: int = 10,
    global_fusion: FusionOperator | None = None,
    convergence: Convergence | None = None,
    make_site: SiteFactory = Site,
    on_model: RoundCallback | None = None,
) -> DAG | None:
    """Run FedGES on Flower's simulation engine and return the final global DAG.

    The Flower counterpart of :class:`fedcausal.FedGES`. Needs
    ``fedcausal[flower-simulation]``; a deployment across machines does not,
    see the deployment guide.

    Parameters
    ----------
    sites : sequence of DiscreteDataset
        One dataset per supernode, sharing their variables and levels.
    n_rounds, global_fusion, convergence, on_model
        As in :func:`run_server_loop`.
    make_site : callable, default=Site
        Builds each node's site from its data.

    Returns
    -------
    DAG or None
        ``None`` if the rounds stopped before the first fusion.
    """
    from fedcausal.flower.loaders import InMemoryPartitions

    run_simulation = simulation_runner()
    globals_: list[DAG] = []

    def collect(stage: str, round_: int, site: int, model: BNModel) -> None:
        if stage == "global":
            globals_.append(model.dag)
        if on_model is not None:
            on_model(stage, round_, site, model)

    client_app = build_client_app(InMemoryPartitions(list(sites)), make_site)
    server_app = build_server_app(
        n_clients=len(sites),
        n_vars=sites[0].n_vars,
        n_rounds=n_rounds,
        global_fusion=global_fusion,
        convergence=convergence,
        on_model=collect,
    )
    run_simulation(server_app=server_app, client_app=client_app, num_supernodes=len(sites))
    return globals_[-1] if globals_ else None


def _collect_models(
    replies: Iterable[Message], n_vars: int, required: int
) -> tuple[list[BNModel], str]:
    """Rebuild the sites' models, ordered by partition id and then node id.

    In simulation that is the order the in-process loop uses; in a deployment,
    where no partition id is set, it is at least the same every round.

    Raises
    ------
    ClientFailureError
        If too few sites replied successfully.
    SchemaMismatchError
        If the sites disagree on their variables.
    """
    good = usable_replies(replies, required, "build")
    schema = check_schemas(good)
    keyed = []
    for reply in good:
        dag = record_to_dag(cast(ArrayRecord, reply.content[_LOCAL_KEY]), n_vars)
        key = (ctrl_int(reply, "node_index"), reply.metadata.src_node_id)
        keyed.append((key, BNModel(dag, ctrl_float(reply, "score"))))
    return [model for _, model in sorted(keyed, key=lambda item: item[0])], schema
