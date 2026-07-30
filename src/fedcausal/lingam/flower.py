"""Flower ``ClientApp`` / ``ServerApp`` running the FedRCD family.

``flwr`` is an optional dependency: importing this module requires it, exactly
as :mod:`fedcausal.flower` requires it for the FedGES transport. The rest of
:mod:`fedcausal.lingam` never imports this module, so the subpackage works
without ``flwr`` installed; ``pip install fedcausal[flower]`` adds the
transport.

Where FedGES's round loop is several build/fuse round-trips (see
:mod:`fedcausal.flower.app`), the one-shot solvers here (``fedrcd``,
``fedrcd_noniterative``, ``ishc_beta``, ``fedishc``, ``fedhc``) need exactly
one round-trip:

1. The server broadcasts a ``train`` message to every supernode.
2. Each client computes its fourth-order raw moment sums locally (raw data
   never leaves the client) and replies with those sums, its row count, and a
   fingerprint of its variable names.
3. The server checks every reply's fingerprint agrees, aggregates the moment
   sums into unbiased cumulants, and runs the selected solver.

``fedrcd_exact`` is the multi-round member of the family: ``p - 1`` rounds,
each one identifying a source, deflating every client's local data by it, and
recomputing fresh statistics from the residuals. Its transport lives in the
same "handler + build_*_app + run_*_server_loop" shape as the one-shot
protocol above, but is stateful -- each client keeps its progressively
deflated working matrix in ``context.state`` between rounds, under the
``_WORKING_STATE_KEY`` :class:`~flwr.common.ArrayRecord`, exactly as
:mod:`fedcausal.flower.app` persists the local DAG under ``_LOCAL_STATE_KEY``.

The server logic for both protocols is split in two layers, mirroring
:func:`fedcausal.flower.app.run_server_loop` / :func:`~fedcausal.flower.app.build_server_app`:

- :func:`run_rcd_server_loop` / :func:`run_exact_server_loop` take a ``Grid``
  directly, so anything that can send and receive messages can drive them --
  including an in-process stub grid in tests, with no Ray anywhere. This is
  where the correctness lives, and it is what the transport-equivalence tests
  exercise.
- :func:`run_flower_rcd` / :func:`run_flower_rcd_exact` are the public
  convenience entry points: they partition a dataset, spin up real supernodes
  through Flower's Ray-based simulation engine
  (:mod:`fedcausal.flower.simulation` uses the same engine for FedGES), and
  hand back the final :class:`~fedcausal.lingam.rcd.RCDResult`.

Client apps load their :class:`~fedcausal.data.ContinuousDataset` through a
:class:`ContinuousLoader`, exactly as :mod:`fedcausal.flower.app` loads a
:class:`~fedcausal.data.DiscreteDataset` through a
:class:`~fedcausal.flower.data.PartitionLoader`. Two implementations ship:
:class:`InMemoryContinuousPartitions` for simulation, and
:class:`NodeConfigContinuousLoader` for a real deployment, where each
supernode reads a CSV named by its own ``node_config`` and nothing else; see
``docs/fedrcd.md`` and ``examples/fedrcd_deployment/``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast, runtime_checkable

import numpy as np

try:
    import flwr as _flwr  # noqa: F401
except ModuleNotFoundError as exc:  # pragma: no cover - exercised only without flwr
    raise ModuleNotFoundError(
        "fedcausal.lingam.flower requires the optional 'flwr' dependency. "
        "Install it with: pip install fedcausal[flower]"
    ) from exc

from flwr.client import ClientApp
from flwr.common import ArrayRecord, Context, Message, MessageType, RecordDict
from flwr.server import Grid, ServerApp

from fedcausal.data import ContinuousDataset
from fedcausal.exceptions import ClientFailureError, ConfigurationError, SchemaMismatchError
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
    from collections.abc import Callable, Iterable, Sequence

    from fedcausal.lingam.cumulants import Cumulants

__all__ = [
    "ContinuousLoader",
    "InMemoryContinuousPartitions",
    "NodeConfigContinuousLoader",
    "build_exact_client_app",
    "build_exact_server_app",
    "build_rcd_client_app",
    "build_rcd_server_app",
    "handle_deflate_message",
    "handle_stats_message",
    "run_exact_server_loop",
    "run_flower_rcd",
    "run_flower_rcd_exact",
    "run_rcd_server_loop",
]

_STATS_KEY = "stats"
_CTRL_KEY = "control"
_BETAS_KEY = "betas"
_WORKING_STATE_KEY = "rcd_working"

#: The one-shot solvers this transport can drive. ``fedrcd_exact`` is
#: multi-round and out of scope for this module.
_SOLVERS: dict[str, Callable[..., RCDResult]] = {
    "fedrcd": fedrcd,
    "fedrcd_noniterative": fedrcd_noniterative,
    "ishc_beta": ishc_beta,
    "fedishc": fedishc,
    "fedhc": fedhc,
}


@runtime_checkable
class ContinuousLoader(Protocol):
    """Resolves a supernode's :class:`~fedcausal.data.ContinuousDataset`.

    The continuous counterpart of :class:`fedcausal.flower.data.PartitionLoader`.
    """

    def __call__(self, context: Context) -> ContinuousDataset:
        """Return the dataset this supernode should compute statistics from."""
        ...


class NodeConfigContinuousLoader:
    """Load a supernode's own CSV, named by a key in its ``node_config``.

    The continuous counterpart of :class:`fedcausal.flower.data.NodeConfigLoader`:
    the deployment path for the FedRCD family. Start each supernode with
    ``--node-config "data-path='/path/to/my/shard.csv'"`` and it reads that file
    and nothing else.

    Parameters
    ----------
    key : str, optional
        Which ``node_config`` key holds the path (default ``"data-path"``).
    """

    def __init__(self, key: str = "data-path") -> None:
        self.key = key

    def __call__(self, context: Context) -> ContinuousDataset:
        """Load and return this supernode's dataset.

        Raises
        ------
        ConfigurationError
            If the key is missing from ``node_config`` or names a path that
            does not exist.
        """
        raw = context.node_config.get(self.key)
        if raw is None:
            raise ConfigurationError(
                f"node_config has no {self.key!r}; start the supernode with "
                f"--node-config \"{self.key}='/path/to/shard.csv'\""
            )
        path = Path(str(raw))
        if not path.exists():
            raise ConfigurationError(f"{self.key}={path} does not exist on this supernode")
        return ContinuousDataset.from_csv(path)


def handle_stats_message(msg: Message, context: Context, loader: ContinuousLoader) -> Message:
    """Handle one ``train`` message and reply with this client's moment sums.

    The whole client side of the one-shot protocol, with no Flower app
    machinery around it, so it can be driven by :func:`build_rcd_client_app`
    or directly by a test.

    Parameters
    ----------
    msg : Message
        The incoming message. Carries no model: the one-shot protocol asks each
        client exactly once for its statistics.
    context : Context
        This supernode's context. Nothing persists between rounds here -- the
        one-shot protocol is stateless on the client.
    loader : ContinuousLoader
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
            _CTRL_KEY: control_config(
                n=int(stats["n"]),
                n_vars=data.n_vars,
                schema=names_fingerprint(data.names),
            ),
        }
    )
    return Message(reply, reply_to=msg)


def build_rcd_client_app(loader: ContinuousLoader) -> ClientApp:
    """Build a ``ClientApp`` running the one-shot FedRCD client logic.

    A thin registration wrapper around :func:`handle_stats_message`; see there
    for the parameters.

    Returns
    -------
    ClientApp
        The configured client application.
    """
    app = ClientApp()

    @app.train()
    def train(msg: Message, context: Context) -> Message:
        return handle_stats_message(msg, context, loader)

    return app


def run_rcd_server_loop(
    grid: Grid,
    n_clients: int,
    solver: str = "fedrcd",
    *,
    min_clients: int | None = None,
    **solver_kwargs: Any,
) -> RCDResult:
    """Run the one-shot FedRCD protocol over `grid` and return the result.

    The whole server side, with no Flower app machinery around it, so an app
    that reads its settings from ``run_config`` can call it directly, and so a
    test can drive it over an in-process stub grid without Flower's Ray-based
    simulation engine anywhere. See :func:`build_rcd_server_app` for the
    parameters this wraps.

    Parameters
    ----------
    grid : Grid
        Anything that can send and receive Flower messages: a real
        :class:`flwr.server.Grid`, or an in-process stand-in for testing.
    n_clients : int
        Number of supernodes to wait for before broadcasting.
    solver : {"fedrcd", "fedrcd_noniterative", "ishc_beta", "fedishc", "fedhc"}, optional
        Which one-shot solver to run on the aggregated statistics.
    min_clients : int or None, optional
        How many usable replies the round needs. Defaults to `n_clients`, so a
        failed client aborts the run rather than silently shrinking the
        ensemble. Lower it to tolerate dropouts.
    **solver_kwargs : Any
        Passed through to the selected solver (``use_lasso``, ``lmbda_type``).

    Returns
    -------
    RCDResult
        The causal order and coefficient matrix the selected solver estimated
        from the aggregated statistics.

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
    required = n_clients if min_clients is None else min_clients
    if not 1 <= required <= n_clients:
        raise ConfigurationError(
            f"min_clients must be between 1 and n_clients={n_clients}, got {min_clients}"
        )

    node_ids = _await_nodes(grid, n_clients)
    replies = grid.send_and_receive(
        _broadcast(node_ids, RecordDict({_CTRL_KEY: control_config(phase="stats")}))
    )
    good = _check_replies(replies, required)
    _check_schemas(good)

    aggregated = _aggregate_replies(good)
    return solver_fn(aggregated, **solver_kwargs)


def build_rcd_server_app(
    n_clients: int,
    solver: str = "fedrcd",
    *,
    min_clients: int | None = None,
    results: list[RCDResult] | None = None,
    **solver_kwargs: Any,
) -> ServerApp:
    """Build a ``ServerApp`` driving the one-shot FedRCD protocol.

    Parameters
    ----------
    n_clients : int
        Number of supernodes to wait for before broadcasting.
    solver : {"fedrcd", "fedrcd_noniterative", "ishc_beta", "fedishc", "fedhc"}, optional
        Which one-shot solver to run on the aggregated statistics.
    min_clients : int or None, optional
        How many usable replies the round needs; see :func:`run_rcd_server_loop`.
    results : list[RCDResult] or None, optional
        If given, the final `RCDResult` is appended to it -- the side channel
        :func:`run_flower_rcd` uses to retrieve the result once
        ``run_simulation`` returns, since a ``ServerApp``'s ``main`` has no
        return value of its own.
    **solver_kwargs : Any
        Passed through to the selected solver (``use_lasso``, ``lmbda_type``).

    Returns
    -------
    ServerApp
        The configured server application.
    """
    app = ServerApp()

    @app.main()
    def main(grid: Grid, context: Context) -> None:
        result = run_rcd_server_loop(
            grid, n_clients, solver, min_clients=min_clients, **solver_kwargs
        )
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

    Mirrors :func:`fedcausal.flower.simulation.run_flower_fedges` for the
    one-shot LiNGAM family: splits `data` into `n_clients` contiguous
    partitions and drives them through real supernodes via Flower's Ray-based
    simulation engine. For the same protocol driven without Ray -- the
    in-process path used to prove the transport does not change the answer --
    see :func:`run_rcd_server_loop`.

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
        The causal order and coefficient matrix the selected solver estimated.

    Raises
    ------
    ConfigurationError
        If Flower's simulation extra is not installed, or `solver` is not one
        of the supported names.
    ClientFailureError
        If too few supernodes register, or too few reply.
    SchemaMismatchError
        If the clients do not agree on their variables.
    """
    try:
        from flwr.simulation import run_simulation
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ConfigurationError(
            "Flower's simulation engine is not installed. Install it with "
            "`pip install 'fedcausal[flower-simulation]'`, or deploy against a "
            "real SuperLink instead; see docs/flower-deployment.md."
        ) from exc

    from fedcausal.data import contiguous_split

    parts = contiguous_split(data, n_clients)
    results: list[RCDResult] = []
    client_app = build_rcd_client_app(InMemoryContinuousPartitions(parts))
    server_app = build_rcd_server_app(n_clients, solver, results=results, **solver_kwargs)
    run_simulation(server_app=server_app, client_app=client_app, num_supernodes=n_clients)

    if not results:
        raise ClientFailureError("the Flower simulation completed without producing a result")
    return results[0]


def handle_deflate_message(msg: Message, context: Context, loader: ContinuousLoader) -> Message:
    """Handle one round of the multi-round FedRCD-Exact protocol and reply.

    Mirrors ``FedRDCExactClient.fit`` from the legacy implementation. On round
    0 the working matrix is seeded from `loader` untouched. On every later
    round, the message carries the source identified last round and the OLS
    coefficients for it; this handler first deflates the client's *persisted*
    working matrix by them (``x_j := x_j - beta_j * x_s`` for every column but
    the source), drops the source column, and only then computes and replies
    with fresh statistics -- so no algebraic error accumulates across rounds,
    only the residual data itself changes.

    Parameters
    ----------
    msg : Message
        The incoming message. Its control record carries ``round``, and from
        round 1 onward also ``src_local``; from round 1 onward the message
        also carries a ``betas`` :class:`~flwr.common.ArrayRecord`.
    context : Context
        This supernode's context. The progressively deflated working matrix
        persists in ``context.state`` under a module-private key between
        rounds -- this protocol is stateful on the client, unlike the
        one-shot family.
    loader : ContinuousLoader
        Resolves this supernode's dataset. Only consulted on round 0.

    Returns
    -------
    Message
        A reply carrying the nine moment-sum arrays computed from the current
        working matrix and the local row count. Round 0's reply also carries a
        fingerprint of the client's original variable names, so the server can
        check every client started from the same schema; later rounds carry no
        fingerprint, since the working matrix no longer corresponds to named
        variables.
    """
    control = msg.content[_CTRL_KEY]
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
    reply = RecordDict({_STATS_KEY: stats_to_record(stats), _CTRL_KEY: control_config(**ctrl)})
    return Message(reply, reply_to=msg)


def build_exact_client_app(loader: ContinuousLoader) -> ClientApp:
    """Build a ``ClientApp`` running the multi-round FedRCD-Exact client logic.

    A thin registration wrapper around :func:`handle_deflate_message`; see
    there for the parameters.

    Returns
    -------
    ClientApp
        The configured client application.
    """
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

    The whole server side, with no Flower app machinery around it -- see
    :func:`run_rcd_server_loop` for why that shape matters. Mirrors
    :func:`fedcausal.lingam.rcd.fedrcd_exact`'s server half step for step:
    round 0's aggregate gives the *original* covariance and sample count, kept
    aside for the final :func:`~fedcausal.lingam.solvers.estimate_b_from_cov`
    call; each subsequent step builds the fourth-order tau matrix from the
    current aggregate, takes the ``argmin`` of its row sums as the source,
    computes OLS coefficients from the current covariance, and -- for every
    step but the last -- broadcasts them so clients deflate and reply with
    fresh statistics for the next step. ``active_global`` tracks which
    original variable each surviving local column corresponds to, so the
    causal order returned is in original variable indices throughout.

    Parameters
    ----------
    grid : Grid
        Anything that can send and receive Flower messages: a real
        :class:`flwr.server.Grid`, or an in-process stand-in for testing.
    n_clients : int
        Number of supernodes to wait for before broadcasting.
    min_clients : int or None, optional
        How many usable replies each round needs. Defaults to `n_clients`, so
        a failed client aborts the run rather than silently shrinking the
        ensemble. Lower it to tolerate dropouts.
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
    required = n_clients if min_clients is None else min_clients
    if not 1 <= required <= n_clients:
        raise ConfigurationError(
            f"min_clients must be between 1 and n_clients={n_clients}, got {min_clients}"
        )

    node_ids = _await_nodes(grid, n_clients)

    replies = grid.send_and_receive(
        _broadcast(node_ids, RecordDict({_CTRL_KEY: control_config(round=0)}))
    )
    good = _check_replies(replies, required)
    _check_schemas(good)

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
                _broadcast(
                    node_ids,
                    RecordDict(
                        {
                            _CTRL_KEY: control_config(round=step + 1, src_local=src_local),
                            _BETAS_KEY: ArrayRecord([betas]),
                        }
                    ),
                )
            )
            good = _check_replies(round_replies, required)
            agg = _aggregate_replies(good)

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
    """Build a ``ServerApp`` driving the multi-round FedRCD-Exact protocol.

    Parameters
    ----------
    n_clients : int
        Number of supernodes to wait for before broadcasting.
    min_clients : int or None, optional
        How many usable replies each round needs; see :func:`run_exact_server_loop`.
    results : list[RCDResult] or None, optional
        If given, the final `RCDResult` is appended to it -- the side channel
        :func:`run_flower_rcd_exact` uses to retrieve the result once
        ``run_simulation`` returns, since a ``ServerApp``'s ``main`` has no
        return value of its own.
    **solver_kwargs : Any
        Passed through to :func:`run_exact_server_loop` (``use_lasso``, ``lmbda_type``).

    Returns
    -------
    ServerApp
        The configured server application.
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

    Mirrors :func:`run_flower_rcd` for the multi-round protocol: splits `data`
    into `n_clients` contiguous partitions and drives them through real
    supernodes via Flower's Ray-based simulation engine, running ``p - 1``
    rounds of communication. For the same protocol driven without Ray -- the
    in-process path used to prove the transport does not change the answer --
    see :func:`run_exact_server_loop`.

    Parameters
    ----------
    data : ContinuousDataset
        Full dataset to partition across clients.
    n_clients : int
        Number of clients and supernodes.
    **solver_kwargs : Any
        Passed through to :func:`run_exact_server_loop` (``use_lasso``, ``lmbda_type``).

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
    try:
        from flwr.simulation import run_simulation
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ConfigurationError(
            "Flower's simulation engine is not installed. Install it with "
            "`pip install 'fedcausal[flower-simulation]'`, or deploy against a "
            "real SuperLink instead; see docs/flower-deployment.md."
        ) from exc

    from fedcausal.data import contiguous_split

    parts = contiguous_split(data, n_clients)
    results: list[RCDResult] = []
    client_app = build_exact_client_app(InMemoryContinuousPartitions(parts))
    server_app = build_exact_server_app(n_clients, results=results, **solver_kwargs)
    run_simulation(server_app=server_app, client_app=client_app, num_supernodes=n_clients)

    if not results:
        raise ClientFailureError("the Flower simulation completed without producing a result")
    return results[0]


class InMemoryContinuousPartitions:
    """Serve one of several in-memory `ContinuousDataset`s, indexed by ``partition-id``.

    Simulation-only continuous analogue of
    :class:`fedcausal.flower.data.InMemoryPartitions`. Used by
    :func:`run_flower_rcd` and :func:`run_flower_rcd_exact`, and public so a
    caller building its own simulation ``ClientApp`` can use it directly.

    Parameters
    ----------
    partitions : Sequence[ContinuousDataset]
        One dataset per partition id.

    Raises
    ------
    ConfigurationError
        If `partitions` is empty.
    """

    def __init__(self, partitions: Sequence[ContinuousDataset]) -> None:
        if not partitions:
            raise ConfigurationError("InMemoryContinuousPartitions needs at least one partition")
        self._partitions = list(partitions)

    def __call__(self, context: Context) -> ContinuousDataset:
        """Return the partition matching this supernode's ``partition-id``.

        Raises
        ------
        ConfigurationError
            If ``partition-id`` is absent from ``node_config``, or out of
            range for the partitions held.
        """
        if "partition-id" not in context.node_config:
            raise ConfigurationError(
                "node_config has no 'partition-id'; InMemoryContinuousPartitions only "
                "works in simulation. Use NodeConfigContinuousLoader for a real deployment."
            )
        pid = int(context.node_config["partition-id"])
        if not 0 <= pid < len(self._partitions):
            raise ConfigurationError(
                f"partition-id {pid} is out of range for {len(self._partitions)} partitions"
            )
        return self._partitions[pid]


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


def _check_replies(replies: Iterable[Message], required: int) -> list[Message]:
    """Return the successful replies, or raise if too few came back.

    Failures are never dropped quietly: a client that errors out shrinks the
    statistics the aggregation sees, and a result silently computed from four
    sites instead of five is worse than a run that stops.

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
            f"stats round got {len(good)} usable replies, needs {required} ({detail})"
        )
    return good


def _check_schemas(replies: Iterable[Message]) -> None:
    """Raise if the clients disagree on their variable-name fingerprint.

    Statistics are aggregated as bare arrays with no variable names attached,
    so this is the only place a mismatch can be caught before the aggregation
    silently produces a meaningless result.

    Raises
    ------
    SchemaMismatchError
        If more than one distinct fingerprint is reported.
    """
    schemas = {_ctrl_str(reply, "schema") for reply in replies}
    if len(schemas) > 1:
        sizes = sorted({_ctrl_int(reply, "n_vars") for reply in replies})
        raise SchemaMismatchError(
            f"clients disagree on their variables ({len(schemas)} distinct schemas, "
            f"variable counts {sizes}). Statistics are aggregated as bare arrays, so "
            "every client must hold the same variables in the same order."
        )


def _aggregate_replies(replies: Iterable[Message]) -> Cumulants:
    """Unpack a batch of stats replies and aggregate them into global cumulants."""
    stats_list = [
        record_to_stats(cast(ArrayRecord, reply.content[_STATS_KEY]), _ctrl_int(reply, "n"))
        for reply in replies
    ]
    return aggregate_and_compute_cumulants(stats_list)


def _ctrl_str(reply: Message, key: str) -> str:
    """Read a string out of a reply's control record."""
    return str(reply.content[_CTRL_KEY][key])


def _ctrl_int(reply: Message, key: str) -> int:
    """Read an integer out of a reply's control record."""
    return int(cast(int, reply.content[_CTRL_KEY][key]))
