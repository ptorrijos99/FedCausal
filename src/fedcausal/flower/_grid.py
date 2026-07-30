"""Message plumbing every fedcausal transport needs.

Waiting for supernodes to register, broadcasting one message to all of them,
collecting the replies without swallowing a failure, and reading control scalars
back out. None of it knows what the messages carry, which is why FedGES and the
FedRCD family share it as-is.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, cast

from flwr.common import Message, MessageType

from fedcausal.exceptions import ClientFailureError, ConfigurationError, SchemaMismatchError

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

    from flwr.common import RecordDict
    from flwr.server import Grid

__all__ = [
    "await_nodes",
    "broadcast",
    "check_schemas",
    "ctrl_float",
    "ctrl_int",
    "ctrl_str",
    "required_clients",
    "simulation_runner",
    "usable_replies",
]

CTRL_KEY = "control"


def required_clients(n_clients: int, min_clients: int | None) -> int:
    """Return how many replies a round needs, defaulting to all of them.

    Raises
    ------
    ConfigurationError
        If `min_clients` is not between 1 and `n_clients`.
    """
    required = n_clients if min_clients is None else min_clients
    if not 1 <= required <= n_clients:
        raise ConfigurationError(
            f"min_clients must be between 1 and n_clients={n_clients}, got {min_clients}"
        )
    return required


def await_nodes(grid: Grid, n_clients: int, tries: int = 60, delay: float = 0.3) -> list[int]:
    """Poll until at least `n_clients` supernodes have registered.

    Raises
    ------
    ClientFailureError
        If too few register within ``tries * delay`` seconds.
    """
    node_ids: list[int] = []
    for _ in range(tries):
        node_ids = list(grid.get_node_ids())
        if len(node_ids) >= n_clients:
            return sorted(node_ids)
        time.sleep(delay)
    raise ClientFailureError(
        f"only {len(node_ids)} of {n_clients} supernodes registered within {tries * delay:.0f}s"
    )


def broadcast(node_ids: Sequence[int], content: RecordDict) -> list[Message]:
    """Build one identical ``train`` message per node."""
    return [
        Message(content=content, dst_node_id=nid, message_type=MessageType.TRAIN, group_id="0")
        for nid in node_ids
    ]


def usable_replies(replies: Iterable[Message], required: int, phase: str) -> list[Message]:
    """Return the successful replies, or raise if too few came back.

    A failure is never dropped quietly. A client that errors out shrinks what
    the server aggregates, and a run that silently learns from four sites
    instead of five gives you a number you cannot interpret.

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


def check_schemas(replies: Iterable[Message]) -> str:
    """Return the fingerprint every client reported, or raise if they differ.

    Everything on the wire is indexed by column position, so this is the only
    place a mismatch can be caught before the server produces something
    meaningless from it.

    Raises
    ------
    SchemaMismatchError
        If more than one distinct fingerprint comes back.
    """
    replies = list(replies)
    schemas = {ctrl_str(reply, "schema") for reply in replies}
    if len(schemas) > 1:
        sizes = sorted({ctrl_int(reply, "n_vars") for reply in replies})
        raise SchemaMismatchError(
            f"clients disagree on their variables ({len(schemas)} distinct schemas, "
            f"variable counts {sizes}). Graphs and statistics travel as bare column "
            "indices, so every client must hold the same variables in the same order."
        )
    return schemas.pop()


def ctrl_str(reply: Message, key: str) -> str:
    """Read a string out of a reply's control record."""
    return str(reply.content[CTRL_KEY][key])


def ctrl_int(reply: Message, key: str) -> int:
    """Read an integer out of a reply's control record."""
    return int(cast(int, reply.content[CTRL_KEY][key]))


def ctrl_float(reply: Message, key: str) -> float:
    """Read a float out of a reply's control record."""
    return float(cast(float, reply.content[CTRL_KEY][key]))


def simulation_runner() -> Callable[..., Any]:
    """Return ``flwr.simulation.run_simulation``.

    Raises
    ------
    ConfigurationError
        If Flower's simulation extra is not installed.
    """
    try:
        from flwr.simulation import run_simulation
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ConfigurationError(
            "Flower's simulation engine is not installed. Install it with "
            "`pip install 'fedcausal[flower-simulation]'`, or deploy against a "
            "real SuperLink instead; see docs/deployment.md."
        ) from exc
    return cast("Callable[..., Any]", run_simulation)
