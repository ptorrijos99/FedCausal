"""Stand in for the part of Flower's runtime the transport tests bypass."""

from __future__ import annotations

from collections.abc import Iterator

import pytest


@pytest.fixture(autouse=True)
def _task_identity() -> Iterator[None]:
    """Record the task identity Flower's runtime would.

    Since flwr 1.38 a ``Message`` carries the run and node of the task that builds
    it, which the runtime records before any app code runs. These tests call the
    handlers directly, outside the runtime. Older releases need nothing.
    """
    try:
        from flwr.supercore.task_identity import TaskIdentity
    except ImportError:
        yield
        return

    TaskIdentity.task_id = 1
    TaskIdentity.run_id = 1
    TaskIdentity.node_id = 1
    yield
    TaskIdentity._task_id = TaskIdentity._run_id = TaskIdentity._node_id = None
