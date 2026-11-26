"""The modules that moved to causalbase stay importable under their old names.

Each old path is an alias of the causalbase module, not a copy: the classes are
the same objects, so ``isinstance`` and ``except`` work across both names. The
first import of an old path warns with the path to use instead.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

MOVED = [
    ("fedcausal.graph", "causalbase.graph"),
    ("fedcausal.graph.cpdag", "causalbase.graph.cpdag"),
    ("fedcausal.scores", "causalbase.scores"),
    ("fedcausal.scores.bdeu", "causalbase.scores.bdeu"),
    ("fedcausal.search", "causalbase.search"),
    ("fedcausal.metrics", "causalbase.metrics"),
    ("fedcausal.metrics.structural", "causalbase.metrics.structural"),
    ("fedcausal.io", "causalbase.io"),
    ("fedcausal.io.sampling", "causalbase.io.sampling"),
    ("fedcausal.datasets", "causalbase.datasets"),
    ("fedcausal.data.dataset", "causalbase.data.dataset"),
    ("fedcausal.data.continuous", "causalbase.data.continuous"),
]


def _run(code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-W", "error::DeprecationWarning", "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        timeout=600,
    )


@pytest.mark.parametrize(("old", "new"), MOVED)
def test_an_old_path_is_the_causalbase_module_and_warns(old: str, new: str) -> None:
    result = _run(
        f"""
        import importlib, warnings
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            old = importlib.import_module({old!r})
        new = importlib.import_module({new!r})
        assert old is new, (old, new)
        messages = [str(w.message) for w in caught if issubclass(w.category, DeprecationWarning)]
        assert any({new!r} in m for m in messages), messages
        """
    )
    assert result.returncode == 0, result.stderr


def test_old_names_are_the_same_objects() -> None:
    result = _run(
        """
        import warnings
        warnings.simplefilter("ignore", DeprecationWarning)
        import causalbase.data, causalbase.graph, causalbase.search, causalbase.exceptions
        from fedcausal import GES, XGES, DAG
        from fedcausal.data import DiscreteDataset, ContinuousDataset, as_discrete_sites
        from fedcausal.graph import DAG as DAG2
        from fedcausal.exceptions import FedCausalError, CycleError, NotFittedError
        assert GES is causalbase.search.GES and XGES is causalbase.search.XGES
        assert DAG is DAG2 is causalbase.graph.DAG
        assert DiscreteDataset is causalbase.data.DiscreteDataset
        assert ContinuousDataset is causalbase.data.ContinuousDataset
        assert as_discrete_sites is causalbase.data.as_discrete_sites
        assert FedCausalError is causalbase.exceptions.CausalBaseError
        assert CycleError is causalbase.exceptions.CycleError
        assert NotFittedError is causalbase.exceptions.NotFittedError
        """
    )
    assert result.returncode == 0, result.stderr


def test_the_federated_api_imports_without_warnings() -> None:
    """fedcausal's own code reaches the core through causalbase, never through an alias."""
    result = _run(
        """
        from fedcausal import FedGES, FedRCD
        from fedcausal.data import contiguous_split, dirichlet_split
        from fedcausal.exceptions import SchemaMismatchError, ClientFailureError
        import fedcausal.fedges, fedcausal.fusion, fedcausal.lingam
        """
    )
    assert result.returncode == 0, result.stderr
