"""fedcausal is the federated layer over causalbase, and only its transport needs Flower.

Its code reaches the core of structure learning (data, graphs, scores, search,
metrics, io, datasets) through causalbase, never through the old
``fedcausal.<core>`` paths, which are kept only as deprecated aliases. The
method packages (fedges, fusion, lingam) never import Flower.
"""

from __future__ import annotations

import ast
import subprocess
import sys
import textwrap
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src" / "fedcausal"
MOVED = [
    "graph",
    "scores",
    "search",
    "metrics",
    "io",
    "datasets",
    "data.dataset",
    "data.continuous",
]
METHODS = ["fedges", "fusion", "lingam"]


def _imported_modules(path: Path) -> set[str]:
    names = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
        elif isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
    return names


def _old_paths(path: Path) -> set[str]:
    return {
        n
        for n in _imported_modules(path)
        if any(n == f"fedcausal.{m}" or n.startswith(f"fedcausal.{m}.") for m in MOVED)
    }


def test_the_core_is_reached_through_causalbase() -> None:
    """Not even under TYPE_CHECKING does fedcausal's own code use an old path."""
    files = sorted(SRC.rglob("*.py"))
    offenders = {p.name: sorted(i) for p in files if (i := _old_paths(p))}
    assert not offenders
    assert len(files) > 20  # the walk found the package


def test_the_methods_never_import_flower() -> None:
    files = [f for package in METHODS for f in sorted((SRC / package).rglob("*.py"))]
    offenders = {
        p.name: sorted(i)
        for p in files
        if (i := {n for n in _imported_modules(p) if n.split(".")[0] == "flwr"})
    }
    assert not offenders
    assert files


def _run_without(blocked: list[str], code: str) -> subprocess.CompletedProcess[str]:
    """Run `code` in a fresh interpreter in which the `blocked` modules cannot be imported."""
    prelude = textwrap.dedent(
        f"""
        import sys
        from importlib.abc import MetaPathFinder

        BLOCKED = {blocked!r}

        class Block(MetaPathFinder):
            def find_spec(self, name, path=None, target=None):
                if any(name == b or name.startswith(b + ".") for b in BLOCKED):
                    raise ModuleNotFoundError(f"No module named {{name!r}}", name=name)
                return None

        sys.meta_path.insert(0, Block())
        """
    )
    return subprocess.run(
        [sys.executable, "-c", prelude + textwrap.dedent(code)],
        capture_output=True,
        text=True,
        timeout=600,
    )


def test_without_flower_the_methods_run_in_one_process() -> None:
    """In a fresh interpreter where Flower cannot be imported.

    FedGES and FedRCD run in one process, and the transport says which extra it
    needs.
    """
    result = _run_without(
        ["flwr", "ray"],
        """
        import numpy as np
        from causalbase.datasets import load_cancer
        from fedcausal import FedGES, FedRCD
        from fedcausal.data import contiguous_split

        bundle = load_cancer()
        FedGES(n_rounds=2).fit(contiguous_split(bundle.data, 2))
        x = np.random.default_rng(0).laplace(size=(400, 3))
        FedRCD().fit([x[:200], x[200:]])
        try:
            import fedcausal.flower.fedges
        except ModuleNotFoundError as exc:
            assert "pip install 'fedcausal[federated]'" in str(exc), exc
        else:
            raise AssertionError("fedcausal.flower imported without flwr")
        """,
    )
    assert result.returncode == 0, result.stderr
