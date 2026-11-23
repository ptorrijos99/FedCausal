"""The package splits into a core and a federated part, and the core stands alone.

The core (data, graph, scores, search, metrics, io, datasets) must never import
the federated part (fedges, fusion, lingam, flower), so that splitting them
into two distributions later is a move rather than a rewrite.
"""

from __future__ import annotations

import ast
import subprocess
import sys
import textwrap
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src" / "fedcausal"
CORE = ["data", "graph", "scores", "search", "metrics", "io", "datasets"]
CORE_MODULES = ["_base.py", "exceptions.py"]
FEDERATED = ["fedges", "fusion", "lingam", "flower"]


def _core_files() -> list[Path]:
    files = [SRC / name for name in CORE_MODULES]
    for package in CORE:
        files += sorted((SRC / package).rglob("*.py"))
    return files


def _imported_modules(path: Path) -> set[str]:
    names = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
        elif isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
    return names


def _federated_imports(path: Path) -> set[str]:
    return {
        name
        for name in _imported_modules(path)
        if any(name == f"fedcausal.{p}" or name.startswith(f"fedcausal.{p}.") for p in FEDERATED)
        or name == "flwr"
        or name.startswith("flwr.")
    }


def test_the_core_never_imports_the_federated_part() -> None:
    """Not even under TYPE_CHECKING: a split package would not have the names."""
    offenders = {p.name: sorted(i) for p in _core_files() if (i := _federated_imports(p))}
    assert not offenders
    assert len(_core_files()) > 20  # the walk found the core


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


def test_without_flower_the_core_stands_alone_and_the_methods_run() -> None:
    """In a fresh interpreter where Flower cannot be imported.

    The core learns without loading any federated module; FedGES and FedRCD run
    in one process; and the transport says which extra it needs.
    """
    result = _run_without(
        ["flwr", "ray"],
        """
        import sys
        import numpy as np
        from fedcausal import GES
        from fedcausal.datasets import load_cancer
        from fedcausal.metrics import shd

        bundle = load_cancer()
        shd(bundle.truth, GES().fit(bundle.data).dag_)
        federated = ("fedcausal.fedges", "fedcausal.fusion", "fedcausal.lingam", "fedcausal.flower")
        loaded = sorted(m for m in sys.modules if m.startswith(federated))
        assert not loaded, loaded

        from fedcausal import FedGES, FedRCD
        from fedcausal.data import contiguous_split

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
