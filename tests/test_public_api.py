"""Every name in an ``__all__`` exists, and every import the docs show works."""

from __future__ import annotations

import importlib
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent

SUBPACKAGES = [
    "fedcausal",
    "fedcausal.data",
    "fedcausal.exceptions",
    "fedcausal.fedges",
    "fedcausal.fusion",
    "fedcausal.graph",
    "fedcausal.io",
    "fedcausal.lingam",
    "fedcausal.metrics",
    "fedcausal.scores",
    "fedcausal.search",
]

DOC_FILES = (
    list(REPO.glob("*.md"))
    + list((REPO / "docs").glob("*.md"))
    + list((REPO / "examples").rglob("*.md"))
)
DOC_IMPORTS = sorted(
    {
        (m.group(1), name.strip())
        for f in DOC_FILES
        for m in re.finditer(
            r"^from (fedcausal[\w.]*) import ([^\n]+)$", f.read_text(encoding="utf-8"), re.M
        )
        for name in m.group(2).split(",")
        if name.strip()
    }
)


@pytest.mark.parametrize("name", SUBPACKAGES)
def test_every_export_resolves(name: str) -> None:
    """Nothing in an ``__all__`` has been renamed out from under it."""
    module = importlib.import_module(name)
    for export in getattr(module, "__all__", []):
        assert hasattr(module, export), f"{name}.__all__ names missing {export}"


@pytest.mark.parametrize(("module", "name"), DOC_IMPORTS, ids=lambda v: str(v))
def test_documented_imports_work(module: str, name: str) -> None:
    """Every ``from fedcausal... import ...`` shown in the docs actually runs."""
    if module.startswith("fedcausal.flower"):
        pytest.importorskip("flwr")
    assert hasattr(importlib.import_module(module), name)


def test_the_docs_import_something() -> None:
    """Guard the guard: an empty scrape would make the test above vacuous."""
    assert len(DOC_IMPORTS) > 15
