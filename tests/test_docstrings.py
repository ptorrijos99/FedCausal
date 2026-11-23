"""Every example in a docstring runs and prints what it shows."""

from __future__ import annotations

import doctest
import importlib
import importlib.util
import pkgutil

import pytest

import fedcausal


def _modules() -> list[str]:
    names = []
    for info in pkgutil.walk_packages(fedcausal.__path__, "fedcausal."):
        if info.name.startswith("fedcausal.flower") and importlib.util.find_spec("flwr") is None:
            continue
        names.append(info.name)
    return sorted(names)


@pytest.mark.parametrize("name", _modules())
def test_docstring_examples_run(name: str) -> None:
    module = importlib.import_module(name)
    flags = doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE
    finder = doctest.DocTestFinder(exclude_empty=True)
    runner = doctest.DocTestRunner(optionflags=flags)
    for test in finder.find(module, name):
        runner.run(test)
    result = runner.summarize(verbose=False)
    assert result.failed == 0, f"{result.failed} of {result.attempted} examples failed in {name}"


def test_some_examples_exist() -> None:
    """Guard the guard: a finder that found nothing would make the test above vacuous."""
    finder = doctest.DocTestFinder(exclude_empty=True)
    found = sum(
        len(finder.find(importlib.import_module(n), n))
        for n in ("fedcausal.datasets", "fedcausal.search._ges", "fedcausal.scores._family")
    )
    assert found >= 3
