"""The package imports, and the algorithms do not need the optional transport."""

from __future__ import annotations

import sys

import pytest


@pytest.mark.parametrize("package", ["fedcausal.fedges", "fedcausal.lingam"])
def test_algorithms_import_without_flwr(package: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Only `fedcausal.flower` may reach for flwr, and it is an optional extra."""
    monkeypatch.setitem(sys.modules, "flwr", None)
    for name in [m for m in sys.modules if m.startswith(package)]:
        monkeypatch.delitem(sys.modules, name, raising=False)

    __import__(package)
