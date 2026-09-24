"""The PGM 2026 reproduction path must keep working. Tiny configuration, real code path."""

import json
import subprocess
import sys

import pytest

REQUIRED = {"method", "n_vars", "n_clients", "seed", "shd", "order_correct"}


@pytest.mark.parametrize(
    ("mode", "extra"), [("main", set()), ("stratification", {"family"})], ids=["main", "strat"]
)
def test_quick_benchmark_runs_and_writes_records(tmp_path, mode, extra):
    out = tmp_path / f"pgm26_{mode}.jsonl"
    result = subprocess.run(
        [
            sys.executable,
            "benchmarks/pgm26/benchmark.py",
            "--mode",
            mode,
            "--quick",
            "--out",
            str(out),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    records = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
    assert records
    for record in records:
        assert REQUIRED | extra <= record.keys()
