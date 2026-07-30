"""The PGM 2026 reproduction path must keep working. Tiny configuration, real code path."""

import json
import subprocess
import sys


def test_quick_benchmark_runs_and_writes_records(tmp_path):
    out = tmp_path / "pgm26.jsonl"
    result = subprocess.run(
        [
            sys.executable,
            "benchmarks/pgm26/benchmark.py",
            "--mode",
            "main",
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
        assert {"method", "n_vars", "n_clients", "seed", "shd", "order_correct"} <= record.keys()


def test_quick_stratification_runs_and_writes_records(tmp_path):
    out = tmp_path / "pgm26_strat.jsonl"
    result = subprocess.run(
        [
            sys.executable,
            "benchmarks/pgm26/benchmark.py",
            "--mode",
            "stratification",
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
    required = {"method", "n_vars", "n_clients", "seed", "shd", "order_correct", "family"}
    for record in records:
        assert required <= record.keys()
