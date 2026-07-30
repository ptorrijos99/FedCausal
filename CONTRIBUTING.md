# Contributing

## Getting set up

```bash
git clone https://github.com/ptorrijos99/fedcausal
cd fedcausal
pip install -e ".[dev,flower]"
pytest
```

The suite is self-contained. No downloads, no environment variables, no JVM. Four xfails are expected and [Validation](docs/validation.md) explains why.

## Before you open a pull request

```bash
ruff check . && ruff format --check .
mypy src
pytest
```

CI runs the same three on Linux and Windows, on Python 3.10 and 3.13.

## What the tests are for

`tests/regression/` pins behaviour against committed reference outputs. If a change moves one of those numbers, that is the change telling you something. Do not update the golden file to match new output without understanding why it moved, and say so in the pull request when you do.

Markers:

- `slow` for anything over a few seconds. CI runs these on one matrix entry.
- `regression` for the reference comparisons.
- `bigdata` for tests needing networks too large to ship. They skip unless `FEDCAUSAL_NETWORKS` points at a directory holding them.

## Performance changes

Two things make a performance claim believable here, and wall-clock is neither of them on its own.

Prefer a deterministic counter: calls avoided, pairs re-scored, cache hits. It is immune to machine noise and it says what actually changed.

If you do report timings, take them within one process, back to back, minimum of several runs. The development machine varied 40% between invocations of the same run because the repository sat in a synced folder, and that swamps most optimisations worth making. [Benchmarks](docs/benchmarks.md) has the details.

Any change to the search must keep the incremental engine byte-identical to the full scan. `tests/search/test_incremental.py` checks that; do not weaken it.

## Adding a method

`fedcausal.fedges` and `fedcausal.lingam` are the two method families, and a third one goes beside them: a package holding the algorithm and nothing about transport. If it federates, add one module to `fedcausal.flower` and reuse the plumbing in `_grid.py`, the loaders in `loaders.py`, and the wire format in `serde.py`. The algorithm packages must not import `flwr`; `tests/flower/test_lingam_oneshot.py` has the check that keeps it that way.

## Style

Ruff handles formatting and import order, so do not argue with it.

Docstrings follow the numpydoc convention. Full parameter blocks on the public API, one line when the signature already says everything. Explain why something is the way it is when that is not obvious; do not restate what the code says.

Development history belongs in commit messages and pull requests, not in source comments. If you need to record why a decision was reversed, the commit is the place.
