# Contributing

## Getting set up

```bash
git clone https://github.com/ptorrijos99/fedcausal
cd fedcausal
pip install -e ".[dev,flower]"
pytest
```

The suite is self-contained. No downloads, no environment variables, no JVM.

## Before you open a pull request

```bash
ruff check . && ruff format --check .
mypy src
pytest
```

CI runs the same three on Linux and Windows, on Python 3.10 and 3.13.

## What the tests are for

A test states something a user could observe: a result, an error, a guarantee in the docs. It should fail only when that stops being true. So:

- Assert behaviour through the public API, not private helpers, cache contents, memory layout or the order a loop happens to visit things.
- Compare floating-point results to a tolerance, never bit for bit. The last few bits move between platforms, BLAS builds and libm versions.
- Do not pin tie-breaks or values recorded from one run. If two answers are equally correct, the test must accept both.
- Do not add a strict xfail to record a known difference. Assert the property that makes the difference acceptable.
- Prefer checking against an independent definition (a brute-force score, Verma and Pearl's characterisation of equivalence) over checking against recorded output.

The whole suite runs in about half a minute. The three tests that drive Flower's Ray simulation engine carry the `slow` marker and are skipped by default: each takes a minute, and Ray now and then hangs when one process starts it twice. Run them with `pytest -m slow` after changing the transport; they need the `flower-simulation` extra.

## Performance changes

Two things make a performance claim believable here, and wall-clock is neither of them on its own.

Compare timings within one process, back to back, after a warm-up fit so that numba's compilation is not counted, and take the minimum of several runs. A synced folder or a busy machine moves wall-clock by tens of percent, which swamps most optimisations worth making. `benchmarks/scale_ges.py` runs the network ladder in [Benchmarks](docs/benchmarks.md).

A change to the search must keep it at a local optimum of the full neighbourhood; `tests/search/test_ges.py` checks that.

## Adding a method

`fedcausal.fedges` and `fedcausal.lingam` are the two method families, and a third one goes beside them: a package holding the algorithm and nothing about transport. If it federates, add one module to `fedcausal.flower` and reuse the plumbing in `_grid.py`, the loaders in `loaders.py`, and the wire format in `serde.py`. The algorithm packages must not import `flwr`; `tests/test_import.py` checks that.

## Style

Ruff handles formatting and import order, so do not argue with it.

Docstrings follow the numpydoc convention. Full parameter blocks on the public API, one line when the signature already says everything. Explain why something is the way it is when that is not obvious; do not restate what the code says.

Development history belongs in commit messages and pull requests, not in source comments. If you need to record why a decision was reversed, the commit is the place.
