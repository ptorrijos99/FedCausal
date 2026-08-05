# Validation

What is checked, so you can decide how much to trust it.

## Run it yourself

```bash
pip install -e ".[dev,flower]"
pytest
```

No downloads, no environment variables, no JVM. Everything the tests need ships with the package. Four xfails are expected; they are explained below.

## What is checked

| Layer | How |
|---|---|
| BDeu score | Every family's local score pinned against reference values |
| GES search | Full runs pinned on BDeu and on the resulting CPDAG |
| Fusion operators | Pinned by edge set, plus property tests over random DAGs |
| Graph operations | SHD, SMHD, moralisation, treewidth, CPDAG conversion all pinned |
| Structure recovery | Learn from data sampled out of a known network, compare to it |
| Against causal-learn | Same data through both, results compared |
| FedRCD solvers | Pinned against the original FedLiNGAM implementation's recorded output |
| Flower transports | Identical results to the in-process code, round by round, for both families |
| Incremental engine | Byte-identical to the full scan on every benchmark network |

The transport check is the one worth knowing about. For FedGES it asserts `BNModel` equality per round rather than only at the end; for the FedRCD family it asserts the same causal order and coefficient matrix to 1e-12. Both run on every platform rather than only where Flower's Ray-based simulation installs, because the loops are driven over a stub grid that routes messages to the real client handlers in-process. Nothing is faked but the wire.

## Why search runs are compared on score, not on edges

A search returns one arbitrary member of a Markov equivalence class. Comparing directed edge sets would be comparing two arbitrary choices, and would fail for reasons that have nothing to do with correctness.

BDeu is identical across every DAG extension of a CPDAG, so it is the quantity that means something. The CPDAG itself is compared separately, in CPDAG space.

The same reasoning applies to metrics. SMHD moralises both graphs and is invariant across an equivalence class, so it is pinned exactly. SHD is not invariant, so it is asserted as a tolerant bound.

## The four expected xfails

On `child` and `alarm`, fedcausal reaches a strictly *better* optimum than the original Java implementation, so the pinned comparison fails in fedcausal's favour. Both are marked `strict`, so the suite goes red if that ever stops being true, and each is counted twice because it is checked at two levels.

On `child` the cause is the `empty_t_gate` quirk described in [Algorithms](algorithms.md): turning that gate on reproduces the Java result bit for bit. On `alarm` the cause is not identified, and the tests say so rather than borrowing `child`'s explanation; what is verified there is that the Java result is not a local optimum of its own operator.

## Limits

The incremental search engine is validated empirically rather than proven. Score deltas are local to the two endpoints, but operator validity is not, so equivalence to the full scan is asserted by a differential test on every benchmark network rather than derived. `incremental=False` gives you the exact-by-construction scan.

The federated loop is checked against fedcausal's own in-process engine, and its pieces are pinned individually, but there is no end-to-end comparison of a full federated run against another implementation.

## Test selection

```bash
pytest -m "not slow"        # skip the long-running ones
pytest -m regression        # only the pinned comparisons
```

`bigdata` tests need networks too large to ship and skip unless `FEDCAUSAL_NETWORKS` points at a directory holding them. See [Benchmarks](benchmarks.md).
