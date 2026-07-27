# Validation

How correctness is checked, and where it stops.

## Regression against committed reference outputs

`tests/regression/` pins behaviour against reference outputs stored as JSON.
They cover the BDeu score family by family, full search runs, the fusion
operators, and the graph operations underneath both (SHD, SMHD, moralisation,
treewidth, CPDAG conversion).

The reference outputs came from the implementation fedcausal was developed
against. The five networks they need ship with the package as gzipped fixtures,
so the whole suite runs with no downloads and no environment variables.

Search runs are compared on BDeu, not on raw directed edge sets. A search
returns one arbitrary member of a Markov equivalence class, so comparing edges
would be comparing two arbitrary choices. BDeu is identical across every DAG
extension of a CPDAG, which makes it the quantity that means something. The
CPDAG itself is compared separately, in CPDAG space.

### The four expected xfails

Two networks fail on purpose, marked `strict` so the suite goes red if they
ever start passing.

On `child`, fedcausal reaches a strictly better optimum. The reference gates
each candidate edge on the `T = {}` case before it will consider a non-empty
`T`; fedcausal enumerates every subset independently. Emulating that gate
reproduces the reference's result bit for bit, which is what identifies it as
the cause.

On `alarm`, fedcausal also reaches a better optimum, and the reason is not
established. What is verified: the reference's returned graph is not a local
optimum of its own operator, since an admissible `Insert(VENTLUNG, INTUBATION,
T={})` with delta +749.05 is still available in it, and that delta is exactly
the reported BDeu gap. Tie-break direction and single-orientation candidate
sets have been ruled out by experiment. The honest statement is that the cause
is unknown, and it is recorded that way rather than borrowing `child`'s
explanation.

In both cases fedcausal's score is higher, not lower, so neither is a defect
here. Both are counted twice because each is checked at two levels.

## Independent cross-checks

Pinning against one implementation only proves the two agree. Three other
things are checked.

Recovery tests learn from data forward-sampled out of a known network and
assert the learned structure is close to it. On `cancer` and `child` the
recovered CPDAG equals the true one exactly; the others assert bounds on SHD
and SMHD.

A cross-check against [causal-learn](https://causal-learn.readthedocs.io) runs
both implementations on the same data and compares. It needs the `compare`
extra and skips without it.

Property-based tests over randomly generated DAGs assert the invariants the
fusion operators must satisfy whatever the tie-breaking: the result is acyclic,
the union contains every input edge in transformed form, and the budgeted
variants respect their budgets.

## Both transports produce the same models

The strongest claim in the project is that the Flower transport and the
in-process engine produce identical global models, round by round, for the same
seed and configuration. `tests/flower/test_transport_equivalence.py` asserts
exactly that, on `BNModel` equality per round rather than only at the end.

It runs everywhere, not only where Flower's simulation engine installs: the
loop is driven over a stub grid that routes messages to the real client
handler in-process. Nothing is faked but the wire. Where Ray is available the
same check runs again through a genuine simulation, which additionally
exercises Flower's own serialization.

## What is not validated

The federated loop as a whole is checked against fedcausal's own in-process
engine, and its pieces are pinned individually, but no end-to-end comparison of
a full federated run against another implementation exists.

The incremental search engine is validated empirically rather than proven; see
[Algorithms](algorithms.md).

SHD is not invariant across a Markov equivalence class, so it is asserted as a
tolerant bound rather than pinned exactly. SMHD is invariant and is pinned.

## Running it

```bash
pytest                      # everything self-contained
pytest -m "not slow"        # skip the long-running tests
pytest -m regression        # only the reference comparisons
```

`bigdata` tests need networks too large to ship and skip unless
`FEDCAUSAL_NETWORKS` points at a directory holding them.
