# Benchmarks

All numbers below are single-threaded, pure Python unless stated, measured on
one developer machine. Treat them as orders of magnitude, not specifications.

## Does it scale

The target was roughly a thousand variables, which is where most Python
structure learners stop being usable.

| Network | Variables | Rows | Time | Edges found |
|---|---:|---:|---:|---:|
| hepar2 | 70 | 5 000 | ~3 s | |
| andes | 223 | 5 000 | ~3 min | |
| pigs | 441 | 5 000 | ~9 min | |
| link | 724 | 5 000 | ~7 min | |
| munin | 1 041 | 5 000 | ~25 min | 1 446 |

`link` finishing faster than `pigs` is not a typo. Runtime tracks neighbourhood
width and edge density more than node count.

## Against causal-learn

Search time in seconds, same data, same machine, both single-threaded:

| Network | Variables | fedcausal | causal-learn |
|---|---:|---:|---:|
| child | 20 | 0.12 | 4.32 |
| insurance | 27 | 0.32 | 12.78 |
| water | 32 | 0.34 | 8.58 |
| mildew | 35 | 0.40 | 79.64 |
| alarm | 37 | 0.57 | 23.07 |
| hailfinder | 56 | 1.86 | 1 062.46 |
| hepar2 | 70 | 2.98 | 58.31 |
| win95pts | 76 | 6.40 | 107.81 |
| pathfinder | 109 | 22.00 | 575.89 |
| andes | 223 | 172.15 | >1 800 |
| diabetes | 413 | 1 465.39 | >1 800 |
| pigs | 441 | 564.22 | >1 800 |

causal-learn is 30-40x slower on the small networks and does not finish the
large ones within a 30-minute budget. The `hailfinder` figure for fedcausal is
with the default `max_t_size=2`; the exhaustive scan takes 24.81 s there and
recovers an identical graph, which is why the cap is the default.

## Where the time goes

Profiling `link` (724 variables) puts the ranking beyond doubt, even though
cProfile inflates absolutes:

| Function | Calls | Time |
|---|---:|---:|
| `is_semidirected_blocked` | 8.26 M | 544 s |
| set and list churn inside that traversal | ~1.5 B | ~190 s |
| CPDAG re-closure after each step | 1 951 | ~95 s |
| the BDeu counting kernel | 2.03 M | 91 s |

The bottleneck is graph traversal in the interpreter, not arithmetic. The score
cache absorbs about 97% of local-score calls, which is why kernel-level
optimisation moves the total so little.

Two byte-exact optimisations follow from that profile. Storing the dataset
column-major makes the kernel's per-column gathers contiguous, worth 1.64x on
the kernel in isolation but only 1.085x end to end, for the reason above.
Skipping the path traversal when the na-only check is already blocked exploits
its monotonicity in the conditioning set, and cuts calls on `diabetes` by 34.5%
(2 238 363 to 1 465 798) for identical output.

The `accel` extra JITs that traversal with numba, worth about 4x on it at
400-700 variables. It is selected automatically above a node-count threshold
and returns a bit-identical result either way.

## A caveat about wall-clock on this machine

The development machine keeps the repository in a OneDrive-synced folder, and
the same run varied between 70 s and 100 s across separate invocations. That
40% swing swamps anything smaller than it.

So the ratios above come from within-process, back-to-back, minimum-of-N
comparisons, and the headline for the traversal optimisation is its
deterministic call-count reduction, which no amount of disk-sync noise can
move. Anything reported as wall-clock across processes is labelled indicative.
If you re-run these, do it outside a synced folder.

## Running them yourself

The harnesses live in `benchmarks/` and assert nothing.

```bash
export FEDCAUSAL_NETWORKS=/path/to/networks
python benchmarks/scale_ges.py                    # the size ladder
python benchmarks/scale_ges.py --profile link     # cProfile one network
python benchmarks/compare_ges.py                  # fedcausal vs causal-learn
python benchmarks/report_ges.py                   # render the results
python benchmarks/compare_fedges.py               # federated quality vs sites
python benchmarks/report_fedges.py
```

`FEDCAUSAL_NETWORKS` should point at a directory laid out as
`<root>/res/networks/BBDD/<net>/<net>.0.csv` for the data and
`<root>/res/networks/<net>.xbif` for the ground truths. Networks are available
from the [Bayesian Network Repository](https://www.bnlearn.com/bnrepository/).
The five smallest ship with the package under `tests/fixtures/networks/`, but
the benchmarks want the larger ones.

`compare_ges.py` runs one subprocess per cell and appends results as they
finish, so a hang costs you one cell and the run resumes where it stopped.
