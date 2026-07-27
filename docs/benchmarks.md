# Benchmarks

Measured on one machine, single-threaded, with the default settings. Treat them as orders of magnitude.

## How fast, and how large

5000 rows per network. Times are search only.

| Network | Variables | Seconds | Edges found | SHD | SMHD |
|---|---:|---:|---:|---:|---:|
| cancer | 5 | < 0.01 | 3 | 1 | 2 |
| asia | 8 | < 0.01 | 8 | 3 | 2 |
| child | 20 | 0.05 | 25 | 2 | 0 |
| insurance | 27 | 0.12 | 45 | 16 | 18 |
| water | 32 | 0.13 | 35 | 55 | 81 |
| mildew | 35 | 0.21 | 33 | 22 | 40 |
| alarm | 37 | 0.21 | 47 | 10 | 11 |
| hailfinder | 56 | 0.75 | 66 | 62 | 73 |
| hepar2 | 70 | 0.74 | 85 | 64 | 66 |
| win95pts | 76 | 1.41 | 153 | 74 | 117 |
| pathfinder | 109 | 3.56 | 198 | 237 | 303 |
| andes | 223 | 14.2 | 366 | 101 | 178 |
| diabetes | 413 | 86.6 | 613 | 332 | 239 |
| pigs | 441 | 51.1 | 592 | 0 | 0 |
| link | 724 | 436 | 1807 | 1628 | 3243 |
| munin | 1041 | 1073 | 1446 | 1603 | 2453 |

Runtime tracks how dense and interconnected a network is, not just how many variables it has. `pigs` has 441 variables to `diabetes`'s 413, and still finishes in 51 s against 87 s.

## Against causal-learn

The same data through both, same machine.

| Network | Variables | fedcausal | causal-learn | Ratio |
|---|---:|---:|---:|---:|
| cancer | 5 | < 0.01 | 0.06 |  |
| asia | 8 | < 0.01 | 0.20 |  |
| child | 20 | 0.05 | 7.45 | 141x |
| insurance | 27 | 0.12 | 31.8 | 257x |
| water | 32 | 0.13 | 18.1 | 140x |
| mildew | 35 | 0.21 | 209 | 985x |
| alarm | 37 | 0.21 | 52.9 | 251x |
| hailfinder | 56 | 0.75 | 1459 | 1941x |
| hepar2 | 70 | 0.74 | 53.2 | 72x |
| win95pts | 76 | 1.41 | 97.5 | 69x |
| pathfinder | 109 | 3.56 | 552 | 155x |
| andes | 223 | 14.2 | > 30 min |  |
| diabetes | 413 | 86.6 | > 30 min |  |
| pigs | 441 | 51.1 | > 30 min |  |
| link | 724 | 436 | not run |  |
| munin | 1041 | 1073 | not run |  |

## What federating costs you

Mean SMHD against the true network, over five sampled datasets per cell. Lower is better. The first column learns from everything at once; the rest split the same data across that many sites.

| Network | Variables | Centralised | 5 sites | 10 sites | 20 sites |
|---|---:|---:|---:|---:|---:|
| cancer | 5 | 0.6 | 1.8 | 1.8 | 2.6 |
| asia | 8 | 1.4 | 2.6 | 1.2 | 2.6 |
| child | 20 | 0.0 | 6.4 | 9.4 | 9.4 |
| insurance | 27 | 21.6 | 29.0 | 33.8 | 39.0 |
| alarm | 37 | 11.2 | 11.6 | 8.0 | 12.2 |
| hepar2 | 70 | 72.2 | 119.6 | 123.8 | 217.8 |

Splitting the same data across more sites costs accuracy, because each site sees less of it. How much depends on the network rather than on its size. The largest cost here is `hepar2`: 72 structural errors learning centrally against 218 across twenty sites. On the smallest networks the gap is a couple of edges either way, which is within the spread between samples.

## What the options buy

`max_t_size=2`, the default, against the exhaustive scan on `hailfinder`, measured back to back:

| | Seconds | Same BDeu |
|---|---:|---|
| `max_t_size=2` (default) | 0.53 | |
| `max_t_size=None` | 4.8 | yes |

That is 9x for an identical result on this network. The cap is not free in theory, since it forfeits the optimality guarantee, but it has not changed a recovered structure on anything measured.

`incremental=True`, the default, against the full scan:

| Network | Variables | Incremental | Full scan | Speedup |
|---|---:|---:|---:|---:|
| alarm | 37 | 0.22 | 0.5 | 2.3x |
| hepar2 | 70 | 0.63 | 2.0 | 3.1x |
| win95pts | 76 | 1.09 | 4.0 | 3.6x |
| andes | 223 | 7.98 | 71.1 | 8.9x |

Both defaults produce scores identical to the exhaustive, non-incremental search everywhere the comparison has been run, which is networks up to 223 variables. Neither has been checked against the exhaustive scan on `link` or `munin`, where a full scan is expensive. The one option known to cost accuracy is `empty_t_gate`; see [Algorithms](algorithms.md).

The `accel` extra JITs the semi-directed path check, the dominant cost on large graphs. Measured back to back on `link` (724 variables) it is 1.67x end to end, for a bit-identical result. Below a variable-count threshold the pure-Python walk wins and is used instead, so on small networks the extra changes nothing.

## Regenerating this page

The tables above are generated, not written:

```bash
python benchmarks/build_docs.py
```

Every number comes from `benchmarks/results/`, so re-running the benchmarks and then this script keeps the page honest. The prose describing each table is derived from it too, so an observation cannot outlive the data behind it.

## Running them yourself

```bash
export FEDCAUSAL_NETWORKS=/path/to/networks
python benchmarks/scale_ges.py                    # the size ladder
python benchmarks/compare_ges.py                  # fedcausal vs causal-learn
python benchmarks/report_ges.py                   # render the results
python benchmarks/compare_fedges.py               # quality against site count
python benchmarks/report_fedges.py
```

`FEDCAUSAL_NETWORKS` should point at a directory laid out as `<root>/res/networks/BBDD/<net>/<net>.0.csv` for the data and `<root>/res/networks/<net>.xbif` for the ground truths. Networks come from the [Bayesian Network Repository](https://www.bnlearn.com/bnrepository/); the five smallest ship with the package under `tests/fixtures/networks/`.

`compare_ges.py` runs one subprocess per cell and appends results as they finish, so a hang costs one cell and the run resumes where it stopped.

## A note on timing

Wall-clock on a machine with a sync client running varied by 40% between invocations of the same job, which is enough to swamp most differences worth measuring. The numbers here were taken with it off, and the ratios come from back-to-back runs in one process. If you re-run these, do it outside a synced folder.
