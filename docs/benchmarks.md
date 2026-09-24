# Benchmarks

Measured on one machine, single-threaded, with the default settings. Structures are exact; timings are not, so read them as orders of magnitude and see the note at the end.

## How fast, and how large

5000 rows per network. Times are search only.

| Network | Variables | Seconds | Edges found | SHD | SMHD |
|---|---:|---:|---:|---:|---:|
| cancer | 5 | < 0.01 | 3 | 1 | 2 |
| asia | 8 | < 0.01 | 7 | 2 | 1 |
| child | 20 | 0.05 | 25 | 2 | 0 |
| insurance | 27 | 0.12 | 44 | 17 | 19 |
| water | 32 | 0.11 | 29 | 49 | 91 |
| mildew | 35 | 0.21 | 33 | 22 | 40 |
| alarm | 37 | 0.20 | 46 | 10 | 10 |
| hailfinder | 56 | 0.58 | 59 | 58 | 72 |
| hepar2 | 70 | 0.69 | 69 | 66 | 75 |
| win95pts | 76 | 1.03 | 108 | 39 | 53 |
| pathfinder | 109 | 1.94 | 147 | 196 | 235 |
| andes | 223 | 6.26 | 285 | 59 | 146 |
| diabetes | 413 | 57.9 | 583 | 309 | 251 |
| pigs | 441 | 38.1 | 592 | 0 | 0 |
| link | 724 | 140 | 855 | 679 | 987 |
| munin | 1041 | 842 | 977 | 1230 | 1582 |

Runtime tracks how dense and interconnected a network is, not just how many variables it has. `pigs` has 441 variables to `diabetes`'s 413, and still finishes in 38 s against 58 s.

## Against causal-learn

The same data through both, same machine.

| Network | Variables | fedcausal | causal-learn | Ratio |
|---|---:|---:|---:|---:|
| cancer | 5 | < 0.01 | 0.06 |  |
| asia | 8 | < 0.01 | 0.20 |  |
| child | 20 | 0.05 | 7.45 | 139x |
| insurance | 27 | 0.12 | 31.8 | 255x |
| water | 32 | 0.11 | 18.1 | 159x |
| mildew | 35 | 0.21 | 209 | 1002x |
| alarm | 37 | 0.20 | 52.9 | 258x |
| hailfinder | 56 | 0.58 | 1459 | 2508x |
| hepar2 | 70 | 0.69 | 53.2 | 78x |
| win95pts | 76 | 1.03 | 97.5 | 94x |
| pathfinder | 109 | 1.94 | 552 | 284x |
| andes | 223 | 6.26 | > 30 min |  |
| diabetes | 413 | 57.9 | > 30 min |  |
| pigs | 441 | 38.1 | > 30 min |  |
| link | 724 | 140 | not run |  |
| munin | 1041 | 842 | not run |  |

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

That is 9x for an identical result on this network. The cap forfeits the optimality guarantee in theory; in practice it has not changed a recovered structure on any network the two have been compared on.

`incremental=True`, the default, against the full scan:

| Network | Variables | Incremental | Full scan | Speedup |
|---|---:|---:|---:|---:|
| alarm | 37 | 0.22 | 0.5 | 2.3x |
| hepar2 | 70 | 0.63 | 2.0 | 3.1x |
| win95pts | 76 | 1.09 | 4.0 | 3.6x |
| andes | 223 | 7.98 | 71.1 | 8.9x |

Both defaults return scores identical to the slower setting everywhere the two have been compared. For `max_t_size` that runs up to `link` at 724 variables, where 2, 4 and `None` all give the same graph. For `incremental` it runs up to `andes` at 223, since a full scan above that is expensive. Neither has been compared on `munin`. The one option known to cost accuracy is `empty_t_gate`; see [Algorithms](algorithms.md).

The `accel` extra JITs the semi-directed path check, the dominant cost on large graphs. Measured back to back on `link` (724 variables) it is 1.67x end to end, for a bit-identical result. Below a variable-count threshold the pure-Python walk wins and is used instead, so on small networks the extra changes nothing.

## Cross-checked against the original implementation

The reference tests pin the search against the original Java implementation on networks up to 37 variables. `link`, at 724, has been run through both on the same data:

| | Edges | SHD | SMHD | BDeu |
|---|---:|---:|---:|---:|
| true structure | 1125 | 0 | 0 | -1,059,715 |
| BayesFL (Java) | 1779 | 1597 | 3153 | -1,114,310 |
| fedcausal | 1770 | 1591 | 3098 | -1,114,303 |

The two scores differ by 7 in 1.1 million, and the skeletons share 1666 edges. Both were scored without a structure prior, which is what the Java implementation effectively uses. fedcausal took 186 s against 2900 s.

## Structure quality on the largest networks

The two biggest networks recover badly, and it is worth being explicit about why, because the cause is not a defect in the search.

On `link` (724 variables, 1125 true edges) the search returns 1770 edges scoring -1,114,303. The true structure scores -1,059,714, so the result is 55,000 worse than the answer. That rules out overfitting, which would score *better* than the truth. Warm-starting the same search from the true structure lands at -1,058,834, better than the truth itself. The search is therefore working; greedy search from an empty graph simply reaches a poor local optimum on a network this large and this sparsely sampled. Chickering's optimality guarantee is asymptotic, and 5000 rows over 724 variables is not that regime.

`max_t_size` is not involved: 2, 4, and `None` return the same graph on `link`.

Two levers help if you work at this scale. A complexity penalty helps most: `structure_prior=1.0` takes `link` from 1770 edges to 855, and SMHD from 3098 to 987. Warm-starting from any reasonable prior structure helps too, which is also what the federated loop gives its clients after the first round.

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

Treat the absolute seconds loosely. Repeating the whole ladder on the same machine, with the same code and identical resulting structures, moved individual timings by up to a factor of two, and a sync client running in the background made it worse still. What survives that noise is the ratios, which are taken back to back in one process, and the orders of magnitude. If you re-run these, do it outside a synced folder and compare within a run rather than across runs.
