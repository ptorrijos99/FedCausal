# Benchmarks

Measured on one laptop with the default settings, 5000 rows per network, from the [Bayesian Network Repository](https://www.bnlearn.com/bnrepository/). Structures are exact; timings are not, so read them as orders of magnitude.

## How fast, and how close to the truth

| Network | Variables | Seconds | Edges | True edges | SHD | SMHD |
|---|---:|---:|---:|---:|---:|---:|
| cancer | 5 | < 0.01 | 3 | 4 | 1 | 2 |
| asia | 8 | 0.01 | 7 | 8 | 2 | 1 |
| child | 20 | 0.04 | 25 | 25 | 2 | 0 |
| insurance | 27 | 0.05 | 44 | 52 | 17 | 19 |
| water | 32 | 0.04 | 31 | 66 | 50 | 88 |
| mildew | 35 | 0.06 | 33 | 46 | 22 | 40 |
| alarm | 37 | 0.06 | 47 | 46 | 10 | 11 |
| hailfinder | 56 | 0.10 | 62 | 66 | 58 | 72 |
| hepar2 | 70 | 0.10 | 71 | 123 | 64 | 73 |
| win95pts | 76 | 0.13 | 109 | 112 | 40 | 52 |
| pathfinder | 109 | 0.29 | 157 | 195 | 203 | 245 |
| andes | 223 | 0.60 | 286 | 338 | 60 | 147 |
| diabetes | 413 | 4.36 | 583 | 602 | 309 | 251 |
| pigs | 441 | 2.65 | 592 | 592 | 0 | 0 |
| link | 724 | 7.22 | 855 | 1125 | 679 | 987 |
| munin | 1041 | 20.7 | 994 | 1397 | 1233 | 1601 |

Runtime follows how dense and interconnected a network is more than its size: `pigs` has more variables than `diabetes` and finishes faster. For comparison, causal-learn's GES on the same data takes 53 s on `alarm` and does not finish `andes` in half an hour.

## Against the previous release

The same data and the same structure prior through 0.1.0 and 0.2.0, on the same machine:

| Network | Variables | 0.1.0 (s) | 0.2.0 (s) | Speedup |
|---|---:|---:|---:|---:|
| alarm | 37 | 0.89 | 0.05 | 18x |
| hailfinder | 56 | 2.26 | 0.10 | 23x |
| win95pts | 76 | 3.59 | 0.15 | 24x |
| pathfinder | 109 | 7.14 | 0.33 | 22x |
| andes | 223 | 24.8 | 0.72 | 34x |
| pigs | 441 | 113.5 | 2.71 | 42x |

Both versions learn structures with the same number of edges and the same BDeu score on every network in the table. The prior was 0.1.0's default, one parent expected per `n_rows - 1` candidates, which 0.2.0 expresses as `expected_parents=(n_vars - 1) / (n_rows - 1)`.

## What federating costs

Mean SHD against the true network over three samples. The first column learns from all the data in one place; the others split the same rows across that many sites.

| Network | Variables | Centralised | 5 sites | 10 sites | 20 sites |
|---|---:|---:|---:|---:|---:|
| cancer | 5 | 0.7 | 1.0 | 2.0 | 3.0 |
| asia | 8 | 2.0 | 2.3 | 2.0 | 4.3 |
| child | 20 | 2.0 | 7.0 | 10.0 | 10.0 |
| insurance | 27 | 21.7 | 26.0 | 28.0 | 29.7 |
| alarm | 37 | 10.0 | 5.0 | 5.0 | 6.7 |
| hepar2 | 70 | 62.7 | 76.7 | 96.7 | 109.7 |

Each site sees less data the more sites there are, and on most networks the error grows with the number of sites; `hepar2` goes from 63 learning centrally to 110 across twenty sites. `alarm` is the exception: the fused structure is closer to the truth than the one learned from all the data at once, as it already was with 0.1.0.

## Running them

```bash
export FEDCAUSAL_NETWORKS=/path/to/networks   # holds xbif/<net>.xbif and BBDD/<net>/<net>.<seed>.csv.gz
python benchmarks/scale_ges.py
python benchmarks/fedges_sites.py --seeds 3
```

Both print the Markdown tables above. Compare timings within one run rather than across runs: a synced folder or a busy machine moves them by tens of percent.
