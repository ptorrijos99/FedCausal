# Algorithms and options

What each knob does and when to touch it. For the numbers, see [Benchmarks](benchmarks.md).

The federated method is [FedGES](https://doi.org/10.1007/s10994-025-06939-2); the local search is Chickering (2002) greedy equivalence search; the fusion is Puerta et al. (2021) structural consensus.

## What happens when you call `fit`

`GES` searches over equivalence classes, not individual DAGs. A forward phase adds the highest-scoring valid edge until none improves the score, then a backward phase removes edges the same way. Both re-close the graph to a canonical CPDAG after every step.

The result is a CPDAG. A directed edge in it means every DAG in the class agrees on that direction; an undirected edge means the data genuinely cannot distinguish the two orientations. `to_dag()` picks one arbitrarily.

Runs are deterministic. Same data in, same graph out, every time.

## `GES` options

```python
GES(score, h_pool="nayx", max_t_size=2, empty_t_gate=False, incremental=True)
```

| Option | Default | What it does | Change it when |
|---|---|---|---|
| `max_t_size` | `2` | Caps the size of the neighbour subset the forward phase considers per candidate edge. | You need the theoretical guarantee. `None` is Chickering-exact and roughly 9x slower on `hailfinder`, for an identical result there and on every other network compared. |
| `incremental` | `True` | Re-scores only the candidate pairs whose neighbourhood changed after each step, instead of everything. | Never, normally. `False` gives the exact-by-construction full scan if you suspect a bug. |
| `h_pool` | `"nayx"` | Which neighbour set the backward phase draws deletion subsets from. `"nayx"` is Chickering Definition 13. | Matching the original Java implementation bit for bit. See [below](#reproducing-published-fedges-numbers). |
| `empty_t_gate` | `False` | Skips a candidate edge entirely unless it already scores positively with an empty neighbour subset. | Matching the original Java implementation. It costs you the optimum. |

**On `max_t_size`.** The cap is the default because it has produced byte-identical BDeu, edges, SHD, and SMHD to the exhaustive scan on every network it has been compared on, from `asia` at 8 variables to `link` at 724, while running roughly 9x faster on `hailfinder`, the network where the two differ most. On `link`, `max_t_size` of 2, 4, and `None` all return the same graph. See [Benchmarks](benchmarks.md) for the measurements. Any finite cap forfeits Chickering's optimality guarantee in theory, so state which you used in published results.

**On `incremental`.** Verified byte-identical to the full scan on every benchmark network, and worth 2.3x at 37 variables rising to 8.9x at 223. Score deltas are local to the two endpoints, so invalidating pairs around changed nodes keeps every cached delta correct. Operator *validity* is not fully local, which is why this is checked empirically by a differential test rather than proved.

## Fusion operators

All of them transform the input graphs into one shared topological order, then union the parent sets. That is what guarantees the result is acyclic. They differ only in how much of the union they keep.

| Operator | Keeps | Use when |
|---|---|---|
| `ConsensusUnion()` | Everything | You want maximum recall, and density is not a problem |
| `Consensus(pct)` | Edges in at least `pct`% of sites | The general-purpose choice; `Consensus(50)` is the server default |
| `MaxFrequency(k)` | Edges in at least `k` sites | You think in absolute counts rather than percentages |
| `MaxEdges(k)` | The `k` most-agreed edges | You have a hard budget on total edges. `k` may also be `"Mean"`, `"Max"` or `"Min"` of the inputs' own edge counts |
| `MaxParents(k)` | Under a per-node cap | You need bounded in-degree for downstream inference |
| `Intersection()` | Only what every site has | You want maximum precision and can afford to miss edges |

Anything with a `fuse(dags)` method works in their place.

**A caveat on `MaxParents`.** It caps the *source* node's parent count rather than the target's in-degree, which is not what the name suggests. This reproduces a quirk of the original Java implementation so that published results stay comparable. If you need a genuine in-degree cap, this is not the operator.

## Convergence criteria

| Criterion | Stops when |
|---|---|
| `ScoreImprovement()` | The mean local score stops strictly improving |
| `ModelEquality()` | Every site re-emits a model it has produced before |
| `IterationEquality(n)` | The current round repeats one of the last `n` |
| `MultipleConvergence([...])` | Any of its members does |
| `NoneConvergence()` | Never; the full iteration budget runs |

The default is `ScoreImprovement or ModelEquality`. None of the built-ins can fire on the first round, so the loop always produces at least one global model.

## Client refinement

After fusing the global structure into its own, a site can re-search the edges the two disagree on. This is much cheaper than a full re-search because the candidate set is tiny.

| `refinement` | Does |
|---|---|
| `"None"` (default) | Keeps the fused model as-is |
| `"FES"` | Re-runs the forward phase over the differences |
| `"BES"` | Re-runs the backward phase |
| `"GES"` | Both, forward then backward |

## The score

`BDeuScore(data, sample_prior=1.0, structure_prior=1.0)`.

`sample_prior` is the equivalent sample size. `structure_prior` penalises parent count, and `0.0` disables it. Watch the direction: a *larger* value is a *weaker* penalty, because the per-parent cost is `log(e / (n_rows - 1))`.

The prior is on by default because it recovers better structure as networks grow. Over 15 networks with five sampled datasets each, the summed mean SMHD is **811 with `structure_prior=1.0` against 1063 with `0.0`**. It improves six networks, costs six, and ties on three — but the sizes of those moves are what matter, because the improvements are large and land on the big networks while the costs are small and land on the small ones:

| Network | Variables | `0.0` | `1.0` (default) |
|---|---:|---:|---:|
| cancer | 5 | 0.0 | 1.8 |
| water | 32 | 80.8 | 88.8 |
| hepar2 | 70 | 75.6 | 78.6 |
| alarm | 37 | 11.2 | **9.0** |
| win95pts | 76 | 148.2 | **51.4** |
| pathfinder | 109 | 344.4 | **237.8** |
| andes | 223 | 209.2 | **151.6** |

So: pass `structure_prior=0.0` if you work on small networks, or need numbers comparable with published FedGES results.

Both columns come from `benchmarks/results/structure_prior_networks.json`. Regenerate them with `benchmarks/sweep_bundled_prior.py`, which forward-samples from the network definitions rather than needing datasets: with no arguments it covers the five networks that ship with the package, and `--networks <dir>` extends it to any directory of `.xbif` files from the [Bayesian Network Repository](https://www.bnlearn.com/bnrepository/).

Local scores are memoised, and that cache carries most of the search's performance. `GESLearner` keeps one score object per dataset for exactly that reason.

## Reproducing published FedGES numbers

Three settings exist only to match the original Java implementation, [BayesFL](https://github.com/ptorrijos99/BayesFL), bit for bit. They are quirks of that codebase rather than part of the published method, so reach for them when you need to match previously published numbers, not because they are better.

| Setting | What it changes |
|---|---|
| `structure_prior=0.0` | The Java implementation declares `0.001` but never propagates it, so it effectively scores with none |
| `h_pool="bayesfl"` | Draws deletion subsets from the neighbours of `y` *not* adjacent to `x`, rather than from `NaYX(x, y)` as Chickering's Definition 13 specifies. Those sets are disjoint, so every subset ties and the backward phase only ever deletes with an empty one. Agrees exactly with `"nayx"` on every network compared, up to 223 variables |
| `empty_t_gate=True` | Skips a candidate edge unless it already scores positively with an empty neighbour subset. The only one of the three that demonstrably costs you a worse structure |

`MaxParents`'s source-cap behaviour, described above, is a fourth.
