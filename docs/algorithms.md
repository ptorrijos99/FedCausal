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
| `h_pool` | `"nayx"` | Which neighbour set the backward phase draws deletion subsets from. `"nayx"` is Chickering Definition 13. | Matching the original Java implementation bit for bit. See [compatibility](#compatibility-options). |
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

## Convergence criteria

| Criterion | Stops when |
|---|---|
| `ScoreImprovement()` | The mean local score stops strictly improving |
| `ModelEquality()` | Every site re-emits a model it has produced before |
| `IterationEquality(n)` | The current round repeats one of the last `n`, or any earlier one if `n` is negative |
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

The structure prior is on by default because it recovers better structure overall. Across twelve networks with five sampled datasets each, the summed mean SMHD is 752 with `structure_prior=1.0` against 944 with `0.0`, and the advantage grows with the network: `win95pts` goes from 125 to 55, `andes` from 199 to 150, `link` from 3098 to 987.

It is not free. Three small networks get worse, most visibly `cancer`, which stops recovering its structure exactly (SMHD 0.6 to 2.0), plus `water` (81 to 87) and `hepar2` (72 to 76). If you work on small networks, or need numbers comparable with published FedGES results, pass `structure_prior=0.0`.

A stronger penalty is not better: `0.001` costs `hepar2` another ten SMHD than `1.0` does, for no gain anywhere except `pathfinder`.

Local scores are memoised, and that cache carries most of the search's performance. `GESLearner` keeps one score object per dataset for exactly that reason.

## Compatibility options

Two settings exist to reproduce the original Java implementation, [BayesFL](https://github.com/ptorrijos99/BayesFL), bit for bit. They are quirks of that codebase rather than part of the published method, so reach for them when you need to match previously published numbers, not because they are the better choice.

`h_pool="bayesfl"` draws deletion subsets from the neighbours of `y` *not* adjacent to `x`, rather than from `NaYX(x, y)` as Chickering's Definition 13 specifies. Those two sets are disjoint, so the score is insensitive to which subset is picked, every subset ties, and the first-seen tie rule keeps the empty one. The backward phase therefore only ever deletes edges with an empty subset. fedcausal implements both pools honestly rather than hardcoding the shortcut, so the degeneracy falls out of the ordinary scan and can be tested. On the networks where the two pools have been compared, up to 223 variables, they agree exactly.

`empty_t_gate=True` is the other. It skips a candidate edge unless it improves with an empty conditioning set, and it demonstrably costs you a worse structure: on `child` it is the whole reason the Java result scores lower.
