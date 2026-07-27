# Algorithms

## Greedy equivalence search

`GES` implements Chickering (2002). A forward phase repeatedly applies the
highest-scoring valid Insert until none improves the score, then a backward
phase does the same over Delete. Both operate on CPDAGs, re-closing to a
canonical form after every step, so the search moves through equivalence
classes rather than individual DAGs.

Ties are broken by first-seen, and candidates are scanned in a fixed order, so
two runs on the same data return the same graph. That determinism is not a nice
extra: without it, a federation of five clients would produce different
structures for reasons unrelated to their data.

### The incremental engine

By default `GES` uses the engine in `fedcausal.search.incremental`, which keeps
the best improving operator per candidate pair and re-scores only the pairs
whose neighbourhood changed after each step. It is roughly 2x faster at 37
variables and 13x at 223, which is what makes per-client search affordable when
you federate.

Score deltas depend only on the two endpoints' neighbourhoods, so invalidating
pairs incident to any node whose neighbourhood changed keeps every cached delta
correct. Operator *validity* is not fully local, though: the clique test looks
at edges among a node's neighbours, and the path check traverses the whole
graph. The incremental invalidation is therefore not *proven* equivalent to the
full scan, and is validated empirically instead. A differential test asserts
the two produce byte-identical CPDAGs on every benchmark network.

Pass `incremental=False` for the full scan, which is exact by construction.

### Options that trade optimality for speed

`max_t_size` caps the size of the `T` set the forward phase enumerates, and
defaults to `2`. On all nine networks measured, the cap produced byte-identical
BDeu, edges, SHD, and SMHD to the exhaustive scan while running up to 20x
faster: `hailfinder` went from 40.2 s to 1.86 s. It is not free in theory,
since any finite cap forfeits Chickering's optimality guarantee. Pass
`max_t_size=None` for the exhaustive scan, and say which you used in anything
you publish.

`empty_t_gate` abandons a candidate pair unless it already passes validity and
scores positively at `T = {}`. It defaults to `False`, and unlike the cap it
does cost the optimum. It exists for compatibility, not as a recommendation.

## Structural fusion

Every fusion operator does the same two things: transform all inputs into one
consensus topological order, then union the parent sets. Because every
transformed graph respects the same order, the union cannot contain a cycle.
The transform is a minimal I-map, so it preserves the conditional independences
of its input as far as the shared order allows.

`consensus_union` takes the full union. The budgeted variants rank the
transformed edges by how many clients contain them and add them under a limit:

- `MaxEdges(k)` stops at `k` edges in total.
- `MaxFrequency(k)` keeps edges present in at least `k` clients.
- `Consensus(pct)` keeps edges present in at least `pct` percent of them.
- `MaxParents(k)` applies a per-node cap.
- `Intersection` keeps only what every client has.

They share the `FusionOperator` protocol, so anything with a `fuse(dags)`
method works in their place.

### A replicated bug in MaxParents

`MaxParents(k)` caps the *source* node's parent count, not the target's
in-degree, which is not what the name suggests and not what the operator
intends. This is reproduced deliberately so results stay comparable with the
reference implementation. If you want a genuine in-degree cap, do not use this
operator.

## The federated loop

Each round: every client rebuilds its local model with GES, warm-started from
the model it produced last round; the server checks convergence; the server
fuses the local graphs into a global one and broadcasts it; each client fuses
that into its own model and optionally refines.

Refinement re-runs a search phase restricted to the edges the fused and
pre-fusion models disagree on, which is much cheaper than a full re-search.
`refinement="FES"`, `"BES"`, or `"GES"` selects the phase; the default `"None"`
keeps the fused model as-is.

Convergence criteria are stateful and see all the local models each round:

- `ScoreImprovement` stops when the mean local score stops strictly improving.
- `ModelEquality` stops when every client re-emits a model it has produced
  before.
- `IterationEquality(n)` stops when the current round repeats one of the last
  `n`.
- `MultipleConvergence([...])` stops when any of its members does.
- `NoneConvergence` never stops early.

None of the built-in criteria can fire on the first round, so the loop always
produces at least one global model. A custom criterion that does fire
immediately raises `ConvergedBeforeFusionError` rather than returning nothing.

## The score

`BDeuScore` is the Bayesian Dirichlet equivalent uniform score, decomposable
over families, with an optional structure prior penalising parent count.
Defaults are `sample_prior=1.0` and `structure_prior=0.0`, matching the
reference implementation's effective behaviour.

Local scores are memoised, and that cache carries most of the search's
performance. A `GESLearner` keeps one score object per dataset for exactly this
reason: rebuilding it every round would throw the cache away.

A sweep over the structure prior on five networks, ten sampled datasets each,
found it worth exposing but not worth changing the default. It roughly halves
SHD on `win95pts`, the densest network measured at 150 edges over 76 nodes, and
is neutral or harmful elsewhere: a strong penalty costs about seven SHD on
`hepar2`.

## Compatibility options

Three settings exist to reproduce the reference implementation rather than
because they are good ideas.

`h_pool="bayesfl"` draws the backward phase's `H` from the undirected
neighbours of `y` that are *not* adjacent to `x`, rather than from `NaYX(x, y)`
as Chickering's Definition 13 says. Under that pool the two sets are disjoint,
so the score is insensitive to which subset is chosen, every subset ties, and
the first-seen tie rule keeps the empty one. The backward phase therefore only
ever applies `Delete(x, y, ∅)`. fedcausal implements both pools honestly rather
than hardcoding the shortcut, so the degeneracy falls out of the ordinary
subset scan and can be tested. On the nine networks measured the two pools
agree exactly.

`empty_t_gate=True` and `MaxParents`'s source-cap behaviour are the other two;
both are described above.
