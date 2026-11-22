# Algorithms and options

What each estimator does, what its options change, and how it is made fast. The measurements behind the defaults are in [Benchmarks](benchmarks.md).

## GES

Greedy Equivalence Search (Chickering, 2002) moves between Markov equivalence classes. A forward phase applies the Insert that most improves the score until none does; a backward phase does the same with Delete. The result is a CPDAG: directed edges are compelled, undirected ones are those the data cannot orient.

```python
GES(
    criterion="bdeu",
    ess=1.0,
    expected_parents=1.0,
    penalty_discount=1.0,
    max_parents=None,
    max_t_size=2,
    required=None,
    forbidden=None,
)
```

| Option | Default | What it does |
|---|---|---|
| `criterion` | `"bdeu"` | The score. `"bdeu"` and `"bic"` take categorical data; `"gaussian_bic"` takes continuous data and scores a linear Gaussian model from its covariance matrix. |
| `ess` | `1.0` | Equivalent sample size of the BDeu score. Larger values favour denser graphs. |
| `expected_parents` | `1.0` | Structure prior, added to any criterion: each of the `m = n_vars - 1` possible parents of a node is present with probability `expected_parents / m`, capped at one half. `None` is the uniform prior, plain BDeu or plain BIC. |
| `penalty_discount` | `1.0` | Multiplier of the BIC penalty, `log(n) / 2` per free parameter. |
| `max_parents` | `None` | Most parents an Insert may give a node. |
| `max_t_size` | `2` | Largest set of neighbours an Insert orients into its head. `None` tries all of them, as Chickering defines it, which is exponential in the degree. |

**The structure prior.** Over twelve networks from 5 to 223 variables, three samples each, one expected parent lowers the summed SHD against the true networks from 659 to 523. The gain grows with the network: `win95pts` drops from 80 to 40, `andes` from 104 to 60. Below about 30 variables the two priors give the same structures.

**The subset cap.** Every Insert considers subsets of the head's undirected neighbours. Capping their size at two has returned the same CPDAG as the exhaustive scan on every network compared, `alarm`, `hailfinder`, `win95pts`, `pathfinder` and `andes` among them, while the exhaustive scan took up to three times longer (on `hailfinder`) and grows exponentially with the degree. A cap forfeits Chickering's optimality guarantee in theory, so state which you used in published results.

`fit(X, initial_graph=dag)` starts the search from a DAG instead of the empty graph.

**Model selection.** `score(X)` is the mean log likelihood per row of held-out data under the learned DAG, its parameters learned from the training rows: the Bayesian predictive likelihood with a BDeu prior of equivalent sample size `ess` on categorical data, which is the same for every DAG of the equivalence class, and the maximum-likelihood linear Gaussian model on continuous data. scikit-learn's parameter searches use it, so `GridSearchCV(GES(), {"expected_parents": [0.5, 1, 2]}).fit(frame)` picks the prior on held-out rows. `fedcausal.scores.log_likelihood(dag, train, test)` is the same number for any DAG.

**Background knowledge.** `GES(required=[("smoking", "cancer")], forbidden=[("age", "sex")])` keeps the first pair adjacent and never joins the second; pairs are given by name or column index, and `FedGES` takes the same arguments and holds them in every site's model and in the global one. Knowledge fixes adjacency, not direction: a search over equivalence classes decides directions from the data. The functional form, `fedcausal.search.ges(score, n_vars, ...)`, also takes `candidates`, a set of ordered pairs to restrict the search to, and runs either phase alone.

### How it is made fast

* **Scoring.** BDeu is computed by a compiled kernel that encodes each parent configuration as one integer and counts directly into a table, sorting instead when the table would be too large; configurations are relabelled when their number outgrows a 64-bit code, so any number of parents is exact. Families are cached, and the families an Insert compares share their parents, so they are counted together and in parallel.
* **Evaluating operators.** Operators are evaluated a child at a time. Every candidate parent with the same neighbours in common with the child shares its candidate subsets, so one traversal of the graph decides the path condition for all of them.
* **Re-scoring.** After a step only the children whose neighbourhood changed, or one of whose undirected neighbours did, are re-scored. Validity also depends on paths elsewhere in the graph, so the best operator is re-checked before it is applied, and when nothing improves every child is re-scored once more before the phase stops. The result is a local optimum of the full neighbourhood.
* **Completion.** After each step the graph is completed to its CPDAG with a compiled Dor-Tarsi extension and Chickering's (1995) edge labelling, both over neighbour and parent lists, and only the pairs whose marks changed are written back.

## FedGES

```python
FedGES(
    n_rounds=10,
    criterion="bdeu",
    ess=1.0,
    expected_parents=1.0,
    penalty_discount=1.0,
    max_parents=None,
    max_t_size=2,
    global_fusion=None,
    local_fusion=None,
    refinement=None,
    convergence=None,
    required=None,
    forbidden=None,
    n_jobs=None,
)
```

The score options are GES's, applied at every site; with `criterion="gaussian_bic"` the sites hold continuous data.

Each round every site runs GES on its own data, starting from its current model; the server fuses the sites' DAGs into a global one (`global_fusion`, by default `Consensus(50)`); each site fuses the global DAG into its own (`local_fusion`, by default `ConsensusUnion()`). The rounds stop at `n_rounds` or when `convergence` says so.

### Fusion operators

All of them transform the input DAGs into one shared topological order and combine the transformed parent sets (Puerta et al., 2021), which makes the result acyclic. They differ in how much of the union they keep.

| Operator | Keeps |
|---|---|
| `ConsensusUnion()` | Every transformed edge |
| `Consensus(pct)` | Edges in at least `pct`% of the inputs |
| `MaxFrequency(k)` | Edges in at least `k` inputs |
| `MaxEdges(k)` | The `k` most agreed edges; `k` may also be `"Mean"`, `"Max"` or `"Min"` of the inputs' edge counts |
| `MaxParents(k)` | The most agreed edges, at most `k` parents per node |
| `Intersection()` | Only what every input has |

Anything with a `fuse(dags)` method works in their place.

Votes are counted after the transform, so an edge the transform adds counts as a vote too. The transform adds edges when a site's compelled edges, its v-structures, contradict the shared order; reversible edges cost nothing, since the minimal I-map for an order depends only on the equivalence class. When sites agree on their v-structures this is harmless. When they do not, the added edges are much the same at every site, because the same order induces them, and they pass any threshold. On `andes` split over ten sites of 500 rows, each site learning without a structure prior overfits to about 420 edges against 338 true ones, a third of them contradicting the order; every transformed DAG then has about 3,300 edges, and `Consensus(50)` keeps 3,038, of which 2,455 are in no site's graph. With the default prior the sites learn about 225 edges each, the transformed DAGs grow by a quarter at most, and `Consensus(50)` keeps 223. The round loop raises a `DenseFusionWarning` when a global model has more than twice the sites' mean number of edges, since that is the sign of it.

### Refinement

After fusing, a site keeps the fused model (`refinement=None`), or starts again from its own model and lets GES decide which differences to accept: `"forward"` may add the edges the fused model brought in, `"backward"` may remove the edges it dropped, `"both"` does one then the other.

### Stopping

The default stops once the sites' mean score no longer improves or every site rebuilds a model it built before (`AnyOf(ScoreImprovement(), ModelEquality())`). Any object with `check_convergence(local_models)` works; `AnyOf()` with no criteria never stops early.

### Parallel sites

With `n_jobs`, the sites are spread over worker processes, each of which keeps its sites, and their score caches, for the whole run; only DAGs and scores travel between processes. Starting a worker costs about as much as importing the package, so it pays off once a site's search takes seconds.

## FedRCD

```python
FedRCD(method="deflation", pruning="adaptive_lasso", lambda_selection="bic")
```

See [FedRCD](fedrcd.md).
