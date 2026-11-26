# Algorithms and options

What each estimator does, what its options change, and how it is made fast. The measurements behind the defaults are in [Benchmarks](benchmarks.md).

GES and XGES, which every site runs, and their score options are documented in [causalbase](https://github.com/ptorrijos99/causalbase).

## FedGES

```python
FedGES(
    n_rounds=10,
    criterion="bdeu",
    ess=1.0,
    expected_parents=1.0,
    penalty_discount=1.0,
    max_parents=None,
    max_t_size=None,
    global_fusion=None,
    local_fusion=None,
    refinement=None,
    convergence=None,
    required=None,
    forbidden=None,
    search="ges",
    n_jobs=None,
)
```

The score options are GES's, applied at every site; with `criterion="gaussian_bic"` the sites hold continuous data.

Each round every site runs GES on its own data, starting from its current model (XGES-0 or XGES with `search="xges0"` or `"xges"`); the server fuses the sites' DAGs into a global one (`global_fusion`, by default `Consensus(50)`); each site fuses the global DAG into its own (`local_fusion`, by default `ConsensusUnion()`). The rounds stop at `n_rounds` or when `convergence` says so.

### Fusion operators

All of them transform the input DAGs into one shared topological order and combine the transformed parent sets (Puerta et al., 2021), which makes the result acyclic. They differ in how much of the union they keep.

| Operator | Keeps |
|---|---|
| `ConsensusUnion()` | Every transformed edge |
| `Consensus(pct)` | Edges in at least `pct`% of the inputs |
| `MaxFrequency(k)` | Edges in at least `k` inputs |
| `MaxEdges(k)` | The `k` most agreed edges; `k` may also be `"mean"`, `"max"` (the default) or `"min"` of the inputs' edge counts |
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
FedRCD(method="deflation", order=4, pruning="adaptive_lasso", lambda_selection="bic")
```

See [FedRCD](fedrcd.md).
