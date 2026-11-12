# FedRCD: federated causal discovery over continuous data

`FedRCD` learns a causal order and a linear coefficient matrix from continuous data spread across sites. Each site reduces its rows to a handful of moment sums, the server aggregates those into unbiased cumulant estimates, and a tau discriminator applied to them gives the causal order. FedGES federates by exchanging graphs; this family federates by exchanging sufficient statistics.

```python
from fedcausal import FedRCD

model = FedRCD(method="deflation").fit(sites)   # one array or DataFrame per site
model.causal_order_, model.coef_, model.dag_
```

| `method` | After each source | Rounds |
|---|---|---|
| `"deflation"` (default) | Removes it from the aggregated cumulants with OLS coefficients | 1 |
| `"exact"` | The sites remove it from their own data | `p - 1` |
| `"ratio"` | Removes it from the aggregated cumulants with cumulant-ratio coefficients | 1 |
| `"sort"` | Nothing: one sort by tau | 1 |

`order=4` (default) uses the fourth-order tau discriminator, which needs excess kurtosis; `order=3` the third-order one, which needs skew, and has the sites send only their third-order sums.

Coefficients are estimated on the order by an adaptive lasso with a BIC-selected penalty (`pruning="adaptive_lasso"`), which sets absent edges to zero. `pruning=None` uses ordinary least squares, which keeps every predecessor as a parent, so `dag_` is then complete. The solvers below are also available as functions of the aggregated statistics, which is what the Flower transport calls.

The method assumes a linear, non-Gaussian structural equation model: `x = Bx + e`, with `B` acyclic and each `e_i` non-Gaussian. Non-Gaussianity is what makes the causal order identifiable at all. Under Gaussian noise the joint distribution cannot distinguish a DAG from its Markov-equivalent alternatives, and no solver here can recover the order.

## What actually crosses the network

Never a row of data. Each client computes `compute_local_raw_stats(x, order)` on its own matrix and sends the result; nothing else leaves. With `c = x - mean` the site's rows centred on their own mean, that is the row count `n`, the `(p,)` vector of column means `mean`, and four `(p, p)` matrices of centred moment sums, two with `order=3`:

| Array | What it sums | Its diagonal |
|---|---|---|
| `m11` | Σ cᵢcⱼ | Σ cᵢ² |
| `m21` | Σ cᵢ²cⱼ | Σ cᵢ³ |
| `m31` | Σ cᵢ³cⱼ | Σ cᵢ⁴ |
| `m22` | Σ cᵢ²cⱼ² | Σ cᵢ⁴ |

The server moves each site's sums from the site's mean to the pooled mean, which the binomial expansion of the centred powers does exactly, adds them up (`aggregate_and_compute_cumulants`), and applies bias-corrected k-statistic formulas to get unbiased third- and fourth-order cumulants. Sums about each site's own mean keep their precision wherever the data are located; raw power sums centred only at the server lose every significant digit once the mean is about ten thousand standard deviations from zero. The sums are not a formal privacy guarantee: the fewer rows a site holds, the more they reveal about them, and with a single row they are the row.

Over Flower (`fedcausal.flower.lingam`, needs the `flower` extra) that is one round trip for every method except `"exact"`. The server broadcasts a `train` message, each supernode replies with its mean and moment sums, its row count and a fingerprint of its variable names, and the server checks the fingerprints agree before aggregating.

## Deployment

The client side loads its data through `NodeConfigLoader(ContinuousDataset)`: each supernode reads a CSV path out of its own `node_config` and that file and nothing else. `InMemoryPartitions` is the simulation counterpart, indexing a list of in-memory datasets by `partition-id`, which is what `run_flower_rcd` and `run_flower_rcd_exact` use internally.

Both are the same classes the discrete side uses, so [Deployment](deployment.md) covers this family too, and the runnable app in [`flower_app/`](https://github.com/ptorrijos99/fedcausal/tree/main/flower_app) serves both.

## One-shot versus multi-round

`fedrcd(stats, method=..., order=...)` runs the one-round methods on the aggregated `Cumulants` dict and never touches client data, so anything that can produce that dict can drive it: the in-process aggregator, the Flower transport, a stub in a test. `fedrcd_exact(sites, order=...)` cannot, since each of its rounds needs fresh statistics from each site's *current* data: the sites deflate their own rows by the broadcast source and coefficients and send the moment sums of the residuals.

The deflation of `method="deflation"` is exact in the population limit. On real samples each step operates on already-deflated statistics, so the error in an early deflation carries into every later one. `fedrcd_exact` pays `p - 1` rounds of O(p²) statistics to avoid that drift, which matters most at small `p` or with noise close to Gaussian, where the tau signal is weak to begin with. For clearly non-Gaussian noise at moderate `p` the one-round methods are usually good enough and much cheaper.

## The published baselines

The methods FedRCD is usually compared against are settings of the same two functions, so they share the statistics, the transport and the coefficient estimator, and differ only in what they do with a source:

| Method | Call | Reference |
|---|---|---|
| FedISHC | `fedrcd(stats, method="ratio", order=3)` | Chen et al., "FedISHC: Federated Causal Discovery via Higher-Order Cumulants", 2025 |
| FedHC | `fedrcd(stats, method="sort", order=3)` | the same |
| HC-LiNGAM | `fedrcd(stats, method="sort")` over one site | Chen et al., "Higher Order Cumulants-Based Method for Direct and Efficient Causal Discovery", IEEE TNNLS 2025 |
| HC | `fedrcd_exact([x])` | the same |

HC-LiNGAM and HC give the orders and coefficients of the original implementation, and FedHC its orders, which `tests/lingam/test_reference_implementation.py` checks. FedISHC's original divides `Cum(x_j, x_j, x_s)` by `Cum(x_s, x_s, x_s)`, which estimates the square of the coefficient of `x_s` in `x_j`; `method="ratio"` divides `Cum(x_s, x_s, x_j)` instead, which estimates the coefficient. The arXiv version of the FedISHC paper writes the ratio this way; the AAAI version swaps the arguments, as the original code does. The ratio is set to zero when the source's third cumulant is below 1e-6 and clipped to ±10, as in the original code. Every remaining variable is deflated, where the paper deflates only those it judges dependent on the source, and the coefficient matrix comes from the covariance like every other method's, not from the inverted mixing matrix of the original. FedHC sorts once by the tau row sums, as the original code does; the paper's description also allows recomputing the sums over the remaining variables at each step.

Every call returns the same `RCDResult` (`causal_order`, `b`, `to_dag(threshold)`).
