# Changelog

This project follows [Semantic Versioning](https://semver.org/). From 1.0.0, what the [API reference](docs/api.rst) lists changes incompatibly only in a major release; the exact structure learned when several score the same, up to rounding, may still change in a minor one.

## 1.0.0 (2026-11-27)

First release. fedcausal learns causal structures from data held by several sites that cannot pool it, and builds on [causalbase](https://github.com/ptorrijos99/causalbase) (0.2 or later), which learns from data held in one place and supplies the datasets, graphs, scores, searches and metrics.

### FedGES

- `FedGES` learns a causal structure from discrete or continuous data spread over sites that exchange only structures and scores (Torrijos, Gámez and Puerta, *Machine Learning*, 2026). Each round every site runs GES, XGES-0 or XGES (`search=`) on its own data with BDeu, BIC or Gaussian BIC, the server fuses the sites' models into a global one, and each site fuses that back into its own, optionally letting GES decide which of the differences to keep (`refinement`).
- The fusion operators of Puerta et al. (2021) in `fedcausal.fusion`: the consensus union and its budgeted variants `MaxEdges`, `MaxFrequency`, `Consensus` and `MaxParents`, plus `Intersection`. A `DenseFusionWarning` flags a fused model far denser than the sites' own, the sign that they disagree on their v-structures.
- Convergence criteria (`ScoreImprovement`, `ModelEquality`, composed with `AnyOf`), required and forbidden adjacencies held through every fusion, a parent cap, `history_` with every local and global model, and `n_jobs` to spread the sites over worker processes.
- `Site` and `run_rounds` run the same loop for those who want to drive the rounds themselves.

### FedRCD

- `FedRCD` learns a linear non-Gaussian model from continuous data spread over sites that exchange only moment sums, centred on each site's own mean and merged exactly by the server (Torrijos, Stella, Gámez and Puerta, PGM 2026). The tau discriminator on the pooled third- or fourth-order cumulants finds the causal order (`order=3` or `4`), and the coefficients come from the covariance, by adaptive lasso or OLS.
- `method` sets what follows each source: `"deflation"` and `"ratio"` remove it from the cumulants in one round, `"exact"` has the sites remove it from their data in `p - 1` rounds, and `"sort"` sorts once. FedISHC, FedHC, HC-LiNGAM and HC (Chen et al., 2025) are settings of the same solvers, and HC-LiNGAM and HC reproduce the original implementation's orders and coefficients.
- The functions behind it, `fedrcd`, `fedrcd_exact`, `moment_sums` and `cumulants_from_sums`, work on the statistics alone. The solvers are causalbase's (`causalbase.lingam`), where `RCD` runs them on one table.

### Running across machines

- `fedcausal.flower` (the `federated` extra; `federated-simulation` adds Ray) runs both families over Flower's Message API, in simulation or against a real SuperLink, with the same results as the in-process loop, round by round. The runnable app in `flower_app/` serves both.
- Each site's data comes through a loader: `NodeConfigLoader` reads the file named in the supernode's own config, and `InMemoryPartitions` serves a list for simulation. Sites send a fingerprint of their variables and the server refuses a federation that disagrees; a failed site stops the round unless `min_clients` allows dropouts.
- `contiguous_split` and `dirichlet_split` simulate sites from one table.
- `FedCausalError` is causalbase's `CausalBaseError`, so one `except` catches every error either package raises on purpose; `SchemaMismatchError`, `ClientFailureError` and `ConvergedBeforeFusionError` are the federation's own.
