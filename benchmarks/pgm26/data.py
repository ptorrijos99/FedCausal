"""Synthetic SEM generator for the PGM 2026 benchmark.

Ports ``PGMSyntheticDataGenerator`` from the paper's ``experiments/pgm26/benchmark.py``
-- the generator the paper's ``main`` and ``stratification`` modes actually call, as
opposed to the older, unused ``utils/data_generation.py`` class of the same repository.
It follows the DirectLiNGAM (2011) benchmark convention: an Erdos-Renyi lower-triangular
mixing matrix with coefficients drawn from ``[-1.5, -0.5] U [0.5, 1.5]``, and one noise
distribution per node drawn from a fixed catalogue.

Left out of this port: the heteroscedastic scale-mixture noise ("`_hetero`" suffix in
the original, used only by its ``hetero`` benchmark mode) and the "poisson"-scale
random-range sampling that mode used. Neither ``main`` nor ``stratification`` -- the two
modes this package reproduces -- exercises them; every node here uses the same unit
noise scale, matching how both modes actually call the original generator.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from fedcausal.graph import DAG

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "ALL_DISTRIBUTIONS",
    "ASYMMETRIC_DISTRIBUTIONS",
    "OTHER_DISTRIBUTIONS",
    "SYMMETRIC_DISTRIBUTIONS",
    "generate_lingam_dataset",
    "true_dag",
]

#: Noise families with zero third-order cumulant (no skew): the third-order
#: tau discriminator that FedRCD/FedISHC/HC rely on has no signal here.
SYMMETRIC_DISTRIBUTIONS: tuple[str, ...] = ("uniform", "laplace", "logistic", "t")

#: Skewed noise families, where the third-order discriminator is informative.
ASYMMETRIC_DISTRIBUTIONS: tuple[str, ...] = ("exponential", "gamma", "pareto")

#: Remaining catalogue entries: not part of the symmetric/asymmetric split the
#: stratification diagnostic uses, but still swept by the main grid.
OTHER_DISTRIBUTIONS: tuple[str, ...] = ("gaussian_cubed", "poisson", "exponential_cubed")

#: The full eleven-distribution catalogue the main grid sweeps (``gaussian`` itself is
#: excluded -- LiNGAM is not identifiable under Gaussian noise, so the original config
#: keeps it out of every aggregate result). "mixed" assigns each node an independent
#: draw from the other ten.
ALL_DISTRIBUTIONS: tuple[str, ...] = (
    SYMMETRIC_DISTRIBUTIONS + ASYMMETRIC_DISTRIBUTIONS + OTHER_DISTRIBUTIONS + ("mixed",)
)

_MIXABLE: tuple[str, ...] = SYMMETRIC_DISTRIBUTIONS + ASYMMETRIC_DISTRIBUTIONS + OTHER_DISTRIBUTIONS


def _noise(rng: np.random.Generator, n_samples: int, dist_type: str) -> npt.NDArray[np.float64]:
    """Draw `n_samples` centred noise values from `dist_type`, unit scale.

    Parameters
    ----------
    rng : numpy.random.Generator
        Source of randomness.
    n_samples : int
        Number of samples to draw.
    dist_type : str
        One of the entries of :data:`_MIXABLE` (i.e. `ALL_DISTRIBUTIONS` minus
        ``"mixed"``, which is resolved to one of these before this function is
        called).

    Returns
    -------
    numpy.ndarray
        Shape ``(n_samples,)``, mean-centred.

    Raises
    ------
    ValueError
        If `dist_type` is not a recognised distribution name.
    """
    if dist_type == "gaussian_cubed":
        noise = rng.normal(0.0, 1.0, n_samples) ** 3
    elif dist_type == "pareto":
        noise = rng.pareto(3.0, n_samples)
    elif dist_type == "logistic":
        noise = rng.logistic(loc=0.0, scale=1.0, size=n_samples)
    elif dist_type == "uniform":
        bound = np.sqrt(3.0)
        noise = rng.uniform(-bound, bound, n_samples)
    elif dist_type == "laplace":
        noise = rng.laplace(loc=0.0, scale=1.0, size=n_samples)
    elif dist_type == "poisson":
        noise = rng.poisson(1.0, n_samples) - 1.0
    elif dist_type == "exponential":
        noise = rng.exponential(1.0, n_samples) - 1.0
    elif dist_type == "t":
        noise = rng.standard_t(5, n_samples)
    elif dist_type == "gamma":
        noise = rng.gamma(shape=2.0, scale=1.0, size=n_samples) - 2.0
    elif dist_type == "exponential_cubed":
        noise = np.power(rng.exponential(1.0, n_samples), 3)
    else:
        raise ValueError(f"unknown noise distribution: {dist_type!r}")
    result: npt.NDArray[np.float64] = noise - np.mean(noise)
    return result


def generate_lingam_dataset(
    rng: np.random.Generator,
    n_samples: int,
    n_vars: int,
    density: float = 0.2,
    dist_type: str = "exponential",
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], tuple[int, ...]]:
    """Generate one synthetic LiNGAM dataset.

    Draws an Erdos-Renyi lower-triangular mixing matrix `b` over the canonical
    order ``0, 1, ..., n_vars - 1`` (edge coefficients uniform in
    ``[-1.5, -0.5] U [0.5, 1.5]``, kept with probability `density`), assigns
    each node an independent noise distribution, and solves ``x = (I - b)^-1
    e`` for the observed data.

    Parameters
    ----------
    rng : numpy.random.Generator
        Source of randomness for both the graph and the noise. Construct with
        ``numpy.random.default_rng(seed)`` for a reproducible cell.
    n_samples : int
        Number of rows to generate.
    n_vars : int
        Number of variables (nodes).
    density : float, optional
        Probability that a given lower-triangular entry becomes an edge.
    dist_type : str, optional
        One of `ALL_DISTRIBUTIONS`. ``"mixed"`` draws each node's family
        independently and uniformly from the other ten.

    Returns
    -------
    x : numpy.ndarray
        Observed data, shape ``(n_samples, n_vars)``.
    b : numpy.ndarray
        The mixing matrix, shape ``(n_vars, n_vars)``. ``b[child, parent]`` is
        the structural coefficient, zero wherever there is no edge -- the same
        convention as :attr:`fedcausal.lingam.RCDResult.b`.
    true_order : tuple[int, ...]
        The causal order used to build `b`: always ``(0, 1, ..., n_vars - 1)``,
        since `b` is constructed lower-triangular in that order.
    """
    b = np.zeros((n_vars, n_vars))
    for i in range(n_vars):
        for j in range(i):
            if rng.random() < density:
                coeff = rng.uniform(0.5, 1.5)
                if rng.random() < 0.5:
                    coeff = -coeff
                b[i, j] = coeff

    e = np.zeros((n_vars, n_samples))
    for i in range(n_vars):
        node_dist = str(rng.choice(np.array(_MIXABLE))) if dist_type == "mixed" else dist_type
        e[i, :] = _noise(rng, n_samples, node_dist)

    # `I - b` is unit lower-triangular by construction (`b[i, j]` is only ever
    # set for `j < i`), so it is always invertible; no singular-matrix retry
    # is needed, unlike the general case.
    x = np.linalg.solve(np.eye(n_vars) - b, e).T

    true_order = tuple(range(n_vars))
    return x, b, true_order


def true_dag(b: npt.NDArray[np.float64]) -> DAG:
    """Build the ground-truth `DAG` from a mixing matrix.

    Parameters
    ----------
    b : numpy.ndarray
        Mixing matrix, shape ``(p, p)``, in the ``b[child, parent]``
        convention produced by :func:`generate_lingam_dataset`.

    Returns
    -------
    DAG
        One edge `parent -> child` per nonzero entry of `b`.
    """
    n_vars = b.shape[0]
    edges: Sequence[tuple[int, int]] = [
        (parent, child)
        for child in range(n_vars)
        for parent in range(n_vars)
        if b[child, parent] != 0.0
    ]
    return DAG.from_edges(n_vars, edges)
