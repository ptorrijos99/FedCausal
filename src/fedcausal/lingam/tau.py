"""The tau discriminator that ranks variables by how exogenous they look.

For a linear non-Gaussian model, a variable that causes another leaves an
asymmetry in their joint higher-order cumulants, while the reverse direction
does not. ``tau[i, j]`` measures that asymmetry for the ordered pair ``(i, j)``,
and the row sums rank the variables: the smallest total belongs to the source.
This is the discriminator behind every solver in this subpackage.

References
----------
[1] Chen, W., et al. "Higher Order Cumulants-Based Method for Direct and
    Efficient Causal Discovery." IEEE TNNLS, 2025.
[2] Torrijos, P., et al. "Federated Causal Discovery via Regression-Directed
    Cumulants." Probabilistic Graphical Models (PGM), 2026.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from fedcausal.lingam.cumulants import cross_cumulants4, unbiased_cumulant4

__all__ = ["tau_matrix_3rd", "tau_matrix_4th", "tau_matrix_4th_from_data"]


def tau_matrix_3rd(
    c3: npt.NDArray[np.float64],
    c12: npt.NDArray[np.float64],
    c21: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Return the third-order tau matrix.

    Parameters
    ----------
    c3 : numpy.ndarray
        Marginal third cumulants, shape ``(p,)``.
    c12 : numpy.ndarray
        ``Cum(x_i, x_j, x_j)``, shape ``(p, p)``.
    c21 : numpy.ndarray
        ``Cum(x_i, x_i, x_j)``, shape ``(p, p)``.

    Returns
    -------
    numpy.ndarray
        Shape ``(p, p)``, non-negative, zero on the diagonal. Entry
        ``(i, j)`` is ``|C3(x_i) * C12(x_i, x_j) - C21(x_i, x_j) * C12(x_j, x_i)|``.
    """
    tau = np.abs(c3[:, np.newaxis] * c12 - c21 * c12.T)
    np.fill_diagonal(tau, 0.0)
    return tau


def tau_matrix_4th(
    c4: npt.NDArray[np.float64],
    c13: npt.NDArray[np.float64],
    c31: npt.NDArray[np.float64],
    c22: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Return the fourth-order tau matrix.

    Parameters
    ----------
    c4 : numpy.ndarray
        Marginal fourth cumulants, shape ``(p,)``.
    c13 : numpy.ndarray
        ``Cum(x_i, x_j, x_j, x_j)``, shape ``(p, p)``.
    c31 : numpy.ndarray
        ``Cum(x_i, x_i, x_i, x_j)``, shape ``(p, p)``.
    c22 : numpy.ndarray
        ``Cum(x_i, x_i, x_j, x_j)``, shape ``(p, p)``.

    Returns
    -------
    numpy.ndarray
        Shape ``(p, p)``, non-negative, zero on the diagonal. Entry
        ``(i, j)`` is ``|C4(x_i) * C13(x_i, x_j) - C22(x_i, x_j) * C31(x_i, x_j)|``.

    Notes
    -----
    The fourth-order discriminator is the stronger of the two and is the default
    everywhere in this subpackage. The third-order one matters when the data has
    skew but little excess kurtosis, which is what the solvers' ``adaptive``
    option detects.
    """
    tau = np.abs(c4[:, np.newaxis] * c13 - c22 * c31)
    np.fill_diagonal(tau, 0.0)
    return tau


def tau_matrix_4th_from_data(
    x: npt.NDArray[np.float64],
    columns: list[int] | None = None,
) -> npt.NDArray[np.float64]:
    """Return the fourth-order tau matrix computed directly from a data matrix.

    Parameters
    ----------
    x : numpy.ndarray
        Data, shape ``(n, p)``.
    columns : list of int, optional
        Column indices to restrict to, in the order they should appear in the
        result. Defaults to all columns.

    Returns
    -------
    numpy.ndarray
        Shape ``(len(columns), len(columns))``.

    Notes
    -----
    Provided for single-machine diagnostics. The federated path never calls it:
    it needs the raw data, which is exactly what does not leave a client.
    """
    cols = list(range(x.shape[1])) if columns is None else list(columns)
    size = len(cols)
    c4_local = np.array([unbiased_cumulant4(x[:, c]) for c in cols], dtype=np.float64)
    tau = np.zeros((size, size), dtype=np.float64)
    for i in range(size):
        for j in range(size):
            if i == j:
                continue
            c13, c31, c22 = cross_cumulants4(x[:, cols[i]], x[:, cols[j]])
            tau[i, j] = abs(c4_local[i] * c13 - c22 * c31)
    return tau
