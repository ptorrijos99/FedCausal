"""The FedRCD estimator."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

from fedcausal._base import Estimator, check_sites, named_adjacency
from fedcausal.data import as_continuous_sites
from fedcausal.lingam.cumulants import cumulants_from_sums, moment_sums
from fedcausal.lingam.rcd import fedrcd, fedrcd_exact

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy.typing as npt
    import pandas as pd
    from typing_extensions import Self

    from fedcausal.data import ContinuousDataset
    from fedcausal.lingam.rcd import LambdaSelection, Pruning, RCDResult

__all__ = ["FedRCD"]

_METHODS = ("deflation", "exact", "ratio", "sort")


class FedRCD(Estimator):
    """Learn a linear non-Gaussian causal model from continuous data held by several sites.

    Each site reduces its rows to additive moment sums; the server turns their
    totals into third- and fourth-order cumulants, from which a tau
    discriminator finds the causal order, and estimates the coefficients on
    it. No row leaves a site.

    Torrijos, Stella, Gámez & Puerta, "Federated Causal Discovery via
    Regression-Directed Cumulants", PGM 2026.

    Parameters
    ----------
    method : {"deflation", "exact", "ratio", "sort"}, default="deflation"
        What the server does after finding a source. ``"deflation"`` removes
        it from the aggregated cumulants with regression coefficients and
        searches again, all in one round of communication. ``"exact"`` has the
        sites deflate their own data instead, one round per variable, so no
        estimation error carries from one step to the next. ``"ratio"``
        deflates with cumulant-ratio coefficients, as FedISHC does.
        ``"sort"`` sorts once by the tau discriminator.
    order : {3, 4}, default=4
        Order of the cumulants behind the tau discriminator, and so of the
        moment sums the sites send. The fourth-order one needs excess
        kurtosis, the third-order one skew. FedISHC is
        ``FedRCD(method="ratio", order=3)`` and FedHC
        ``FedRCD(method="sort", order=3)``; with a single site, HC-LiNGAM is
        ``method="sort"`` and HC ``method="exact"``.
    pruning : {"adaptive_lasso", None}, default="adaptive_lasso"
        How coefficients are estimated on the order. ``None`` is ordinary
        least squares, which keeps every predecessor as a parent.
    lambda_selection : {"bic", "fixed"}, default="bic"
        Penalty selection for the adaptive lasso.

    Attributes
    ----------
    causal_order_ : list of str
        The variables, most exogenous first.
    coef_ : ndarray of shape (n_vars, n_vars)
        ``coef_[child, parent]`` is the coefficient of `parent` in `child`'s
        structural equation.
    dag_ : DAG
        The non-zero coefficients as a DAG.
    adjacency_ : pandas.DataFrame
        `dag_` as a named adjacency matrix, ``A[i, j] = 1`` for ``i -> j``.
    n_features_in_ : int
    feature_names_in_ : ndarray of str
    """

    def __init__(
        self,
        method: Literal["deflation", "exact", "ratio", "sort"] = "deflation",
        order: Literal[3, 4] = 4,
        pruning: Pruning = "adaptive_lasso",
        lambda_selection: LambdaSelection = "bic",
    ) -> None:
        self.method = method
        self.order = order
        self.pruning = pruning
        self.lambda_selection = lambda_selection

    def fit(
        self, X: Sequence[ContinuousDataset | pd.DataFrame | npt.ArrayLike], y: None = None
    ) -> Self:
        """Learn the model from the sites' data.

        Parameters
        ----------
        X : sequence of ContinuousDataset, pandas.DataFrame or array-like
            One table per site, all over the same variables.
        y : None
            Ignored.

        Returns
        -------
        self

        Raises
        ------
        TypeError
            If `X` is a single table rather than one per site.
        ValueError
            If the sites disagree on their variables, hold too few rows
            between them, or a hyperparameter is out of range.
        """
        if self.method not in _METHODS:
            raise ValueError(f"method must be one of {_METHODS}, got {self.method!r}")
        if self.order not in (3, 4):
            raise ValueError(f"order must be 3 or 4, got {self.order!r}")
        check_sites(X)
        sites = as_continuous_sites(X)
        pruning, selection = self.pruning, self.lambda_selection
        result: RCDResult
        if self.method == "exact":
            result = fedrcd_exact(
                [s.values for s in sites],
                order=self.order,
                pruning=pruning,
                lambda_selection=selection,
            )
        else:
            result = fedrcd(
                cumulants_from_sums([moment_sums(s.values, self.order) for s in sites]),
                method=self.method,
                order=self.order,
                pruning=pruning,
                lambda_selection=selection,
            )

        names = np.asarray(sites[0].names, dtype=object)
        self.causal_order_ = [str(names[i]) for i in result.causal_order]
        self.coef_ = result.coef
        self.dag_ = result.to_dag()
        self.n_features_in_ = len(names)
        self.feature_names_in_ = names
        return self

    @property
    def adjacency_(self) -> pd.DataFrame:
        """pandas.DataFrame: The learned DAG as a named adjacency matrix."""
        self._check_fitted("dag_")
        return named_adjacency(self.dag_, self.feature_names_in_)
