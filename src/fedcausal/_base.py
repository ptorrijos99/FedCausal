"""The estimator base class (causalbase's) and the check that input is one table per site."""

from __future__ import annotations

from causalbase._base import Estimator, NotFittedError, is_frame, named_adjacency

__all__ = ["Estimator", "NotFittedError", "check_sites", "is_frame", "named_adjacency"]


def check_sites(X: object) -> None:
    """Raise a TypeError if `X` is one table rather than a sequence of tables, one per site.

    Iterating a DataFrame yields its column names and a 2-D array its rows, so
    without this check a single table fails later with a confusing message.
    """
    from causalbase.data import ContinuousDataset, DiscreteDataset

    single = (
        is_frame(X)
        or isinstance(X, (ContinuousDataset, DiscreteDataset))
        or (getattr(X, "ndim", None) == 2 and not isinstance(X, (list, tuple)))
    )
    if single:
        raise TypeError(
            f"expected one table per site, got a single {type(X).__name__}; "
            "wrap it in a list to learn from one site"
        )
