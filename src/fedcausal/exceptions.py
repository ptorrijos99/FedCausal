"""Exceptions raised by fedcausal.

Everything the library raises deliberately derives from :class:`FedCausalError`,
so callers can catch the whole family with one ``except``. Programming errors
(a wrong type, an out-of-range index) still surface as the ordinary built-in
exceptions.
"""

from __future__ import annotations

__all__ = [
    "FedCausalError",
    "ConfigurationError",
    "ConvergedBeforeFusionError",
    "SchemaMismatchError",
    "ClientFailureError",
]


class FedCausalError(Exception):
    """Base class for every error fedcausal raises deliberately."""


class ConfigurationError(FedCausalError, ValueError):
    """An argument or federation setting is invalid.

    Also a :class:`ValueError`, so existing ``except ValueError`` handlers keep
    working.
    """


class ConvergedBeforeFusionError(FedCausalError):
    """The federated loop stopped before it produced any global model.

    Raised when the convergence criterion fires on the first round, or when the
    iteration budget is exhausted without a fusion, leaving no global model to
    return. The built-in criteria never converge on a first sighting, so this
    normally points at a custom :class:`~fedcausal.fedges.Convergence`.
    """


class SchemaMismatchError(FedCausalError):
    """Clients disagree about the variables they are learning over.

    Structures travel between clients as bare edge indices, so every
    participant must agree on variable names, order, and cardinalities.
    Detected up front, because the alternative is a silently wrong result.
    """


class ClientFailureError(FedCausalError):
    """Too few clients returned a usable reply in a federated round."""
