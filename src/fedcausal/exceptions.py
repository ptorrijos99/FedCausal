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

    Also a :class:`ValueError`, so ``except ValueError`` catches it too.
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

    Structures and statistics travel as bare column indices, so every site must
    agree on the variables, their order and, for discrete data, their levels.
    """


class ClientFailureError(FedCausalError):
    """Too few clients returned a usable reply in a federated round."""
