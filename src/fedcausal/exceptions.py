"""Exceptions raised by fedcausal.

The errors of a federation, and reading an estimator's results before
``fit``, derive from :class:`FedCausalError`, so callers can catch them with
one ``except``. An invalid argument or input raises :class:`ValueError` (or
:class:`ConfigurationError`, which is one), a wrong kind of input
:class:`TypeError`, and a cycle in a :class:`~fedcausal.graph.DAG`
:class:`~fedcausal.graph.CycleError`, also a :class:`ValueError`.
"""

from __future__ import annotations

__all__ = [
    "FedCausalError",
    "ConfigurationError",
    "ConvergedBeforeFusionError",
    "SchemaMismatchError",
    "ClientFailureError",
    "DenseFusionWarning",
    "NotFittedError",
]


class FedCausalError(Exception):
    """Base class for every error fedcausal raises deliberately."""


class ConfigurationError(FedCausalError, ValueError):
    """An argument or federation setting is invalid.

    Also a :class:`ValueError`, so ``except ValueError`` catches it too.
    """


class NotFittedError(FedCausalError, AttributeError):
    """An estimator's result was read before ``fit`` was called.

    Also an :class:`AttributeError`, so ``hasattr`` and scikit-learn's
    checks treat an unfitted result as missing.
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


class DenseFusionWarning(UserWarning):
    """A fused global model came out far denser than the sites' own models.

    The consensus operators transform every input into one shared order, and
    the edges that transformation adds are much the same at every site, so they
    pass any vote threshold. It happens when the sites disagree on their
    v-structures, typically because each overfits a small share of the data.
    The rounds then slow down as every site searches back from a dense graph.
    """
