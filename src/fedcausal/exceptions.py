"""Exceptions raised by fedcausal.

The errors of a federation derive from causalbase's base error, as do a cycle
in a :class:`~causalbase.graph.DAG` and reading an estimator's results before
``fit``, so callers can catch all of them with one ``except``:
:class:`FedCausalError` is that base error under its fedcausal 0.x name. An
invalid argument or input raises :class:`ValueError` (or
:class:`ConfigurationError`, which is one), and a wrong kind of input
:class:`TypeError`.
"""

from __future__ import annotations

from causalbase.exceptions import CausalBaseError, ConfigurationError, CycleError, NotFittedError

__all__ = [
    "FedCausalError",
    "ConfigurationError",
    "ConvergedBeforeFusionError",
    "CycleError",
    "SchemaMismatchError",
    "ClientFailureError",
    "DenseFusionWarning",
    "NotFittedError",
]

FedCausalError = CausalBaseError
"""Base class for every error fedcausal and causalbase raise deliberately."""


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
