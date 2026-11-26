API reference
=============

Everything below is public. Anything not listed here is an implementation
detail and may change without notice: modules, functions, methods and
attributes whose names start with an underscore, and names a module does not
list in its ``__all__``.

The estimators, :class:`~fedcausal.fedges.FedGES` and
:class:`~fedcausal.lingam.FedRCD`, are also importable from ``fedcausal``
itself. The data, graphs, scores, searches and metrics they use are
causalbase's, and documented there.

Stability
---------

fedcausal follows `Semantic Versioning <https://semver.org/>`_ from 1.0.0:
within a major version, the names below keep their signatures and meaning,
and code written against one release runs on the next. What may change in a
minor release is the exact structure learned when two operators or two
fusions score the same up to rounding, since a faster kernel may sum the same
terms in another order; such changes are listed in the changelog. Scores of a
given structure on given data do not change beyond rounding.

The score objects cache what they compute and are not safe to share between
threads; use one per thread, or processes.

Only the transport, :mod:`fedcausal.flower`, needs the ``federated`` extra.

Data
----

.. automodule:: fedcausal.data

Fusion
------

.. automodule:: fedcausal.fusion

Federated learning
------------------

.. automodule:: fedcausal.fedges

.. automodule:: fedcausal.lingam

Transport
---------

.. automodule:: fedcausal.flower

.. automodule:: fedcausal.flower.fedges

.. automodule:: fedcausal.flower.lingam

Exceptions
----------

.. automodule:: fedcausal.exceptions
