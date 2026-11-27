.. _api:

#############
API reference
#############

Everything listed here is public. Modules, functions, methods and attributes
whose names start with an underscore, and names a module does not list in its
``__all__``, are implementation details and may change without notice. The
data, graphs, scores, searches and metrics the federated methods use are
causalbase's, and documented in the `causalbase API reference
<https://ptorrijos99.github.io/causalbase/api.html>`_.

Stability
=========

fedcausal follows `semantic versioning <https://semver.org/>`_ from 1.0.0:
within a major version, the names below keep their signatures and meaning,
and code written against one release runs on the next. What may change in a
minor release is the exact structure learned when two operators or two
fusions score the same up to rounding, since a faster kernel may sum the same
terms in another order; such changes are listed in the changelog. Scores of a
given structure on given data do not change beyond rounding.

The score objects cache what they compute and are not safe to share between
threads; use one per thread, or processes.

Only the transport, :mod:`fedcausal.flower`, needs the ``federated`` extra.

Estimators
==========

Both are importable from ``fedcausal`` itself. Hyperparameters go to the
constructor, :meth:`fit` takes one table per site, and the results end in an
underscore.

.. currentmodule:: fedcausal

.. autosummary::
   :toctree: generated/
   :template: estimator.rst

   fedges.FedGES
   lingam.FedRCD

FedGES
======

The rounds of FedGES, run in one process. ``fedges.Search`` names the
searches a site can run (``"ges"``, ``"xges0"``, ``"xges"``) and
``fedges.Refinement`` how a site refines the fused network (``"forward"``,
``"backward"``, ``"both"``).

.. autosummary::
   :toctree: generated/
   :template: class.rst

   fedges.Site
   fedges.SitePool
   fedges.BNModel
   fedges.Convergence
   fedges.ModelEquality
   fedges.ScoreImprovement
   fedges.AnyOf

.. autosummary::
   :toctree: generated/
   :template: function.rst

   fedges.run_rounds
   fedges.default_convergence

Fusion
======

Each operator is a class, which carries its settings, and a function of the
same name in lower case. ``fusion.EdgeBudget`` names the budgets of
:class:`~fedcausal.fusion.MaxEdges` (``"mean"``, ``"max"``, ``"min"``).

.. autosummary::
   :toctree: generated/
   :template: class.rst

   fusion.FusionOperator
   fusion.Consensus
   fusion.ConsensusUnion
   fusion.MaxEdges
   fusion.MaxFrequency
   fusion.MaxParents
   fusion.Intersection

.. autosummary::
   :toctree: generated/
   :template: function.rst

   fusion.consensus
   fusion.consensus_union
   fusion.max_edges
   fusion.max_frequency
   fusion.max_parents
   fusion.intersection
   fusion.edge_frequency
   fusion.alpha_order
   fusion.beta_to_alpha

FedRCD
======

The solvers are pure functions of the moment sums each site sends.
``lingam.MomentSums`` and ``lingam.Cumulants`` are the types of those
statistics, and ``lingam.Pruning`` and ``lingam.LambdaSelection`` name the
options of the coefficient estimator.

.. autosummary::
   :toctree: generated/
   :template: function.rst

   lingam.fedrcd
   lingam.fedrcd_exact
   lingam.moment_sums
   lingam.cumulants_from_sums

.. autosummary::
   :toctree: generated/
   :template: class.rst

   lingam.RCDResult

Sites from one table
====================

.. autosummary::
   :toctree: generated/
   :template: function.rst

   data.contiguous_split
   data.dirichlet_split

Transport
=========

Federation across processes or machines with `Flower <https://flower.ai>`_;
:doc:`deployment` shows how to run it.

.. autosummary::
   :toctree: generated/
   :template: function.rst

   flower.run_flower_fedges
   flower.run_flower_rcd
   flower.fedges.build_client_app
   flower.fedges.build_server_app
   flower.fedges.handle_train_message
   flower.fedges.run_server_loop
   flower.lingam.build_client_app
   flower.lingam.build_server_app
   flower.lingam.handle_train_message
   flower.lingam.run_server_loop
   flower.control_config
   flower.dag_to_record
   flower.record_to_dag
   flower.sums_to_record
   flower.record_to_sums
   flower.names_fingerprint
   flower.schema_fingerprint

.. autosummary::
   :toctree: generated/
   :template: class.rst

   flower.PartitionLoader
   flower.InMemoryPartitions
   flower.NodeConfigLoader

Exceptions
==========

``FedCausalError`` is causalbase's ``CausalBaseError``, so one ``except``
catches every error either package raises on purpose, and
``ConfigurationError``, ``CycleError`` and ``NotFittedError`` are
causalbase's too. The errors and the warning of the federated methods are
fedcausal's own:

.. autosummary::
   :toctree: generated/
   :template: class.rst

   exceptions.ConvergedBeforeFusionError
   exceptions.SchemaMismatchError
   exceptions.ClientFailureError
   exceptions.DenseFusionWarning
