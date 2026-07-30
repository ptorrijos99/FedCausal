"""XMLBIF (``.xbif``) reader for discrete Bayesian networks.

This module parses the XMLBIF v0.3 dialect produced by SamIam (the format
used for the ground-truth networks in the standard network repository) using only
the standard library's :mod:`xml.etree.ElementTree`. It builds a
:class:`DiscreteBN`: a :class:`~fedcausal.graph.DAG` plus, per variable, its
declared outcome labels and conditional probability table.

XMLBIF dialect notes (observed in ``asia.xbif``)
-------------------------------------------------
- Tags are upper-case: ``NETWORK``, ``VARIABLE``, ``NAME``, ``OUTCOME``,
  ``DEFINITION``, ``FOR``, ``GIVEN``, ``TABLE``, ``PROPERTY``.
- A ``VARIABLE`` element declares a variable's name (its ``NAME`` child) and
  its outcome labels, in declaration order, one per ``OUTCOME`` child.
- A ``DEFINITION`` element gives one variable's conditional probability
  table: ``FOR`` names the variable, zero or more ``GIVEN`` elements name
  its parents (in declaration order), and ``TABLE`` holds the flattened
  probabilities as whitespace-separated floats (any mix of spaces and
  newlines; a trailing separator is common).
- The flattened ``TABLE`` reshapes to ``(q, r)`` in row-major (C) order,
  where ``r`` is the variable's own outcome count and ``q`` is the product
  of its parents' cardinalities taken in ``GIVEN`` declaration order (the
  first ``GIVEN`` varies slowest, the last varies fastest) -- confirmed
  against ``asia.xbif``'s ``dysp`` table, which is given ``bronc`` then
  ``either``.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from fedcausal.graph import DAG

if TYPE_CHECKING:
    from pathlib import Path

__all__ = ["DiscreteBN", "load_xbif"]


@dataclass
class DiscreteBN:
    """A discrete Bayesian network loaded from an XMLBIF file.

    Attributes
    ----------
    dag : DAG
        Directed acyclic graph over the variables, indexed in the same
        order as `names`.
    names : list[str]
        Variable names, in the order they were declared in the source
        file. `dag` node `j` corresponds to `names[j]`.
    outcomes : list[list[str]]
        Outcome (level) labels per variable, **in XBIF declaration order**
        (not resorted). ``outcomes[j][k]`` is the label of the `k`-th
        outcome of variable `j`.
    parent_order : list[tuple[int, ...]]
        Parent node indices per variable, in `GIVEN` declaration order.
        ``parent_order[j]`` gives the column order used to enumerate the
        parent configurations that index the rows of ``cpts[j]``.
    cpts : list[np.ndarray]
        Conditional probability tables. ``cpts[j]`` has shape
        ``(q_j, r_j)`` where ``r_j == len(outcomes[j])`` and ``q_j`` is
        the product of the cardinalities of ``parent_order[j]`` (``1``
        when the variable has no parents). Row `i` is the distribution
        over variable `j`'s outcomes conditional on the `i`-th parent
        configuration, enumerated in row-major order of
        ``parent_order[j]`` (the first parent varies slowest). Each row
        sums to 1.
    """

    dag: DAG
    names: list[str]
    outcomes: list[list[str]]
    parent_order: list[tuple[int, ...]]
    cpts: list[np.ndarray]


def _text(elem: ET.Element, tag: str) -> str:
    """Return the stripped text of `elem`'s first `tag` child.

    Parameters
    ----------
    elem : xml.etree.ElementTree.Element
        Parent element to search.
    tag : str
        Child tag name to look for.

    Returns
    -------
    str
        The stripped text content of the first matching child.

    Raises
    ------
    ValueError
        If `elem` has no `tag` child, or that child has no text.
    """
    child = elem.find(tag)
    if child is None or child.text is None:
        raise ValueError(f"expected a <{tag}> element with text inside <{elem.tag}>")
    return child.text.strip()


def load_xbif(path: str | Path) -> DiscreteBN:
    """Parse an XMLBIF (``.xbif``) file into a `DiscreteBN`.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the ``.xbif`` file.

    Returns
    -------
    DiscreteBN
        The parsed network.

    Raises
    ------
    FileNotFoundError
        If `path` does not exist.
    ValueError
        If the file is missing expected elements (a ``<VARIABLE>``
        without a ``<NAME>``, a ``<DEFINITION>`` referencing an unknown
        variable, a ``<TABLE>`` whose flattened value count does not
        match its expected ``(q, r)`` shape, or a ``<TABLE>`` whose rows
        do not sum to 1 after reshaping -- the latter signals an
        ambiguous or corrupt table layout).

    Notes
    -----
    Only the stdlib :mod:`xml.etree.ElementTree` is used; no external XML
    or BIF-parsing library is required.
    """
    path_str = str(path)
    try:
        tree = ET.parse(path_str)  # noqa: S314 - trusted, locally-authored fixture files
    except FileNotFoundError:
        raise
    except OSError as exc:
        raise FileNotFoundError(f"XBIF file not found: {path_str}") from exc

    root = tree.getroot()
    network = root.find("NETWORK")
    if network is None:
        raise ValueError(f"{path_str}: no <NETWORK> element found")

    # --- Pass 1: variables, in declaration order. ---
    names: list[str] = []
    outcomes: list[list[str]] = []
    name_to_index: dict[str, int] = {}
    for var_elem in network.findall("VARIABLE"):
        name = _text(var_elem, "NAME")
        var_outcomes = [
            oc.text.strip() for oc in var_elem.findall("OUTCOME") if oc.text is not None
        ]
        if not var_outcomes:
            raise ValueError(f"{path_str}: variable {name!r} declares no <OUTCOME> entries")
        name_to_index[name] = len(names)
        names.append(name)
        outcomes.append(var_outcomes)

    num_nodes = len(names)
    parent_order: list[tuple[int, ...]] = [() for _ in range(num_nodes)]
    cpts: list[np.ndarray | None] = [None for _ in range(num_nodes)]
    edges: list[tuple[int, int]] = []

    # --- Pass 2: definitions (CPTs), one <DEFINITION> per variable. ---
    for def_elem in network.findall("DEFINITION"):
        for_name = _text(def_elem, "FOR")
        if for_name not in name_to_index:
            raise ValueError(
                f"{path_str}: <DEFINITION><FOR>{for_name}</FOR> references an undeclared variable"
            )
        j = name_to_index[for_name]

        given_names = [g.text.strip() for g in def_elem.findall("GIVEN") if g.text is not None]
        parents: list[int] = []
        for g_name in given_names:
            if g_name not in name_to_index:
                raise ValueError(
                    f"{path_str}: <DEFINITION> for {for_name!r} has <GIVEN>{g_name}</GIVEN>, "
                    "an undeclared variable"
                )
            parents.append(name_to_index[g_name])
        parent_order[j] = tuple(parents)
        for p in parents:
            edges.append((p, j))

        table_elem = def_elem.find("TABLE")
        if table_elem is None or table_elem.text is None:
            raise ValueError(f"{path_str}: <DEFINITION> for {for_name!r} has no <TABLE> entries")
        flat = [float(tok) for tok in table_elem.text.split()]

        r = len(outcomes[j])
        q = 1
        for p in parents:
            q *= len(outcomes[p])

        if len(flat) != q * r:
            raise ValueError(
                f"{path_str}: <TABLE> for {for_name!r} has {len(flat)} entries, "
                f"expected q*r = {q}*{r} = {q * r}"
            )

        table = np.array(flat, dtype=np.float64).reshape((q, r))
        row_sums = table.sum(axis=1)
        if not np.allclose(row_sums, 1.0, atol=1e-6):
            raise ValueError(
                f"{path_str}: <TABLE> for {for_name!r} does not reshape to rows summing to 1 "
                f"under shape (q={q}, r={r}) with parents {[names[p] for p in parents]} "
                f"in declared order; row sums were {row_sums.tolist()} -- table layout is "
                "ambiguous or the file is corrupt"
            )
        cpts[j] = table

    missing = [names[j] for j, cpt in enumerate(cpts) if cpt is None]
    if missing:
        raise ValueError(f"{path_str}: no <DEFINITION> found for variable(s): {missing}")

    dag = DAG.from_edges(num_nodes, edges)
    resolved_cpts: list[np.ndarray] = [cpt for cpt in cpts if cpt is not None]
    return DiscreteBN(
        dag=dag,
        names=names,
        outcomes=outcomes,
        parent_order=parent_order,
        cpts=resolved_cpts,
    )
