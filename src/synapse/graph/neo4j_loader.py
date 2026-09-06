"""Phase C6 wiring: materialise the AuraDB skill graph as a NetworkX graph.

Why materialise rather than query per traversal:

`Matcher` runs one multi-source Dijkstra per candidate over an in-process
`nx.Graph`. Replacing that with per-hop Cypher would change the code Phase B
measured, and C6.4 requires the deployed service to reproduce those numbers. So
Aura is the system of record and this module is the adapter: one query at
startup, then the scoring path is byte-identical to the evaluated one.

The graph is small enough for this to be the obvious choice - 213 skills, 15,459
similarity pairs, ~1.2MB - so the whole thing costs one round trip and a
negligible slice of the 512MB budget (NFR1).

Edge direction: the migration stores each similarity in BOTH directions (30,918
relationships for 15,459 logical pairs). `iter_skill_graph` collapses them to
unordered pairs, and `nx.Graph` is undirected, so the result matches the pickle
exactly - verified pair-for-pair with zero weight differences.
"""

from __future__ import annotations

import logging
import time

import networkx as nx

from .neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)


class GraphParityError(RuntimeError):
    """The loaded graph does not match what the caller expected.

    Raised rather than warned: silently serving a graph of unexpected shape is
    how a deploy ends up producing numbers that do not match the evaluation it
    was validated against.
    """


def load_graph_from_neo4j(
    client: Neo4jClient,
    expected_skills: int | None = None,
    expected_pairs: int | None = None,
    with_roles: bool = True,
) -> nx.Graph:
    """Build the scoring graph from Aura.

    Node/edge attributes mirror `build_graph.py` exactly, because `Matcher`
    selects on `node_type == "skill"` and `relation == "similar"`, and
    `EntityLinker` reads `category` to build its node texts. A rename here would
    silently produce an empty skill graph.

    `expected_*` turn a wrong-shaped graph into a startup failure instead of a
    subtly wrong ranking. Pass the counts the evaluation was run against.
    """
    started = time.monotonic()
    nodes, edges = client.iter_skill_graph()

    G = nx.Graph()
    for node in nodes:
        G.add_node(
            node["name"],
            node_type="skill",
            category=node.get("category") or "",
            embed_category=node.get("embed_category"),
            source=node.get("source") or "onet",
        )

    for edge in edges:
        # Both endpoints are guaranteed to be Skill nodes by the query's own
        # pattern, so no membership check is needed.
        G.add_edge(edge["lo"], edge["hi"], relation="similar", weight=edge["weight"])

    if with_roles:
        # Roles are loaded as nodes first, then edges. Deriving them from the
        # REQUIRES list alone silently drops the two O*NET roles that have no
        # requirements, leaving the graph two nodes short of the pickle.
        for role in client.iter_roles():
            G.add_node(role["name"], node_type="role", soc=role.get("soc"))
        for role, skill in client.iter_role_requirements():
            G.add_edge(role, skill, relation="requires")

    skills = sum(1 for _, d in G.nodes(data=True) if d.get("node_type") == "skill")
    pairs = sum(1 for _, _, d in G.edges(data=True) if d.get("relation") == "similar")
    elapsed = time.monotonic() - started
    logger.info(
        "Loaded graph from Neo4j in %.2fs: %d skills, %d similarity pairs, "
        "%d total nodes, %d total edges", elapsed, skills, pairs,
        G.number_of_nodes(), G.number_of_edges(),
    )

    problems = []
    if expected_skills is not None and skills != expected_skills:
        problems.append(f"expected {expected_skills} skills, loaded {skills}")
    if expected_pairs is not None and pairs != expected_pairs:
        problems.append(f"expected {expected_pairs} similarity pairs, loaded {pairs}")
    if not skills:
        problems.append("graph contains no Skill nodes - wrong database?")
    if problems:
        raise GraphParityError(
            "Neo4j graph does not match expectations: " + "; ".join(problems)
            + ". Compare against data/skill_graph.pkl, and note that "
            "count_edges() reports stored relationships (2x) while this counts "
            "logical pairs - use count_similar_pairs() to compare."
        )

    return G
