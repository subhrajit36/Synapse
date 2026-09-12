#!/usr/bin/env python
"""Phase F0: migrate the rebuilt skill graph into AuraDB. Migration only.

Run it when the local `data/skill_graph.pkl` has been rebuilt and AuraDB needs
to match:

    python scripts/migrate_graph.py --dry-run     # inspect, write nothing
    python scripts/migrate_graph.py               # migrate
    python scripts/migrate_graph.py --verify-only # just compare the two

WHY THIS EXISTS RATHER THAN `src/synapse/graph/migrate_to_neo4j.py`
-------------------------------------------------------------------
That module contains no DELETE. It is pure MERGE, which is correct for a first
migration and wrong for every one after. The similarity edges are the thing that
changes when the graph is rebuilt, and MERGE cannot remove an edge that no
longer exists - so running it against a populated database would UNION the new
edges with the stale ones and leave a graph denser than either.

This script therefore does a targeted replace:

    delete every SIMILAR relationship, then rewrite them from the pickle.

WHAT IT DELIBERATELY DOES NOT TOUCH
-----------------------------------
`Candidate` nodes and their `HAS_SKILL` edges - the ingested candidate pool.
That is safe here because the rebuild changed only the edges *between* skills,
not the skill node set itself, so MERGE updates each Skill's properties in place
and every candidate's links survive. The script verifies that assumption before
writing anything: if the pickle's skill names differ from what is stored, it
stops, because dropping a Skill node would silently take a candidate's skills
with it.

Roles and REQUIRES edges are MERGEd for completeness. They are not used in
scoring - `Matcher` builds a skill-only view - but keeping the stored graph
identical to the artifact means `graph_stats` totals stay comparable.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

# Allow `python scripts/migrate_graph.py` from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import networkx as nx  # noqa: E402

from synapse.graph.neo4j_client import Neo4jClient  # noqa: E402

DEFAULT_PICKLE = "data/skill_graph.pkl"
EMBEDDING_DIM = 384

# AuraDB Free ceilings (NFR4). Checked before writing rather than discovered
# halfway through a partial migration.
MAX_NODES = 200_000
MAX_RELATIONSHIPS = 400_000


# --------------------------------------------------------------------- read


def load_graph(path: str) -> nx.Graph:
    graph_path = Path(path)
    if not graph_path.exists():
        raise SystemExit(
            f"Missing {graph_path}. Build it first:\n"
            "  python -m synapse.graph.build_graph"
        )
    with graph_path.open("rb") as fh:
        return pickle.load(fh)


def summarise(G: nx.Graph) -> dict:
    """Everything the migration needs to know about the artifact."""
    skills = [n for n, d in G.nodes(data=True) if d.get("node_type") == "skill"]
    roles = [n for n, d in G.nodes(data=True) if d.get("node_type") == "role"]

    similar, requires = [], []
    for u, v, data in G.edges(data=True):
        relation = data.get("relation")
        if relation == "similar":
            # Canonical order, so the pair set is comparable with what the
            # database reports back.
            lo, hi = (u, v) if u < v else (v, u)
            similar.append((lo, hi, data.get("weight"),
                            data.get("edge_source", "embedding")))
        elif relation == "requires":
            # A requires edge is role -> skill; nx does not preserve direction.
            role, skill = (u, v) if u in roles else (v, u)
            requires.append((role, skill))

    return {
        "skills": skills,
        "roles": roles,
        "similar": similar,
        "requires": requires,
        # SIMILAR is stored in both directions, so the relationship count is
        # twice the logical pair count. Compare pairs, not relationships.
        "relationship_total": len(similar) * 2 + len(requires),
        "node_total": len(skills) + len(roles),
    }


def check_ceilings(facts: dict) -> None:
    if facts["node_total"] > MAX_NODES:
        raise SystemExit(
            f"{facts['node_total']} nodes exceeds the AuraDB Free limit of {MAX_NODES}."
        )
    if facts["relationship_total"] > MAX_RELATIONSHIPS:
        raise SystemExit(
            f"{facts['relationship_total']} relationships exceeds the AuraDB Free "
            f"limit of {MAX_RELATIONSHIPS}."
        )


# ------------------------------------------------------------------ compare


def report(facts: dict, client: Neo4jClient | None) -> None:
    print("\n--- local artifact ---")
    print(f"  Skill nodes        : {len(facts['skills'])}")
    print(f"  Role nodes         : {len(facts['roles'])}")
    print(f"  SIMILAR pairs      : {len(facts['similar'])}")
    print(f"  REQUIRES edges     : {len(facts['requires'])}")
    print(f"  relationships to write: {facts['relationship_total']} "
          f"(SIMILAR stored both directions)")
    print(f"  free-tier usage    : {facts['node_total']}/{MAX_NODES} nodes, "
          f"{facts['relationship_total']}/{MAX_RELATIONSHIPS} relationships")

    if client is None:
        return

    print("\n--- AuraDB ---")
    nodes = client.count_nodes()
    edges = client.count_edges()
    pairs = client.count_similar_pairs()
    candidates = client.count_candidates()
    print(f"  Skill nodes        : {nodes.get('Skill', 0)}")
    print(f"  Role nodes         : {nodes.get('Role', 0)}")
    print(f"  SIMILAR pairs      : {pairs}   "
          f"(stored as {edges.get('SIMILAR', 0)} relationships)")
    print(f"  REQUIRES edges     : {edges.get('REQUIRES', 0)}")
    print(f"  Candidate nodes    : {candidates}  <- preserved, never touched")

    drift = []
    if nodes.get("Skill", 0) != len(facts["skills"]):
        drift.append("skill count")
    if pairs != len(facts["similar"]):
        drift.append("SIMILAR pairs")
    if edges.get("REQUIRES", 0) != len(facts["requires"]):
        drift.append("REQUIRES")
    print(f"\n  differs from the artifact in: {', '.join(drift) if drift else 'nothing'}")


def assert_skill_sets_match(facts: dict, client: Neo4jClient) -> None:
    """Refuse to migrate if the stored skill names are not the artifact's.

    The whole reason this script can leave `Candidate` nodes alone is that it
    only ever MERGEs skills, never deletes them. If the name sets have diverged,
    some stored Skill is not in the new artifact, and reconciling it would mean
    deleting a node that candidates may point at. That is a decision for a
    person, not for a migration script.
    """
    stored = {row["name"] for row in client.iter_skill_graph()[0]}
    if not stored:
        return  # empty database; nothing to reconcile

    local = set(facts["skills"])
    only_stored = stored - local
    only_local = local - stored
    if not only_stored:
        return

    print("\nABORTING: AuraDB holds Skill nodes absent from the new artifact.")
    print(f"  only in AuraDB ({len(only_stored)}): {sorted(only_stored)[:10]}")
    print(f"  only local     ({len(only_local)}): {sorted(only_local)[:10]}")
    print(
        "\nThis script never deletes Skill nodes, because candidates in the pool\n"
        "link to them. Reconcile deliberately before migrating."
    )
    raise SystemExit(1)


# ------------------------------------------------------------------ migrate


def migrate(facts: dict, client: Neo4jClient, batch_size: int = 500) -> None:
    print("\n--- migrating ---")

    client.ensure_indexes(EMBEDDING_DIM)
    client.ensure_candidate_schema()
    print("  indexes and constraints ensured")

    # 1. Clear the stale similarity edges. This is the step that distinguishes a
    #    re-migration from a first one, and the step migrate_to_neo4j.py lacks.
    with client.session() as session:
        removed = session.run(
            "MATCH ()-[r:SIMILAR]->() DELETE r RETURN count(r) AS n"
        ).single()["n"]
    print(f"  deleted {removed} stale SIMILAR relationships")

    # 2. Skills: MERGE, so properties and embeddings refresh in place and any
    #    HAS_SKILL edge from a candidate survives.
    G = facts["graph"]
    skills = facts["skills"]
    for i, name in enumerate(skills):
        data = G.nodes[name]
        embedding = data.get("embedding")
        client.upsert_skill(
            name=name,
            embedding=embedding if embedding is not None else None,
            category=data.get("category"),
            category_all=data.get("category_all"),
            category_n=data.get("category_n"),
            embed_category=data.get("embed_category"),
            source=data.get("source", "onet"),
        )
        if (i + 1) % batch_size == 0:
            print(f"  skills {i + 1}/{len(skills)}")
    print(f"  merged {len(skills)} Skill nodes")

    # 3. Roles and their requirements.
    for name in facts["roles"]:
        client.upsert_role(name, soc=G.nodes[name].get("soc"))
    print(f"  merged {len(facts['roles'])} Role nodes")

    for i, (role, skill) in enumerate(facts["requires"]):
        client.upsert_requires(role, skill)
        if (i + 1) % batch_size == 0:
            print(f"  requires {i + 1}/{len(facts['requires'])}")
    print(f"  merged {len(facts['requires'])} REQUIRES edges")

    # 4. Similarity, written both ways to match the stored convention that
    #    `iter_skill_graph` and `count_similar_pairs` already expect.
    for i, (a, b, weight, source) in enumerate(facts["similar"]):
        client.upsert_similar(a, b, weight, source)
        client.upsert_similar(b, a, weight, source)
        if (i + 1) % batch_size == 0:
            print(f"  similar {i + 1}/{len(facts['similar'])}")
    print(f"  wrote {len(facts['similar'])} SIMILAR pairs "
          f"({len(facts['similar']) * 2} relationships)")


def verify(facts: dict, client: Neo4jClient) -> bool:
    """Compare on the figures that actually mean something."""
    nodes = client.count_nodes()
    edges = client.count_edges()
    pairs = client.count_similar_pairs()

    checks = [
        ("Skill nodes", nodes.get("Skill", 0), len(facts["skills"])),
        ("Role nodes", nodes.get("Role", 0), len(facts["roles"])),
        ("SIMILAR pairs", pairs, len(facts["similar"])),
        ("REQUIRES edges", edges.get("REQUIRES", 0), len(facts["requires"])),
    ]

    print("\n--- verification ---")
    ok = True
    for label, got, want in checks:
        status = "OK" if got == want else "MISMATCH"
        ok &= got == want
        print(f"  {label:<16} aura={got:<8} local={want:<8} {status}")

    print(f"  {'Candidate nodes':<16} {client.count_candidates()} preserved")
    return ok


# ---------------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Migrate the rebuilt skill graph into AuraDB (migration only)."
    )
    parser.add_argument("--pickle", default=DEFAULT_PICKLE)
    parser.add_argument("--dry-run", action="store_true",
                        help="Inspect and compare; write nothing.")
    parser.add_argument("--verify-only", action="store_true",
                        help="Compare AuraDB against the artifact and exit.")
    parser.add_argument("--batch-size", type=int, default=500,
                        help="Progress reporting interval.")
    args = parser.parse_args(argv)

    G = load_graph(args.pickle)
    facts = summarise(G)
    facts["graph"] = G
    check_ceilings(facts)

    with Neo4jClient() as client:
        print(f"config: {client.config.describe()}")
        connected = client.ping()
        if not connected:
            report(facts, None)
            print(
                "\nNot connected. If port 7687 times out rather than refusing, "
                "outbound Bolt is blocked by the network."
            )
            # A dry run without a database is still useful - it validates the
            # artifact - so only a real migration is an error here.
            return 0 if args.dry_run else 1

        report(facts, client)

        if args.verify_only:
            return 0 if verify(facts, client) else 1

        if args.dry_run:
            print(
                f"\nDry run. Would delete every SIMILAR relationship and write "
                f"{len(facts['similar'])} pairs. Candidate nodes untouched."
            )
            return 0

        assert_skill_sets_match(facts, client)
        migrate(facts, client, batch_size=args.batch_size)

        if not verify(facts, client):
            print("\nMigration finished but verification FAILED - inspect before use.")
            return 1

    print("\nMigration complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
