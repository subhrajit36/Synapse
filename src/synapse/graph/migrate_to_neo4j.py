"""Phase C2: One-time migration script from NetworkX pickle to Neo4j AuraDB.

Reads data/skill_graph.pkl, creates:
- (:Skill {name, embedding, category, ...}) nodes
- (:Role {name, soc}) nodes
- [:REQUIRES] edges (role -> skill)
- [:SIMILAR {weight, distance, edge_source}] edges (skill <-> skill)
- Vector index on Skill.embedding (384 dim for bge-small-en-v1.5)

Includes pre-migration count check against AuraDB Free tier limits:
- 200k nodes / 400k relationships ceiling

Usage:
    python -m synapse.graph.migrate_to_neo4j [--dry-run]
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import networkx as nx
import numpy as np

from synapse.graph.build_graph import build_skill_graph, add_semantic_edges, add_seed_edges
from synapse.graph.neo4j_client import Neo4jClient, Neo4jConfig


# AuraDB Free tier limits
MAX_NODES = 200_000
MAX_RELATIONSHIPS = 400_000

EMBEDDING_DIM = 384  # bge-small-en-v1.5


def load_graph(pickle_path: str = "data/skill_graph.pkl") -> nx.Graph:
    """Load the skill graph from pickle, or rebuild if not found."""
    path = Path(pickle_path)
    if path.exists():
        print(f"Loading graph from {pickle_path}...")
        with open(path, "rb") as f:
            G = pickle.load(f)
        print(f"Loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        return G

    print("Pickle not found, rebuilding graph from O*NET...")
    G = add_seed_edges(add_semantic_edges(build_skill_graph()))
    return G


def count_graph_elements(G: nx.Graph) -> dict:
    """Count nodes and edges by type for pre-migration validation."""
    skill_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "skill"]
    role_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "role"]

    requires_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("relation") == "requires"]
    similar_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("relation") == "similar"]

    return {
        "skill_nodes": len(skill_nodes),
        "role_nodes": len(role_nodes),
        "total_nodes": len(skill_nodes) + len(role_nodes),
        "requires_edges": len(requires_edges),
        "similar_edges": len(similar_edges),
        "total_edges": len(requires_edges) + len(similar_edges),
    }


def validate_limits(counts: dict) -> bool:
    """Check if graph fits within AuraDB Free tier limits."""
    ok = True
    if counts["total_nodes"] > MAX_NODES:
        print(f"ERROR: {counts['total_nodes']} nodes exceeds AuraDB Free limit of {MAX_NODES}")
        ok = False
    if counts["total_edges"] > MAX_RELATIONSHIPS:
        print(f"ERROR: {counts['total_edges']} edges exceeds AuraDB Free limit of {MAX_RELATIONSHIPS}")
        ok = False

    if ok:
        print(f"✓ Pre-migration check passed:")
        print(f"  Nodes: {counts['total_nodes']} / {MAX_NODES} ({counts['total_nodes']/MAX_NODES:.1%})")
        print(f"  Edges: {counts['total_edges']} / {MAX_RELATIONSHIPS} ({counts['total_edges']/MAX_RELATIONSHIPS:.1%})")
    return ok


def extract_embeddings(G: nx.Graph) -> dict[str, np.ndarray]:
    """Extract embeddings from skill nodes if present, otherwise return empty dict."""
    embeddings = {}
    for node in G.nodes:
        if G.nodes[node].get("node_type") == "skill" and "embedding" in G.nodes[node]:
            emb = G.nodes[node]["embedding"]
            if isinstance(emb, np.ndarray):
                embeddings[node] = emb
            elif isinstance(emb, list):
                embeddings[node] = np.array(emb, dtype=np.float32)
    return embeddings


def migrate_graph(
    G: nx.Graph,
    client: Neo4jClient,
    dry_run: bool = False,
    batch_size: int = 100,
) -> dict:
    """Migrate the graph to Neo4j."""
    stats = {
        "skills_created": 0,
        "roles_created": 0,
        "requires_created": 0,
        "similar_created": 0,
        "errors": 0,
    }

    skill_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "skill"]
    role_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "role"]

    # Check for pre-computed embeddings
    embeddings = extract_embeddings(G)
    has_embeddings = len(embeddings) > 0

    if not has_embeddings:
        print("WARNING: No embeddings found in graph nodes. Will need to compute them.")
        print("Consider running add_semantic_edges with model that stores embeddings,")
        print("or compute embeddings separately before migration.")

    # Ensure indexes exist
    if not dry_run:
        print("Creating indexes and constraints...")
        client.ensure_indexes(EMBEDDING_DIM)

    # --- Migrate Skill nodes ---
    print(f"\nMigrating {len(skill_nodes)} skill nodes...")
    for i, skill in enumerate(skill_nodes):
        if i % batch_size == 0:
            print(f"  Progress: {i}/{len(skill_nodes)}")

        node_data = G.nodes[skill]
        embedding = embeddings.get(skill) if has_embeddings else None

        if not dry_run:
            try:
                client.upsert_skill(
                    name=skill,
                    embedding=embedding,
                    category=node_data.get("category"),
                    category_all=node_data.get("category_all"),
                    category_n=node_data.get("category_n"),
                    embed_category=node_data.get("embed_category"),
                    source=node_data.get("source", "onet"),
                )
                stats["skills_created"] += 1
            except Exception as e:
                print(f"  ERROR creating skill {skill}: {e}")
                stats["errors"] += 1
        else:
            stats["skills_created"] += 1

    # --- Migrate Role nodes ---
    print(f"\nMigrating {len(role_nodes)} role nodes...")
    for i, role in enumerate(role_nodes):
        if i % batch_size == 0:
            print(f"  Progress: {i}/{len(role_nodes)}")

        node_data = G.nodes[role]

        if not dry_run:
            try:
                client.upsert_role(role, soc=node_data.get("soc"))
                stats["roles_created"] += 1
            except Exception as e:
                print(f"  ERROR creating role {role}: {e}")
                stats["errors"] += 1
        else:
            stats["roles_created"] += 1

    # --- Migrate REQUIRES edges (role -> skill) ---
    requires_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("relation") == "requires"]
    print(f"\nMigrating {len(requires_edges)} REQUIRES edges...")
    for i, (role, skill) in enumerate(requires_edges):
        if i % batch_size == 0:
            print(f"  Progress: {i}/{len(requires_edges)}")

        if not dry_run:
            try:
                client.upsert_requires(role, skill)
                stats["requires_created"] += 1
            except Exception as e:
                print(f"  ERROR creating REQUIRES {role} -> {skill}: {e}")
                stats["errors"] += 1
        else:
            stats["requires_created"] += 1

    # --- Migrate SIMILAR edges (skill <-> skill) ---
    similar_edges = [(u, v, d) for u, v, d in G.edges(data=True) if d.get("relation") == "similar"]
    print(f"\nMigrating {len(similar_edges)} SIMILAR edges...")
    for i, (a, b, d) in enumerate(similar_edges):
        if i % batch_size == 0:
            print(f"  Progress: {i}/{len(similar_edges)}")

        weight = d.get("weight", 0.0)
        edge_source = d.get("edge_source", "embedding")

        if not dry_run:
            try:
                client.upsert_similar(a, b, weight, edge_source)
                # Also create reverse edge for undirected traversal
                client.upsert_similar(b, a, weight, edge_source)
                stats["similar_created"] += 2  # bidirectional
            except Exception as e:
                print(f"  ERROR creating SIMILAR {a} <-> {b}: {e}")
                stats["errors"] += 1
        else:
            stats["similar_created"] += 2

    return stats


def verify_migration(client: Neo4jClient) -> dict:
    """Verify the migration by counting nodes and edges in Neo4j."""
    print("\nVerifying migration...")
    node_counts = client.count_nodes()
    edge_counts = client.count_edges()

    print("Node counts:")
    for label, count in node_counts.items():
        print(f"  {label}: {count}")

    print("Edge counts:")
    for etype, count in edge_counts.items():
        print(f"  {etype}: {count}")

    return {"nodes": node_counts, "edges": edge_counts}


def main():
    parser = argparse.ArgumentParser(description="Migrate skill graph to Neo4j AuraDB")
    parser.add_argument("--dry-run", action="store_true", help="Validate without writing to Neo4j")
    parser.add_argument("--pickle", default="data/skill_graph.pkl", help="Path to skill graph pickle")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild graph from O*NET")
    args = parser.parse_args()

    # Load or rebuild graph
    if args.rebuild:
        G = add_seed_edges(add_semantic_edges(build_skill_graph()))
    else:
        G = load_graph(args.pickle)

    # Pre-migration validation
    counts = count_graph_elements(G)
    print("\n--- Pre-migration counts ---")
    for k, v in counts.items():
        print(f"  {k}: {v}")

    if not validate_limits(counts):
        print("\nMigration aborted: exceeds AuraDB Free tier limits.")
        sys.exit(1)

    if args.dry_run:
        print("\nDry run complete. No data written to Neo4j.")
        return

    # Connect to Neo4j
    config = Neo4jConfig()
    if not config.password:
        print("ERROR: NEO4J_PASSWORD environment variable not set.")
        print("Set NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, NEO4J_DATABASE")
        sys.exit(1)

    client = Neo4jClient(config)
    try:
        # Test connection
        with client.session() as session:
            session.run("RETURN 1")
        print("✓ Connected to Neo4j")
    except Exception as e:
        print(f"ERROR: Failed to connect to Neo4j: {e}")
        sys.exit(1)

    # Migrate
    print("\n--- Starting migration ---")
    stats = migrate_graph(G, client, dry_run=False)

    print("\n--- Migration complete ---")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    # Verify
    verify_migration(client)

    client.close()


if __name__ == "__main__":
    main()