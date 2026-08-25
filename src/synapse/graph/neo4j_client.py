"""Phase C2: Neo4j AuraDB client for skill graph operations.

Provides parameterized Cypher queries for:
- Node lookup (skill -> canonical node)
- Shortest-path queries for bridgeable gaps
- Vector similarity search for entity linking
- Dynamic MERGE with deduplication check (C2.5 fix)
"""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from neo4j import GraphDatabase, Driver, Session
from neo4j.exceptions import Neo4jError

# from synapse.matching.aliases import normalize  # reserved for future use


@dataclass
class Neo4jConfig:
    """Neo4j connection configuration from environment variables."""
    uri: str = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
    username: str = os.getenv("NEO4J_USERNAME", "neo4j")
    password: str = os.getenv("NEO4J_PASSWORD", "")
    database: str = os.getenv("NEO4J_DATABASE", "neo4j")


class Neo4jClient:
    """Thin wrapper around Neo4j driver with parameterized queries."""

    def __init__(self, config: Neo4jConfig | None = None):
        self.config = config or Neo4jConfig()
        self._driver: Driver | None = None

    def connect(self) -> Driver:
        """Create and cache the driver connection."""
        if self._driver is None:
            self._driver = GraphDatabase.driver(
                self.config.uri,
                auth=(self.config.username, self.config.password),
            )
        return self._driver

    def close(self) -> None:
        """Close the driver connection."""
        if self._driver is not None:
            self._driver.close()
            self._driver = None

    @contextmanager
    def session(self) -> Iterable[Session]:
        """Context manager for a Neo4j session."""
        driver = self.connect()
        session = driver.session(database=self.config.database)
        try:
            yield session
        finally:
            session.close()

    # ------------------------------------------------------------------ schema

    def ensure_indexes(self, embedding_dim: int = 384) -> None:
        """Create vector index and constraints. Idempotent."""
        with self.session() as session:
            # Unique constraint on skill name
            session.run("""
                CREATE CONSTRAINT skill_name_unique IF NOT EXISTS
                FOR (s:Skill) REQUIRE s.name IS UNIQUE
            """)

            # Vector index for skill embeddings (bge-small-en-v1.5 = 384 dim)
            session.run(f"""
                CREATE VECTOR INDEX skill_embedding IF NOT EXISTS
                FOR (s:Skill) ON (s.embedding)
                OPTIONS {{indexConfig: {{
                    `vector.dimensions`: {embedding_dim},
                    `vector.similarity_function`: 'cosine'
                }}}}
            """)

            # Index on role name for faster lookups
            session.run("""
                CREATE INDEX role_name IF NOT EXISTS
                FOR (r:Role) ON (r.name)
            """)

    # ------------------------------------------------------------------ ingestion

    def upsert_skill(
        self,
        name: str,
        embedding: np.ndarray | None = None,
        category: str | None = None,
        category_all: list[str] | None = None,
        category_n: int | None = None,
        embed_category: str | None = None,
        source: str = "onet",
    ) -> None:
        """Upsert a skill node with properties. Uses parameterized query."""
        props = {
            "name": name,
            "category": category,
            "category_all": json.dumps(category_all) if category_all else "[]",
            "category_n": category_n,
            "embed_category": embed_category,
            "source": source,
        }
        if embedding is not None:
            props["embedding"] = embedding.tolist()

        with self.session() as session:
            session.run("""
                MERGE (s:Skill {name: $name})
                SET s.category = $category,
                    s.category_all = $category_all,
                    s.category_n = $category_n,
                    s.embed_category = $embed_category,
                    s.source = $source,
                    s.embedding = COALESCE($embedding, s.embedding)
            """, props)

    def upsert_role(self, name: str, soc: str | None = None) -> None:
        """Upsert a role node."""
        with self.session() as session:
            session.run("""
                MERGE (r:Role {name: $name})
                SET r.soc = $soc
            """, {"name": name, "soc": soc})

    def upsert_requires(self, role_name: str, skill_name: str) -> None:
        """Create role-requires-skill edge."""
        with self.session() as session:
            session.run("""
                MATCH (r:Role {name: $role_name})
                MATCH (s:Skill {name: $skill_name})
                MERGE (r)-[:REQUIRES]->(s)
            """, {"role_name": role_name, "skill_name": skill_name})

    def upsert_similar(self, skill_a: str, skill_b: str, weight: float, edge_source: str = "embedding") -> None:
        """Create skill-similar-skill edge with weight."""
        with self.session() as session:
            session.run("""
                MATCH (a:Skill {name: $a})
                MATCH (b:Skill {name: $b})
                MERGE (a)-[rel:SIMILAR]->(b)
                SET rel.weight = $weight,
                    rel.edge_source = $edge_source,
                    rel.distance = 1.0 - $weight
            """, {"a": skill_a, "b": skill_b, "weight": weight, "edge_source": edge_source})

    # ------------------------------------------------------------------ queries

    def find_skill_by_name(self, name: str) -> dict | None:
        """Exact lookup of a skill node by canonical name."""
        with self.session() as session:
            result = session.run("""
                MATCH (s:Skill {name: $name})
                RETURN s.name AS name, s.category AS category, s.embedding AS embedding
            """, {"name": name})
            record = result.single()
            return dict(record) if record else None

    def vector_search_skills(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        min_score: float = 0.60,
    ) -> list[dict]:
        """Vector similarity search for entity linking.

        Returns nodes with score >= min_score, ordered by similarity desc.
        """
        with self.session() as session:
            result = session.run("""
                CALL db.index.vector.queryNodes('skill_embedding', $top_k, $query)
                YIELD node, score
                WHERE score >= $min_score
                RETURN node.name AS name, node.category AS category, score
                ORDER BY score DESC
            """, {
                "top_k": top_k,
                "query": query_embedding.tolist(),
                "min_score": min_score,
            })
            return [dict(r) for r in result]

    def shortest_path_distance(
        self,
        source_skills: list[str],
        target_skill: str,
        max_hops: int | None = None,
        max_distance: float | None = None,
    ) -> dict | None:
        """Find shortest weighted path from any source skill to target.

        Returns dict with distance, hops, and via (first hop).
        Uses Dijkstra via GDS if available, otherwise falls back to Cypher.
        """
        # Build the variable-length pattern with optional bounds
        hops_clause = f"..{max_hops}" if max_hops else ""
        distance_clause = ""
        if max_distance is not None:
            distance_clause = "WHERE total_distance <= $max_distance"

        query = f"""
            MATCH (source:Skill)
            WHERE source.name IN $sources
            MATCH (target:Skill {{name: $target}})
            MATCH path = (source)-[:SIMILAR*{hops_clause}]-(target)
            WITH path,
                 reduce(dist = 0.0, rel IN relationships(path) | dist + rel.distance) AS total_distance,
                 length(path) AS hops,
                 relationships(path)[0].startNode.name AS via
            {distance_clause}
            ORDER BY total_distance ASC
            LIMIT 1
            RETURN total_distance AS distance, hops, via
        """

        params = {"sources": source_skills, "target": target_skill}
        if max_distance is not None:
            params["max_distance"] = max_distance

        with self.session() as session:
            result = session.run(query, params)
            record = result.single()
            if record:
                return {
                    "distance": record["distance"],
                    "hops": record["hops"],
                    "via": record["via"],
                }
            return None

    def get_skill_neighbors(self, skill_name: str, min_weight: float = 0.3) -> list[dict]:
        """Get direct similar neighbors of a skill."""
        with self.session() as session:
            result = session.run("""
                MATCH (s:Skill {name: $name})-[r:SIMILAR]-(n:Skill)
                WHERE r.weight >= $min_weight
                RETURN n.name AS name, r.weight AS weight, r.edge_source AS source
                ORDER BY r.weight DESC
            """, {"name": skill_name, "min_weight": min_weight})
            return [dict(r) for r in result]

    def skill_exists(self, name: str) -> bool:
        """Check if a skill node exists."""
        with self.session() as session:
            result = session.run("""
                MATCH (s:Skill {name: $name})
                RETURN count(s) > 0 AS exists
            """, {"name": name})
            return result.single()["exists"]

    def get_or_create_skill_with_dedup(
        self,
        name: str,
        embedding: np.ndarray,
        category: str | None = None,
        min_similarity: float = 0.82,
    ) -> tuple[str, bool]:
        """C2.5 fix: Get existing similar skill or create new one.

        Before MERGEing a new skill, vector-search for similar existing nodes.
        If a node with similarity >= min_similarity exists, return that node's name.
        Otherwise, create the new node and return its name.

        Returns: (canonical_name, created_new)
        """
        # First check exact match
        existing = self.find_skill_by_name(name)
        if existing:
            return existing["name"], False

        # Vector search for near-duplicates
        similar = self.vector_search_skills(embedding, top_k=1, min_score=min_similarity)
        if similar:
            # Found a sufficiently similar existing node - use it instead
            return similar[0]["name"], False

        # No similar node exists - create new
        self.upsert_skill(name, embedding=embedding, category=category, source="dynamic")
        return name, True

    # ------------------------------------------------------------------ diagnostics

    def count_nodes(self) -> dict:
        """Count nodes by label."""
        with self.session() as session:
            result = session.run("""
                MATCH (n)
                RETURN labels(n)[0] AS label, count(*) AS count
            """)
            return {r["label"]: r["count"] for r in result}

    def count_edges(self) -> dict:
        """Count edges by type."""
        with self.session() as session:
            result = session.run("""
                MATCH ()-[r]->()
                RETURN type(r) AS type, count(*) AS count
            """)
            return {r["type"]: r["count"] for r in result}


if __name__ == "__main__":
    # Quick connection test
    client = Neo4jClient()
    try:
        with client.session() as session:
            result = session.run("RETURN 1 AS ok")
            print(f"Neo4j connection: {result.single()['ok']}")
    except Neo4jError as e:
        print(f"Neo4j connection failed: {e}")
    finally:
        client.close()