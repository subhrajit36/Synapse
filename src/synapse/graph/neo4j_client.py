"""Phase C2: Neo4j AuraDB client for skill graph operations.

Provides parameterized Cypher queries for:
- Node lookup (skill -> canonical node)
- Shortest-path queries for bridgeable gaps
- Vector similarity search for entity linking
- Dynamic MERGE with deduplication check (C2.5 fix)
"""

from __future__ import annotations

import json
import logging
import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator

import numpy as np
from neo4j import GraphDatabase, Driver, Session
from neo4j.exceptions import Neo4jError

# from synapse.matching.aliases import normalize  # reserved for future use

logger = logging.getLogger(__name__)

# A dense graph makes unbounded variable-length matching pathological: at 68.5%
# density every skill is within 2 hops of every other, so `[:SIMILAR*]` with no
# bound enumerates a combinatorial number of paths and never returns. Traversal
# is always bounded; `max_hops=None` means "use the default", not "unbounded".
DEFAULT_MAX_HOPS = 2
MAX_HOPS_CEILING = 3


@dataclass
class Neo4jConfig:
    """Neo4j connection configuration from environment variables.

    Every field uses `default_factory` so the environment is read when a config
    is *instantiated*, not when this module is imported. Plain
    `os.getenv(...)` defaults are evaluated once at class-definition time, which
    means any caller that loads a `.env` file after importing this module gets
    `localhost` and an empty password with no error - the failure then looks
    like a network problem rather than a configuration one.
    """

    uri: str = field(
        default_factory=lambda: os.getenv("NEO4J_URI", "neo4j://localhost:7687"))
    username: str = field(
        default_factory=lambda: os.getenv("NEO4J_USERNAME", "neo4j"))
    password: str = field(
        default_factory=lambda: os.getenv("NEO4J_PASSWORD", ""))
    database: str = field(
        default_factory=lambda: os.getenv("NEO4J_DATABASE", "neo4j"))
    # Bounded so an unreachable server fails fast. Networks that drop packets to
    # port 7687 (rather than refusing them) otherwise leave the driver hanging
    # for its full default timeout, which turns a graceful fallback into a stall.
    connection_timeout: float = field(
        default_factory=lambda: float(os.getenv("NEO4J_CONNECTION_TIMEOUT", "15")))
    max_retry_time: float = field(
        default_factory=lambda: float(os.getenv("NEO4J_MAX_RETRY_TIME", "10")))

    @property
    def is_configured(self) -> bool:
        """True when a password is present. Cheap pre-flight before dialing."""
        return bool(self.password)

    def describe(self) -> str:
        """Connection details with the password redacted, safe to log."""
        return (f"uri={self.uri} user={self.username} db={self.database} "
                f"password={'set' if self.password else 'MISSING'}")


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
                connection_timeout=self.config.connection_timeout,
                max_transaction_retry_time=self.config.max_retry_time,
            )
        return self._driver

    def close(self) -> None:
        """Close the driver connection."""
        if self._driver is not None:
            self._driver.close()
            self._driver = None

    def __enter__(self) -> "Neo4jClient":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    @contextmanager
    def session(self) -> Iterator[Session]:
        """Context manager for a Neo4j session."""
        driver = self.connect()
        session = driver.session(database=self.config.database)
        try:
            yield session
        finally:
            session.close()

    # ------------------------------------------------------------------ health

    def ping(self) -> bool:
        """Is the database reachable and authenticating? Never raises.

        This is the predicate a caller needs to decide between the remote graph
        and a local fallback, so it must answer rather than propagate. It
        deliberately does no work beyond a round trip.
        """
        if not self.config.is_configured:
            logger.info("Neo4j not configured (%s)", self.config.describe())
            return False
        try:
            with self.session() as session:
                session.run("RETURN 1").consume()
            return True
        except Exception as exc:  # noqa: BLE001 - any failure means "use fallback"
            logger.warning("Neo4j unreachable (%s): %s: %s",
                           self.config.describe(), type(exc).__name__, exc)
            return False

    # ------------------------------------------------------------------ schema

    def ensure_indexes(self, embedding_dim: int = 384) -> None:
        """Create vector index and constraints. Idempotent."""
        # Index options cannot be parameterized in Cypher, so this value is
        # interpolated. Validate it rather than trusting the caller.
        embedding_dim = int(embedding_dim)
        if not 0 < embedding_dim <= 4096:
            raise ValueError(f"Implausible embedding dimension: {embedding_dim}")

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
        # `$embedding` is referenced unconditionally by the query below, so the
        # key must always be present: Cypher raises ParameterMissing for an
        # undeclared parameter, which meant every `upsert_skill(name)` call with
        # no embedding failed outright. COALESCE handles the None case.
        props = {
            "name": name,
            "category": category,
            "category_all": json.dumps(category_all) if category_all else "[]",
            "category_n": category_n,
            "embed_category": embed_category,
            "source": source,
            "embedding": embedding.tolist() if embedding is not None else None,
        }

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
        """Find the cheapest weighted path from any source skill to the target.

        Returns `{distance, hops, via}`, where `via` is the source skill the
        winning path starts from - matching `Matcher._reachability`, which
        reports `path[0]`. Returns None when no path exists within the bounds.

        Three defects are fixed here relative to the original:

        * `relationships(path)[0].startNode.name` is not valid Cypher - a
          relationship has no `startNode` property - so `via` was never
          populated. The start node is `nodes(path)[0]`.
        * `WHERE` was placed before `ORDER BY`/`LIMIT` inside a `WITH`, which is
          a syntax error. Filtering now happens right after the `WITH`, and
          ordering on the `RETURN`.
        * `max_hops=None` produced an unbounded `[:SIMILAR*]`. On a graph this
          dense that does not terminate; traversal is now always bounded.

        Note the match is undirected (`-[:SIMILAR*]-`) so it is correct whether
        edges were stored once or in both directions. Aura currently holds both
        directions, which doubles the branching factor - another reason the hop
        bound is not optional.
        """
        hops = DEFAULT_MAX_HOPS if max_hops is None else int(max_hops)
        if hops < 1:
            raise ValueError(f"max_hops must be >= 1, got {max_hops}")
        hops = min(hops, MAX_HOPS_CEILING)

        # `hops` is interpolated because Cypher cannot parameterize a
        # variable-length bound; it is an int by construction above.
        distance_filter = "WHERE total_distance <= $max_distance" if max_distance is not None else ""
        query = f"""
            MATCH (source:Skill) WHERE source.name IN $sources
            MATCH (target:Skill {{name: $target}})
            MATCH path = (source)-[:SIMILAR*1..{hops}]-(target)
            WITH reduce(d = 0.0, rel IN relationships(path) | d + rel.distance)
                     AS total_distance,
                 length(path) AS hops,
                 nodes(path)[0].name AS via
            {distance_filter}
            RETURN total_distance AS distance, hops, via
            ORDER BY total_distance ASC
            LIMIT 1
        """

        params: dict = {"sources": list(source_skills), "target": target_skill}
        if max_distance is not None:
            params["max_distance"] = max_distance

        with self.session() as session:
            record = session.run(query, params).single()
            return dict(record) if record else None

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
        """Count relationships by type.

        NOTE: this counts stored relationships, not logical edges. SIMILAR is
        migrated in both directions, so a graph of N undirected similarities
        reports 2N here. Use `count_similar_pairs()` to compare against a
        NetworkX edge count.
        """
        with self.session() as session:
            result = session.run("""
                MATCH ()-[r]->()
                RETURN type(r) AS type, count(*) AS count
            """)
            return {r["type"]: r["count"] for r in result}

    def count_similar_pairs(self) -> int:
        """Distinct unordered SIMILAR pairs - the NetworkX-comparable number.

        Comparing `count_edges()["SIMILAR"]` against `G.number_of_edges()` looks
        like a 2x mismatch and invites a needless re-migration. This is the
        figure a parity check should assert on.
        """
        with self.session() as session:
            record = session.run("""
                MATCH (a:Skill)-[:SIMILAR]->(b:Skill)
                WITH CASE WHEN a.name < b.name THEN a.name ELSE b.name END AS lo,
                     CASE WHEN a.name < b.name THEN b.name ELSE a.name END AS hi
                RETURN count(DISTINCT lo + '\\u0000' + hi) AS pairs
            """).single()
            return record["pairs"] if record else 0

    def iter_skill_graph(self) -> tuple[list[dict], list[dict]]:
        """Everything needed to rebuild the NetworkX graph, in two queries.

        Returned rather than streamed because the whole graph is small (213
        skills, 15.5k pairs) and the caller wants it materialised anyway.
        SIMILAR is de-duplicated to unordered pairs here so the caller does not
        have to know how the migration stored direction.
        """
        with self.session() as session:
            nodes = [dict(r) for r in session.run("""
                MATCH (s:Skill)
                RETURN s.name AS name, s.category AS category,
                       s.embed_category AS embed_category, s.source AS source
                ORDER BY s.name
            """)]
            edges = [dict(r) for r in session.run("""
                MATCH (a:Skill)-[r:SIMILAR]->(b:Skill)
                WITH CASE WHEN a.name < b.name THEN a.name ELSE b.name END AS lo,
                     CASE WHEN a.name < b.name THEN b.name ELSE a.name END AS hi,
                     r.weight AS weight
                RETURN lo, hi, max(weight) AS weight
                ORDER BY lo, hi
            """)]
        return nodes, edges


if __name__ == "__main__":
    import logging as _logging

    _logging.basicConfig(level=_logging.INFO, format="%(levelname)s %(message)s")

    with Neo4jClient() as client:
        print(f"config: {client.config.describe()}")
        if not client.ping():
            print("Neo4j unreachable. If the port times out rather than "
                  "refusing, outbound 7687 is likely blocked by the network.")
            raise SystemExit(1)

        print("connected.")
        try:
            print(f"  nodes         : {client.count_nodes()}")
            print(f"  relationships : {client.count_edges()}")
            print(f"  SIMILAR pairs : {client.count_similar_pairs()}  "
                  "(distinct unordered; compare this to NetworkX)")
        except Neo4jError as e:
            print(f"query failed: {e}")
            raise SystemExit(1)