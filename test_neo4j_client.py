"""Test Neo4j client methods (requires running Neo4j instance)."""

from synapse.graph.neo4j_client import Neo4jClient, Neo4jConfig
import numpy as np

# Test configuration
config = Neo4jConfig(
    uri="neo4j://localhost:7687",
    username="neo4j",
    password="testpassword",  # Set this to your Neo4j password
)

client = Neo4jClient(config)

try:
    # Test connection
    with client.session() as session:
        result = session.run("RETURN 1 AS ok")
        print(f"Connection test: {result.single()['ok']}")

    # Test index creation
    client.ensure_indexes(384)
    print("Indexes created/verified")

    # Test skill upsert
    test_embedding = np.random.rand(384).astype(np.float32)
    test_embedding = test_embedding / np.linalg.norm(test_embedding)

    client.upsert_skill(
        name="Test Skill",
        embedding=test_embedding,
        category="Test Category",
        source="test",
    )
    print("Skill upserted")

    # Test find by name
    found = client.find_skill_by_name("Test Skill")
    print(f"Found skill: {found}")

    # Test vector search
    similar = client.vector_search_skills(test_embedding, top_k=5, min_score=0.5)
    print(f"Vector search results: {len(similar)}")

    # Test deduplication check
    canonical, created = client.get_or_create_skill_with_dedup(
        name="New Test Skill",
        embedding=test_embedding,
        category="Test Category",
        min_similarity=0.82,
    )
    print(f"Dedup check: canonical={canonical}, created={created}")

    # Try again with similar embedding (should find the existing one)
    canonical2, created2 = client.get_or_create_skill_with_dedup(
        name="Another Test Skill",
        embedding=test_embedding * 0.99 + np.random.rand(384).astype(np.float32) * 0.01,
        category="Test Category",
        min_similarity=0.82,
    )
    print(f"Dedup check 2: canonical={canonical2}, created={created2}")

    # Cleanup
    with client.session() as session:
        session.run("MATCH (s:Skill {name: $name}) DETACH DELETE s", {"name": "Test Skill"})
        session.run("MATCH (s:Skill {name: $name}) DETACH DELETE s", {"name": canonical})
        session.run("MATCH (s:Skill {name: $name}) DETACH DELETE s", {"name": canonical2})
    print("Cleanup done")

except Exception as e:
    print(f"Error: {e}")
finally:
    client.close()