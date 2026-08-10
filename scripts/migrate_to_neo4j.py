import os
import pickle
from neo4j import GraphDatabase

# Fetch credentials from environment variables

URI = os.environ.get("NEO4J_URI", "neo4j+s://06064c34.databases.neo4j.io")


try:
    AUTH = ("06064c34", os.environ["NEO4J_PASSWORD"]) ## we have change this 06064c34 username for different graph DB
except KeyError:
    print("Error: NEO4J_PASSWORD environment variable is not set.")
    print("Please set it in your terminal before running this script.")
    exit(1)


def migrate_graph(pkl_path="data/skill_graph.pkl"):
    print(f"Connecting to Neo4j AuraDB at {URI}...")
    
    try:
        with GraphDatabase.driver(URI, auth=AUTH, connection_timeout=30.0) as driver:
            # Verify credentials before executing queries
            driver.verify_connectivity()
            print("Authentication successful! Starting migration...")

            with open(pkl_path, "rb") as f:
                G = pickle.load(f)

            nodes_data = [
                {"name": n, "node_type": d.get("node_type", "skill")}
                for n, d in G.nodes(data=True)
            ]
            
            edges_data = [
                {"source": u, "target": v, "weight": d.get("weight", 1.0), "relation": d.get("relation", "similar")}
                for u, v, d in G.edges(data=True)
            ]

            # 1. Create Unique Constraint
            driver.execute_query("CREATE CONSTRAINT IF NOT EXISTS FOR (s:Skill) REQUIRE s.name IS UNIQUE")
            
            # 2. Initialize Vector Index
            index_query = """
            CREATE VECTOR INDEX skill_embeddings IF NOT EXISTS
            FOR (s:Skill) ON (s.embedding)
            OPTIONS {
                indexConfig: {
                    `vector.dimensions`: 384,
                    `vector.similarity_function`: 'cosine'
                }
            }
            """
            driver.execute_query(index_query)

            # 3. Batch Insert Nodes
            node_query = """
            UNWIND $batch AS node
            MERGE (s:Skill {name: node.name})
            SET s.node_type = node.node_type
            """
            driver.execute_query(node_query, batch=nodes_data)

            # 4. Batch Insert Edges
            edge_query = """
            UNWIND $batch AS edge
            MATCH (source:Skill {name: edge.source})
            MATCH (target:Skill {name: edge.target})
            MERGE (source)-[r:RELATED_TO {type: edge.relation}]->(target)
            SET r.weight = edge.weight
            """
            driver.execute_query(edge_query, batch=edges_data)
            
            print(f"Migration complete! Successfully added {len(nodes_data)} nodes and {len(edges_data)} edges to AuraDB.")

    except Exception as e:
        print(f"Error during migration: {e}")

if __name__ == "__main__":
    migrate_graph()