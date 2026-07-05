import sys
import pickle
sys.path.insert(0, "src")
from synapse.graph.build_graph import build_skill_graph, add_semantic_edges

OUT = "data/skill_graph.pkl"


def main():
    G = build_skill_graph()          # O*NET -> roles/skills + 'requires' edges
    G = add_semantic_edges(G)        # + embedding-derived skill<->skill edges
    with open(OUT, "wb") as f:
        pickle.dump(G, f)            # serialize the whole graph to disk
    print(f"Saved {G.number_of_nodes()} nodes, {G.number_of_edges()} edges -> {OUT}")


if __name__ == "__main__":
    main()
