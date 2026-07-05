import pandas as pd
import networkx as nx
from sentence_transformers import SentenceTransformer, util


ONET_DIR = "data/taxonomies/onet/db_30_3_text"

# Generic office/productivity tools: high frequency, low signal. We exclude them.
STOP_SKILLS = {
    "Microsoft Office software", "Microsoft PowerPoint", "Microsoft Word",
    "Microsoft Outlook", "Microsoft Visio", "Microsoft Project",
}

def build_skill_graph(onet_dir: str = ONET_DIR) -> nx.Graph:
    """Build a role<->skill knowledge graph from O*NET (software domain)."""
    occ = pd.read_csv(f"{onet_dir}/Occupation Data.txt", sep="\t")
    sw = pd.read_csv(f"{onet_dir}/Software Skills.txt", sep="\t")

    # 1. Scope to software-domain roles (SOC family 15).
    occ = occ[occ["O*NET-SOC Code"].str.startswith("15-")]
    sw = sw[sw["O*NET-SOC Code"].str.startswith("15-")]

    # 2. Keep only market-relevant tools, minus generic office tools.
    sw = sw[(sw["Hot Technology"] == "Y") | (sw["In Demand"] == "Y")]
    sw = sw[~sw["Workplace Example"].isin(STOP_SKILLS)]

    G = nx.Graph()

    # 3. Add role nodes. node_type lets us distinguish roles from skills later.
    for _, row in occ.iterrows():
        G.add_node(row["Title"], node_type="role", soc=row["O*NET-SOC Code"])

    # 4. Look-up: SOC code -> role title (skills reference roles by code).
    code_to_title = dict(zip(occ["O*NET-SOC Code"], occ["Title"]))

    # 5. Add skill nodes and role--skill edges.
    for _, row in sw.iterrows():
        role = code_to_title.get(row["O*NET-SOC Code"])
        if role is None:
            continue
        skill = row["Workplace Example"]
        G.add_node(skill, node_type="skill", category=row["Element Name"])
        G.add_edge(role, skill, relation="requires")

    return G

def add_semantic_edges(G, model_name="all-MiniLM-L6-v2", k=5, min_sim=0.30):
    """Add skill<->skill 'similar' edges from embedding similarity (top-k per skill)."""
    skills = [n for n, d in G.nodes(data=True) if d["node_type"] == "skill"]

    # Context-enrich each skill with its O*NET category for a sharper embedding.
    texts = [f"{s} ({G.nodes[s]['category']})" for s in skills]

    model = SentenceTransformer(model_name)
    emb = model.encode(texts, convert_to_tensor=True, show_progress_bar=True)
    sim = util.cos_sim(emb, emb)                 # 217x217 pairwise similarity matrix

    for i, skill in enumerate(skills):
        row = sim[i].clone()
        row[i] = -1.0                            # never connect a skill to itself
        top = row.topk(k)                        # the k highest similarities
        for score, j in zip(top.values.tolist(), top.indices.tolist()):
            if score < min_sim:                  # low floor: skip weak links
                continue
            other = skills[j]
            if not G.has_edge(skill, other):
                G.add_edge(skill, other, relation="similar", weight=round(score, 3))
    return G



if __name__ == "__main__":
    G = build_skill_graph()
    G = add_semantic_edges(G)

    roles = [n for n, d in G.nodes(data=True) if d["node_type"] == "role"]
    skills = [n for n, d in G.nodes(data=True) if d["node_type"] == "skill"]
    sim_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("relation") == "similar"]
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"  roles: {len(roles)} | skills: {len(skills)} | skill-skill edges: {len(sim_edges)}")

    # Demo the differentiator: what does the graph think is most like Docker?
    docker_like = [(nbr, d["weight"]) for nbr, d in G["Docker"].items()
                   if d.get("relation") == "similar"]
    print("\nMost similar to Docker:")
    for skill, w in sorted(docker_like, key=lambda x: -x[1]):
        print(f"  {skill:30s} {w}")
