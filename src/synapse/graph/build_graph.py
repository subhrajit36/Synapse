import pandas as pd
import networkx as nx
from collections import Counter, defaultdict
import numpy as np

# Use FastEmbed for ONNX CPU-only embeddings (Phase C1)
# Falls back to sentence-transformers for dev if fastembed not available
try:
    from fastembed import TextEmbedding
    FASTEMBED_AVAILABLE = True
except ImportError:
    FASTEMBED_AVAILABLE = False

ONET_DIR = "data/taxonomies/onet/db_30_3_text"

# Generic office/productivity tools: high frequency, low signal. We exclude them.
STOP_SKILLS = {
    "Microsoft Office software", "Microsoft PowerPoint", "Microsoft Word",
    "Microsoft Outlook", "Microsoft Visio", "Microsoft Project",
}

# Generic O*NET category *descriptors* that arrive as "Workplace Example" values
# but are taxonomy buckets, not tools (e.g. the real skills are MySQL, Git, React
# -- not "...database management system software"). The edge-construction audit
# (data/eval/edge_construction_audit.md) showed these become high-degree hubs that
# are "similar" to everything: "Relational database management system software"
# (sim-degree 19) out-scored the MySQL<->PostgreSQL pair itself. Since scoring
# bridges via shortest path over 'similar' edges, such a hub is a low-distance
# shortcut that makes unrelated skills look falsely bridgeable -- it corrupts the
# exact signal Phase B must validate (and violates NFR6 traceability). Excluded.
GENERIC_SKILLS = {
    "Relational database management system software",
    "Web application software",
    "Web application framework software",
    "Web server software",
    "JavaScript framework software",
    "Statistical software",
    "Software development tools",
    "Vector database software",
    "Version control software",
    "Content management systems CMS",
    "Firewall software",
    "Chatbot software",
    "Operational Data Store ODS software",
    "Geographic information system GIS software",
    "Geographic information system GIS systems",
}

# --- Custom Seed List for Modern Tech ---
CUSTOM_SKILLS = [
    # Deep Learning & ML
    {"skill": "PyTorch", "category": "Deep Learning Framework"},
    {"skill": "TensorFlow", "category": "Deep Learning Framework"},
    {"skill": "Keras", "category": "Deep Learning Framework"},
    {"skill": "XGBoost", "category": "Gradient Boosting Library"},
    {"skill": "CatBoost", "category": "Gradient Boosting Library"},
    {"skill": "LightGBM", "category": "Gradient Boosting Library"},
    {"skill": "Optuna", "category": "Hyperparameter Optimization"},

    # NLP & LLMs
    {"skill": "LangChain", "category": "LLM Orchestration"},
    {"skill": "Llama", "category": "Large Language Model"},
    {"skill": "Mistral", "category": "Large Language Model"},
    {"skill": "BERT", "category": "Transformer Model"},
    {"skill": "DistilBERT", "category": "Transformer Model"},
    {"skill": "Hugging Face", "category": "AI Ecosystem"},

    # Frameworks & App Dev
    {"skill": "FastAPI", "category": "Web Framework"},
    {"skill": "React Native", "category": "Mobile Development"},
]

# `category` (O*NET Element Name) is preserved for provenance; `embed_category`
# is the normalized vocabulary used only to build the embedding string.
CATEGORY_OVERRIDES = {item["skill"]: item["category"] for item in CUSTOM_SKILLS}

# --- Curated substitutability edges ---
# Pairs the embedder already finds unaided have been REMOVED from this list.
# The diagnostic showed PyTorch->TensorFlow at 0.736 and PyTorch->Keras at 0.673,
# both at rank #1-2 once the embedding vocabulary was normalized. Seeding them
# added nothing and would have let a hand-typed constant masquerade as a
# discovery in the Phase B tables.
#
# Everything below must justify itself the same way. Run the module and delete
# any pair reported as `already found` - if the embedder got there, the seed is
# dead weight.
#
# RULE: seeds fill holes. They never overwrite a measurement (see add_seed_edges).
SEED_SIMILAR_EDGES = [
    ("XGBoost", "LightGBM", 0.90),
    ("XGBoost", "CatBoost", 0.88),
    ("LightGBM", "CatBoost", 0.88),
    ("BERT", "DistilBERT", 0.93),
    ("Llama", "Mistral", 0.85),
    ("BERT", "Hugging Face", 0.70),
    ("Hugging Face", "PyTorch", 0.70),
    ("LangChain", "Hugging Face", 0.68),
    ("FastAPI", "Flask", 0.80),
    ("FastAPI", "Django", 0.72),
    ("React Native", "React", 0.85),
]


def pick_category(counter: Counter) -> str:
    """Choose one Element Name for a skill, deterministically.

    --- FIX: row-order dependence ---
    The previous loop ran `G.add_node(skill, category=row["Element Name"])` on
    every matching row, so the surviving category was whichever SOC row happened
    to come last in the CSV. That category is half of the embedding string, so
    every edge weight in the graph depended on file ordering - an NFR7 violation
    sitting underneath the whole scoring layer. It is also why PyTorch surfaced
    as 'Data base user interface and query software'.

    Rule: most frequent Element Name across the skill's occupations, ties broken
    alphabetically. Order-independent, so a rebuild always yields the same graph.
    """
    return min(counter.items(), key=lambda kv: (-kv[1], kv[0]))[0]


def build_skill_graph(onet_dir: str = ONET_DIR, verbose: bool = True) -> nx.Graph:
    """Build a role<->skill knowledge graph from O*NET (software domain)."""
    occ = pd.read_csv(f"{onet_dir}/Occupation Data.txt", sep="\t")
    sw = pd.read_csv(f"{onet_dir}/Software Skills.txt", sep="\t")

    # 1. Scope to software-domain roles (SOC family 15).
    occ = occ[occ["O*NET-SOC Code"].str.startswith("15-")]
    sw = sw[sw["O*NET-SOC Code"].str.startswith("15-")]

    # 2. Keep only market-relevant tools, minus generic office tools and
    #    generic category-descriptor buckets (see STOP_SKILLS / GENERIC_SKILLS).
    sw = sw[(sw["Hot Technology"] == "Y") | (sw["In Demand"] == "Y")]
    sw = sw[~sw["Workplace Example"].isin(STOP_SKILLS | GENERIC_SKILLS)]

    G = nx.Graph()

    for _, row in occ.iterrows():
        G.add_node(row["Title"], node_type="role", soc=row["O*NET-SOC Code"])

    code_to_title = dict(zip(occ["O*NET-SOC Code"], occ["Title"]))

    # Pass 1: aggregate every Element Name each skill appears under.
    cat_counts: dict[str, Counter] = defaultdict(Counter)
    for _, row in sw.iterrows():
        cat_counts[row["Workplace Example"]][row["Element Name"]] += 1

    # Pass 2: add skill nodes with a deterministic category.
    for skill, counter in cat_counts.items():
        G.add_node(
            skill,
            node_type="skill",
            category=pick_category(counter),
            category_all=sorted(counter),      # provenance: what we chose between
            category_n=len(counter),
            source="onet",
        )
        if skill in CATEGORY_OVERRIDES:
            G.nodes[skill]["embed_category"] = CATEGORY_OVERRIDES[skill]

    # Pass 3: role--skill edges.
    for _, row in sw.iterrows():
        role = code_to_title.get(row["O*NET-SOC Code"])
        if role is not None:
            G.add_edge(role, row["Workplace Example"], relation="requires")

    # Inject custom skills that O*NET does not carry at all.
    for item in CUSTOM_SKILLS:
        if item["skill"] not in G:
            G.add_node(
                item["skill"],
                node_type="skill",
                category=item["category"],
                category_all=[item["category"]],
                category_n=1,
                embed_category=item["category"],
                source="custom",
            )

    if verbose:
        ambiguous = {s: c for s, c in cat_counts.items() if len(c) > 1}
        print(f"category assignment: {len(cat_counts)} O*NET skills, "
              f"{len(ambiguous)} carried >1 Element Name "
              f"(resolved by frequency, ties alphabetical)")
        for s, c in sorted(ambiguous.items(), key=lambda kv: -len(kv[1]))[:5]:
            print(f"  {s:<22} {len(c)} categories -> {pick_category(c)!r}")

    return G


def _embed_text(G: nx.Graph, node: str) -> str:
    """The string actually fed to the embedder for a skill node."""
    d = G.nodes[node]
    return f"{node} ({d.get('embed_category') or d.get('category', '')})"


def add_semantic_edges(
    G,
    model_name="BAAI/bge-small-en-v1.5",
    k=5,
    min_sim=0.30,
    strong_sim=0.60,
    use_embed_category=True,
    model=None,
):
    """Add skill<->skill 'similar' edges from embedding similarity.

    Two passes, unioned:
      1. rank-based  - top-k neighbours per skill, floored at `min_sim`
      2. threshold   - ANY pair at or above `strong_sim`, regardless of rank

    Pass 2 exists because a fixed top-k budget gets consumed inside dense
    clusters, silently dropping genuine high-similarity pairs at rank k+1.

    Uses FastEmbed (ONNX, CPU-only) by default. Set `model_name` to a
    sentence-transformers model to use the old backend (dev only).
    """
    skills = [n for n, d in G.nodes(data=True) if d.get("node_type") == "skill"]
    if len(skills) < 2:
        return G

    if use_embed_category:
        texts = [_embed_text(G, s) for s in skills]
    else:
        texts = [f"{s} ({G.nodes[s].get('category', '')})" for s in skills]

    # Use FastEmbed if available and model is a FastEmbed model
    use_fastembed = FASTEMBED_AVAILABLE and model_name.startswith("BAAI/")

    if use_fastembed:
        # FastEmbed: encode as passages (no query prefix for corpus)
        if model is None:
            model = TextEmbedding(model_name=model_name)
        embeddings = np.array(list(model.embed(texts)), dtype=np.float32)
    else:
        # sentence-transformers fallback
        if model is None:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(model_name)
        embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

    # Store embeddings in node attributes for Neo4j migration
    for i, skill in enumerate(skills):
        G.nodes[skill]["embedding"] = embeddings[i]

    # Compute cosine similarity
    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.clip(norms, 1e-12, None)
    sim = embeddings @ embeddings.T
    np.fill_diagonal(sim, -1.0)  # never connect a skill to itself

    pairs: dict[tuple[str, str], float] = {}

    def offer(i: int, j: int, score: float):
        key = (skills[i], skills[j]) if skills[i] < skills[j] else (skills[j], skills[i])
        if score > pairs.get(key, -1.0):
            pairs[key] = score

    # Top-k neighbors
    n_neighbors = min(k, len(skills) - 1)
    for i in range(len(skills)):
        # Get top-k similarities
        top_indices = np.argpartition(sim[i], -n_neighbors)[-n_neighbors:]
        top_indices = top_indices[np.argsort(sim[i][top_indices])[::-1]]
        for j in top_indices:
            score = float(sim[i][j])
            if score >= min_sim:
                offer(i, j, score)

    # Threshold-based: all pairs above strong_sim
    if strong_sim is not None:
        for i in range(len(skills)):
            for j in range(i + 1, len(skills)):
                score = float(sim[i][j])
                if score >= strong_sim:
                    offer(i, j, score)

    for (a, b), score in pairs.items():
        if G.has_edge(a, b):               # don't clobber 'requires' edges
            continue
        G.add_edge(a, b, relation="similar",
                   weight=round(score, 3), edge_source="embedding")
    return G


def add_seed_edges(G, edges=SEED_SIMILAR_EDGES, verbose=True):
    """Overlay curated substitutability edges. Call AFTER add_semantic_edges.

    --- FIX: add-only ---
    The previous version raised an existing edge's weight to the curated value
    whenever the curated value was higher. That meant 9 of 14 seeds overwrote
    real measurements: PyTorch->TensorFlow was measured at 0.736 and silently
    replaced by a hand-typed 0.90. Any downstream claim that "the graph found
    this link" would then have been describing my typing.

    Seeds now only fill genuine holes. Anything reported as `already found`
    should be deleted from SEED_SIMILAR_EDGES - the embedder covers it.
    """
    added, already, absent = [], [], []
    for a, b, w in edges:
        if a not in G or b not in G:
            absent.append((a, b))          # not in the taxonomy; a coverage gap
            continue
        if G.has_edge(a, b):
            already.append((a, b, G[a][b].get("weight")))
            continue
        G.add_edge(a, b, relation="similar", weight=w, edge_source="seed")
        added.append((a, b, w))

    if verbose:
        print(f"\nseed edges: {len(added)} added | {len(already)} already found "
              f"by the embedder | {len(absent)} skipped (node absent)")
        for a, b, w in added:
            print(f"  + {a} <-> {b}  (asserted {w})")
        for a, b, w in already:
            print(f"  = {a} <-> {b}  (measured {w}) -> DELETE from SEED_SIMILAR_EDGES")
        for a, b in absent:
            print(f"  ! {a} <-> {b}  -> taxonomy coverage gap, not an edge problem")
    return G


def build_categorical_graph(
    onet_dir: str = ONET_DIR,
    edge_weight: float = 0.5,
    max_category_size: int | None = None,
    verbose: bool = True,
) -> nx.Graph:
    """Skill graph with edges from O*NET Element Name membership alone.

    Two skills sharing an Element Name get an edge of fixed weight.
    No embedder is loaded, imported, or called anywhere in this path.

    max_category_size: if set, categories with more members than this are
        skipped entirely rather than contributing a large clique. Report the
        skipped set; do not silently drop.
    """
    import pandas as pd
    from collections import Counter, defaultdict

    occ = pd.read_csv(f"{onet_dir}/Occupation Data.txt", sep="\t")
    sw = pd.read_csv(f"{onet_dir}/Software Skills.txt", sep="\t")

    # 1. Scope to software-domain roles (SOC family 15).
    occ = occ[occ["O*NET-SOC Code"].str.startswith("15-")]
    sw = sw[sw["O*NET-SOC Code"].str.startswith("15-")]

    # 2. Keep only market-relevant tools, minus generic office tools and
    #    generic category-descriptor buckets (see STOP_SKILLS / GENERIC_SKILLS).
    sw = sw[(sw["Hot Technology"] == "Y") | (sw["In Demand"] == "Y")]
    sw = sw[~sw["Workplace Example"].isin(STOP_SKILLS | GENERIC_SKILLS)]

    G = nx.Graph()

    for _, row in occ.iterrows():
        G.add_node(row["Title"], node_type="role", soc=row["O*NET-SOC Code"])

    code_to_title = dict(zip(occ["O*NET-SOC Code"], occ["Title"]))

    # Pass 1: aggregate every Element Name each skill appears under.
    # (Reusing the exact same logic as build_skill_graph for identical node sets)
    cat_counts: dict[str, Counter] = defaultdict(Counter)
    for _, row in sw.iterrows():
        cat_counts[row["Workplace Example"]][row["Element Name"]] += 1

    # Pass 2: add skill nodes with a deterministic category.
    for skill, counter in cat_counts.items():
        G.add_node(
            skill,
            node_type="skill",
            category=pick_category(counter),
            category_all=sorted(counter),      # provenance: what we chose between
            category_n=len(counter),
            source="onet",
        )
        if skill in CATEGORY_OVERRIDES:
            G.nodes[skill]["embed_category"] = CATEGORY_OVERRIDES[skill]

    # Pass 3: role--skill edges.
    for _, row in sw.iterrows():
        role = code_to_title.get(row["O*NET-SOC Code"])
        if role is not None:
            G.add_edge(role, row["Workplace Example"], relation="requires")

    # Inject custom skills that O*NET does not carry at all.
    for item in CUSTOM_SKILLS:
        if item["skill"] not in G:
            G.add_node(
                item["skill"],
                node_type="skill",
                category=item["category"],
                category_all=[item["category"]],
                category_n=1,
                embed_category=item["category"],
                source="custom",
            )

    # Pass 4: categorical edges — cliques per Element Name
    # Build category -> skills map using the DETERMINISTIC category (same as node attr)
    cat_to_skills: dict[str, list[str]] = defaultdict(list)
    for skill, counter in cat_counts.items():
        cat = pick_category(counter)
        cat_to_skills[cat].append(skill)

    # Add custom skills to their categories too
    for item in CUSTOM_SKILLS:
        if item["skill"] in G:
            cat = item["category"]
            cat_to_skills[cat].append(item["skill"])

    skipped_categories = []
    added_edges = 0

    for cat, skills_in_cat in cat_to_skills.items():
        n = len(skills_in_cat)
        if max_category_size is not None and n > max_category_size:
            skipped_categories.append((cat, n))
            if verbose:
                print(f"  SKIP category '{cat}' (size {n} > max_category_size={max_category_size})")
            continue
        # Add all pairwise edges within this category (clique)
        for i in range(n):
            for j in range(i + 1, n):
                a, b = skills_in_cat[i], skills_in_cat[j]
                if not G.has_edge(a, b):  # don't clobber 'requires' edges
                    G.add_edge(a, b, relation="similar",
                               weight=edge_weight, edge_source="category")
                    added_edges += 1

    if verbose:
        print(f"\nCategorical graph: {len(cat_to_skills)} categories, "
              f"{added_edges} similar edges added")
        if skipped_categories:
            print(f"  Skipped categories (size > {max_category_size}): {skipped_categories}")

    return G


def audit_graph(G, bridge_cutoff: float = 0.6) -> dict:
    """Report the two conditions that silently break bridging."""
    skills = [n for n, d in G.nodes(data=True) if d.get("node_type") == "skill"]
    sim_edges = [(u, v, d["weight"]) for u, v, d in G.edges(data=True)
                 if d.get("relation") == "similar"]

    connected = {u for u, v, _ in sim_edges} | {v for u, v, _ in sim_edges}
    orphans = sorted(set(skills) - connected)
    dead = [e for e in sim_edges if (1 - e[2]) > bridge_cutoff]

    report = {
        "skills": len(skills),
        "similar_edges": len(sim_edges),
        "orphan_skills": orphans,
        "untraversable_edges": len(dead),
        "min_traversable_similarity": round(1 - bridge_cutoff, 3),
    }
    print(f"\n--- graph audit (bridge_cutoff={bridge_cutoff}) ---")
    print(f"skills: {report['skills']} | similar edges: {report['similar_edges']}")
    print(f"orphan skills (no similar edge, can never bridge): {len(orphans)}")
    if orphans:
        print(f"  {orphans[:15]}{' ...' if len(orphans) > 15 else ''}")
    print(f"edges below {report['min_traversable_similarity']} similarity "
          f"(present but untraversable): {len(dead)}")
    return report


if __name__ == "__main__":
    G = build_skill_graph()
    G = add_semantic_edges(G)
    G = add_seed_edges(G)

    roles = [n for n, d in G.nodes(data=True) if d["node_type"] == "role"]
    skills = [n for n, d in G.nodes(data=True) if d["node_type"] == "skill"]
    sim_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("relation") == "similar"]
    print(f"\nGraph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"  roles: {len(roles)} | skills: {len(skills)} | skill-skill edges: {len(sim_edges)}")

    audit_graph(G)

    for anchor in ("Docker", "PyTorch"):
        if anchor not in G:
            print(f"\n{anchor}: NOT IN GRAPH")
            continue
        nbrs = [(n, d["weight"], d.get("edge_source", "?")) for n, d in G[anchor].items()
                if d.get("relation") == "similar"]
        print(f"\nMost similar to {anchor}  [embeds as: {_embed_text(G, anchor)}]")
        for skill, w, src in sorted(nbrs, key=lambda x: -x[1]):
            print(f"  {skill:30s} {w:5.3f}  d={1 - w:5.3f}  ({src})")