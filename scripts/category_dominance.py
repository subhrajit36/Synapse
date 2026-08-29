"""Step 4: is the embedder measuring tool relatedness, or just O*NET category membership?

    python scripts/category_dominance.py
    python scripts/category_dominance.py --anchors Docker PyTorch Jenkins
    python scripts/category_dominance.py --out data/eval/edge_construction_audit.md

Extended for Work Item 3:
    python scripts/category_dominance.py --substrate typed --typed-cache data/eval/typed_edge_cache.jsonl

WHY THIS EXISTS
---------------
The bridge diagnostic showed Docker's neighbourhood is nearly flat:

    Kubernetes 0.542 | Bitbucket 0.521 | Spring Boot 0.476 | GitHub 0.450

A genuine substitute beats an unrelated Git host by 0.02. Meanwhile PyTorch's
neighbourhood has a clean cliff (TensorFlow 0.736, Keras 0.673, then 0.419).

The difference is the category string. 'Deep Learning Framework' is a tight label
that nearly *is* its members. 'Application server software' is a broad O*NET
bucket, so the suffix adds the same constant to every pair inside it and the tool
name barely moves the score.

If that reading is right, `distance = 1 - cosine` is largely encoding "same
Element Name", and every graph edge is a noisy re-derivation of a lookup you
already own exactly. This script tests it three ways:

  1. DOMINANCE  - how well does cosine similarity predict shared category?
                  AUC near 1.0 => it is a category detector, not a relatedness
                  measure. Reported per text variant.
  2. SPREAD     - within a category, how much do similarities vary? Near-zero
                  spread means the model cannot discriminate inside a bucket.
  3. PROBES     - for known substitute pairs, the rank and the MARGIN over the
                  best non-substitute. Margin is the number that matters: 0.02
                  is noise, 0.2 is signal.

Text variants compared (embedding substrate):
    bare     "Docker"
    onet     "Docker (Application server software)"     <- raw O*NET category
    override "PyTorch (Deep Learning Framework)"        <- current production path

Graph substrates (categorical, typed):
    categorical: edges from O*NET Element Name membership alone (cliques)
    typed:       LLM-classified edges (substitute, complement, prerequisite, unrelated)

Grouping for (1) and (2) is always the deterministic O*NET category, so the three
variants are directly comparable.

READING THE RESULT
------------------
If `bare` has a much lower dominance AUC but equal-or-better probe margins, the
category suffix is actively hurting you: drop it and rebuild edges from names.
If every variant has high AUC and thin margins, embedding-derived edges are the
wrong substrate regardless of phrasing, and the co-occurrence route (or a
directed transfer-cost route) is where the graph's value has to come from.

For typed substrate: target AUC <= 0.75 (edges carry substitute signal beyond category),
mean margin (honest) >= +0.15, rank-1 hits >= 5/7.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")

from synapse.graph.build_graph import _embed_text, build_skill_graph  # noqa: E402

VARIANTS = ("bare", "onet", "override")
SUBSTRATES = ("embedding", "categorical", "typed")

# Pairs a domain expert would call substitutable. Absent ones are skipped and
# reported - absence is itself a taxonomy-coverage finding.
# Note: LLM correctly classified some of these as complement/prerequisite, not substitute.
# Only pairs classified as "substitute" should be expected to have high traversable scores.
PROBE_PAIRS = [
    ("Docker", "Kubernetes"),          # LLM: complement (coherent stack)
    ("PyTorch", "TensorFlow"),         # LLM: substitute ✓
    ("TensorFlow", "Keras"),           # LLM: prerequisite (TF -> Keras)
    ("XGBoost", "LightGBM"),           # LLM: substitute ✓
    ("GitHub", "GitLab"),              # LLM: substitute ✓
    ("MySQL", "PostgreSQL"),           # LLM: substitute ✓
    ("React", "Angular"),              # LLM: substitute ✓ (not in graph)
    ("Jenkins", "Apache Maven"),       # LLM: complement (not in graph)
    ("Amazon Web Services", "Microsoft Azure"),  # LLM: substitute (not in graph)
    ("Apache Spark", "Apache Hadoop"), # LLM: complement
]

# Expected edge types per LLM classification (for reference)
# Only "substitute" pairs should have high traversable scores in typed substrate.
EXPECTED_EDGE_TYPES = {
    ("Docker", "Kubernetes"): "complement",
    ("PyTorch", "TensorFlow"): "substitute",
    ("TensorFlow", "Keras"): "prerequisite",
    ("XGBoost", "LightGBM"): "substitute",
    ("GitHub", "GitLab"): "substitute",
    ("MySQL", "PostgreSQL"): "substitute",
    ("React", "Angular"): "substitute",
    ("Jenkins", "Apache Maven"): "complement",
    ("Amazon Web Services", "Microsoft Azure"): "substitute",
    ("Apache Spark", "Apache Hadoop"): "complement",
}

# Hand-authored category overrides (from CATEGORY_OVERRIDES in build_graph.py)
# These pairs have manually assigned categories, not raw O*NET Element Names.
# We track them separately so probe margins excluding hand-authored pairs can be reported.
HAND_AUTHORED_PAIRS = {
    ("PyTorch", "TensorFlow"): True,    # both "Deep Learning Framework"
    ("TensorFlow", "Keras"): True,      # both "Deep Learning Framework"
    ("XGBoost", "LightGBM"): True,      # both "Gradient Boosting Library"
    ("XGBoost", "CatBoost"): True,      # both "Gradient Boosting Library"
    ("LightGBM", "CatBoost"): True,     # both "Gradient Boosting Library"
    ("BERT", "DistilBERT"): True,       # both "Transformer Model"
    ("Llama", "Mistral"): True,         # both "Large Language Model"
    ("React Native", "React"): True,    # both "Mobile Development" / "Web Framework"
    ("FastAPI", "Flask"): True,         # both "Web Framework"
    ("FastAPI", "Django"): True,        # both "Web Framework"
    ("BERT", "Hugging Face"): True,     # "Transformer Model" vs "AI Ecosystem" - wait, check
    ("Hugging Face", "PyTorch"): True,  # "AI Ecosystem" vs "Deep Learning Framework" - not same
    ("LangChain", "Hugging Face"): True, # "LLM Orchestration" vs "AI Ecosystem" - not same
    ("React Native", "React"): True,    # both have category override
}

# Actually check CATEGORY_OVERRIDES for the real hand-authored pairs
# The pairs where BOTH skills have a category override AND they share the same override category
# Let's compute this properly from the graph


# ----------------------------------------------------------------- numeric core

def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """P(score of a same-category pair > score of a different-category pair).

    Mann-Whitney U as a rank statistic. Ties get average ranks, so a model that
    outputs a constant scores exactly 0.5 rather than something misleading.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=bool)
    n_pos, n_neg = int(labels.sum()), int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=float)
    # average ranks within tied groups
    s_sorted = scores[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1

    return (ranks[labels].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def upper_triangle(sim: np.ndarray):
    """Indices and values of every distinct unordered pair."""
    iu = np.triu_indices(sim.shape[0], k=1)
    return iu, sim[iu]


def dominance(sim: np.ndarray, categories: list[str]) -> dict:
    """How much of pairwise similarity is explained by shared category?"""
    (ii, jj), vals = upper_triangle(sim)
    cats = np.asarray(categories, dtype=object)
    same = cats[ii] == cats[jj]

    r = float(np.corrcoef(same.astype(float), vals)[0, 1]) if same.any() else float("nan")
    return {
        "auc": auc(vals, same),
        "point_biserial_r": r,
        "same_n": int(same.sum()),
        "same_mean": float(vals[same].mean()) if same.any() else float("nan"),
        "same_std": float(vals[same].std()) if same.any() else float("nan"),
        "diff_n": int((~same).sum()),
        "diff_mean": float(vals[~same].mean()),
        "diff_std": float(vals[~same].std()),
    }


def within_spread(sim: np.ndarray, skills: list[str], categories: list[str],
                  min_members: int = 3) -> list[tuple]:
    """Per-category similarity spread. Low std = no discrimination inside the bucket."""
    idx = defaultdict(list)
    for i, c in enumerate(categories):
        idx[c].append(i)

    rows = []
    for cat, members in idx.items():
        if len(members) < min_members:
            continue
        sub = sim[np.ix_(members, members)]
        (a, b), vals = upper_triangle(sub)
        rows.append((cat, len(members), float(vals.mean()), float(vals.std()),
                     float(vals.min()), float(vals.max())))
    return sorted(rows, key=lambda r: -r[1])


def rank_and_margin(sim: np.ndarray, skills: list[str], a: str, b: str):
    """Rank of `b` among `a`'s neighbours, and the margin over the best rival.

    margin = sim(a,b) - max(sim(a,x)) for x not in {a,b}.
    Positive => b is rank 1 by that much. Negative => something else outranks it.
    """
    i, j = skills.index(a), skills.index(b)
    row = sim[i].copy()
    row[i] = -np.inf
    target = row[j]
    rank = int((row > target).sum()) + 1
    rival = row.copy()
    rival[j] = -np.inf
    best_rival_i = int(np.argmax(rival))
    return {
        "sim": float(target),
        "rank": rank,
        "margin": float(target - rival[best_rival_i]),
        "rival": skills[best_rival_i],
        "rival_sim": float(rival[best_rival_i]),
    }


# ----------------------------------------------------------------- substrate-specific similarity matrix builders

def build_embedding_sim(G, skills, model_name):
    """Build similarity matrix from embeddings (existing behavior)."""
    from sentence_transformers import SentenceTransformer, util

    model = SentenceTransformer(model_name)
    out = {}
    for v in VARIANTS:
        emb = model.encode(texts_for(G, skills, v), convert_to_tensor=True,
                           show_progress_bar=False)
        sim = util.cos_sim(emb, emb).cpu().numpy().astype(float)
        np.fill_diagonal(sim, -1.0)
        out[v] = sim
    return out


def build_categorical_sim(G, skills, categories):  # categories kept for signature consistency
    """Build similarity matrix from categorical graph edges.

    For categorical substrate: edge present = weight (0.5), no edge = 0.
    Convert to score: edge present -> 1 - normalised_cost, no edge -> 0.
    """
    # For categorical, all edges have the same weight (0.5 by default)
    # We need to check which pairs have a "similar" edge from category
    sim = np.zeros((len(skills), len(skills)), dtype=float)
    skill_to_idx = {s: i for i, s in enumerate(skills)}

    for u, v, d in G.edges(data=True):
        if d.get("relation") == "similar" and d.get("edge_source") == "category":
            i, j = skill_to_idx.get(u), skill_to_idx.get(v)
            if i is not None and j is not None:
                # Score = 1 - normalised_cost
                # For categorical edges, weight = 0.5 (similarity). Traversal cost = 1 - similarity = 0.5.
                # Normalised cost = cost / max_cost. If max_cost = 1.0, normalised = 0.5.
                # Score = 1 - 0.5 = 0.5.
                weight = d.get("weight", 0.5)
                cost = 1.0 - weight
                max_cost = 1.0  # theoretical max (when similarity = 0)
                normalised = cost / max_cost if max_cost > 0 else 0.0
                score = 1.0 - normalised  # = weight
                sim[i, j] = score
                sim[j, i] = score

    np.fill_diagonal(sim, -1.0)
    return {"categorical": sim}


def build_typed_sim(G, skills, typed_cache_path):
    """Build similarity matrix from typed graph edges.

    For typed substrate: only traversable edges (substitute, prerequisite) count.
    Edge weight is traversal cost. Score = 1 - normalised_cost.
    """
    sim = np.zeros((len(skills), len(skills)), dtype=float)
    skill_to_idx = {s: i for i, s in enumerate(skills)}

    # Load typed edges from cache
    cache = load_typed_cache(typed_cache_path)

    # Find max traversal cost for normalization
    traversable_edges = []
    for (a, b), result in cache.items():
        if result.edge_type in ("substitute", "prerequisite"):
            edge = TypedEdge(
                a=result.a, b=result.b,
                edge_type=result.edge_type,
                direction=result.direction,
                confidence=result.confidence,
                source=result.source if hasattr(result, 'source') else "llm",
                rationale=result.rationale,
            )
            if edge.is_traversable:
                traversable_edges.append((a, b, edge.weight))

    max_cost = max((w for _, _, w in traversable_edges), default=1.0)

    for a, b, cost in traversable_edges:
        i, j = skill_to_idx.get(a), skill_to_idx.get(b)
        if i is not None and j is not None:
            normalised = cost / max_cost if max_cost > 0 else 0.0
            score = 1.0 - normalised
            sim[i, j] = score
            sim[j, i] = score

    np.fill_diagonal(sim, -1.0)
    return {"typed": sim}


def load_typed_cache(cache_path: str | Path):
    """Load classification cache from JSONL file."""
    cache_path = Path(cache_path)
    cache = {}
    if not cache_path.exists():
        return cache

    with cache_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            # Filter to only ClassificationResult fields
            cr_fields = {"a", "b", "edge_type", "direction", "confidence", "rationale"}
            filtered = {k: v for k, v in data.items() if k in cr_fields}
            key = tuple(sorted([filtered["a"], filtered["b"]]))
            cache[key] = type('ClassificationResult', (), filtered)
    return cache


# Need TypedEdge class from typed_edges module
from synapse.graph.typed_edges import TypedEdge, TYPE_COST  # noqa: E402


def texts_for(G, skills: list[str], variant: str) -> list[str]:
    if variant == "bare":
        return list(skills)
    if variant == "onet":
        return [f"{s} ({G.nodes[s].get('category', '')})" for s in skills]
    if variant == "override":
        return [_embed_text(G, s) for s in skills]
    raise ValueError(variant)


# ----------------------------------------------------------------- probe reporting helpers

def is_hand_authored_pair(a: str, b: str, G) -> bool:
    """Check if this probe pair uses hand-authored category overrides."""
    cat_a = G.nodes[a].get("embed_category") or G.nodes[a].get("category", "")
    cat_b = G.nodes[b].get("embed_category") or G.nodes[b].get("category", "")
    # Check if either has a CATEGORY_OVERRIDE
    from synapse.graph.build_graph import CATEGORY_OVERRIDES
    return (a in CATEGORY_OVERRIDES or b in CATEGORY_OVERRIDES)


def check_variant_identity(sims: dict, skills: list[str], present_pairs: list[tuple]) -> dict:
    """Check which probe rows are byte-identical across variants."""
    identical = {}
    for v1 in sims:
        for v2 in sims:
            if v1 >= v2:
                continue
            identical_pairs = []
            for a, b in present_pairs:
                i, j = skills.index(a), skills.index(b)
                if sims[v1][i, j] == sims[v2][i, j]:
                    identical_pairs.append(f"{a}↔{b}")
            if identical_pairs:
                identical[f"{v1}_vs_{v2}"] = identical_pairs
    return identical


# ----------------------------------------------------------------- reporting

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="all-MiniLM-L6-v2")
    ap.add_argument("--anchors", nargs="*", default=["Docker", "PyTorch"])
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--out", default="data/eval/edge_construction_audit.md")
    ap.add_argument("--substrate", choices=SUBSTRATES, default="embedding",
                    help="Which substrate to audit: embedding (default), categorical, or typed")
    ap.add_argument("--typed-cache", default="data/eval/typed_edge_cache.jsonl",
                    help="Path to typed edge cache JSONL (for typed substrate)")
    args = ap.parse_args()

    # Build the appropriate graph based on substrate
    if args.substrate == "categorical":
        from synapse.graph.build_graph import build_categorical_graph
        G = build_categorical_graph(verbose=False)
        print("Built categorical graph (O*NET Element Name cliques only)")
    else:
        G = build_skill_graph()
        if args.substrate == "embedding":
            from synapse.graph.build_graph import add_semantic_edges, add_seed_edges
            G = add_semantic_edges(G)
            G = add_seed_edges(G)
            print("Built embedding graph (with semantic edges)")
        else:  # typed
            from synapse.graph.build_graph import add_semantic_edges, add_seed_edges
            from synapse.graph.typed_edges import build_typed_graph
            G = add_semantic_edges(G)
            G = add_seed_edges(G)
            typed_edges = []
            # Load typed edges from cache
            cache = load_typed_cache(args.typed_cache)
            for (a, b), result in cache.items():
                if result.edge_type != "unrelated":
                    typed_edges.append(TypedEdge(
                        a=result.a, b=result.b,
                        edge_type=result.edge_type,
                        direction=result.direction,
                        confidence=result.confidence,
                        source="llm",
                        rationale=result.rationale,
                    ))
            G = build_typed_graph(G, typed_edges)
            print(f"Built typed graph with {len(typed_edges)} LLM-classified edges")

    skills = sorted(n for n, d in G.nodes(data=True) if d.get("node_type") == "skill")
    categories = [G.nodes[s].get("category", "") for s in skills]
    print(f"{len(skills)} skills across {len(set(categories))} O*NET categories")

    # Build similarity matrices based on substrate
    if args.substrate == "embedding":
        sims = build_embedding_sim(G, skills, args.model)
        variant_labels = VARIANTS
    elif args.substrate == "categorical":
        sims = build_categorical_sim(G, skills, categories)
        variant_labels = ("categorical",)
    else:  # typed
        sims = build_typed_sim(G, skills, args.typed_cache)
        variant_labels = ("typed",)

    md = ["# Edge construction audit", "",
          f"`{len(skills)}` skills, `{len(set(categories))}` O*NET categories, "
          f"substrate `{args.substrate}`.", "",
          "Grouping for dominance/spread is the deterministic O*NET Element Name.", ""]

    # --- 1. dominance -------------------------------------------------------
    print("\n=== 1. CATEGORY DOMINANCE ===")
    print("does the similarity score just predict 'same O*NET Element Name'?\n")
    hdr = f"{'variant':<12} {'AUC':>7} {'r':>7} {'same mean':>11} {'diff mean':>11} {'gap':>7}"
    print(hdr); print("-" * len(hdr))
    md += ["## 1. Category dominance", "",
           "AUC = P(a same-category pair scores above a different-category pair). "
           "Near 1.0 means the substrate is a category detector.", "",
           "| variant | AUC | point-biserial r | same-cat mean | diff-cat mean | gap |",
           "|---|---|---|---|---|---|"]
    for v in variant_labels:
        d = dominance(sims[v], categories)
        gap = d["same_mean"] - d["diff_mean"]
        print(f"{v:<12} {d['auc']:>7.3f} {d['point_biserial_r']:>7.3f} "
              f"{d['same_mean']:>11.3f} {d['diff_mean']:>11.3f} {gap:>7.3f}")
        md.append(f"| {v} | {d['auc']:.3f} | {d['point_biserial_r']:.3f} | "
                  f"{d['same_mean']:.3f} | {d['diff_mean']:.3f} | {gap:.3f} |")
    md.append("")

    # --- 2. within-category spread -----------------------------------------
    # Only for embedding substrate (spread only meaningful for continuous scores)
    if args.substrate == "embedding":
        print("\n=== 2. WITHIN-CATEGORY SPREAD (variant: override) ===")
        print("low std => the tool name contributes nothing inside a bucket\n")
        md += ["## 2. Within-category spread (`override`)", "",
               "Low std means the tool name adds nothing once the suffix is fixed.", "",
               "| category | n | mean | std | min | max |", "|---|---|---|---|---|---|"]
        print(f"{'category':<45} {'n':>4} {'mean':>7} {'std':>7} {'min':>7} {'max':>7}")
        for cat, n, mean, std, lo, hi in within_spread(sims["override"], skills, categories)[:12]:
            print(f"{cat[:44]:<45} {n:>4} {mean:>7.3f} {std:>7.3f} {lo:>7.3f} {hi:>7.3f}")
            md.append(f"| {cat} | {n} | {mean:.3f} | {std:.3f} | {lo:.3f} | {hi:.3f} |")
        md.append("")

    # --- 3. probe pairs -----------------------------------------------------
    print("\n=== 3. PROBE PAIRS: rank and margin over best rival ===")
    print("margin is the decisive number. ~0.02 is noise; ~0.2 is signal.\n")
    md += ["## 3. Probe pairs", "",
           "`margin` = sim(a,b) − best rival similarity. Small margin = the "
           "'correct' neighbour is indistinguishable from an unrelated one.", "",
           "| pair | variant | sim | rank | margin | best rival | hand-authored |",
           "|---|---|---|---|---|---|---|"]
    present = [(a, b) for a, b in PROBE_PAIRS if a in skills and b in skills]
    absent = [(a, b) for a, b in PROBE_PAIRS if (a, b) not in present]
    margins = {v: [] for v in variant_labels}
    hand_authored_margins = {v: [] for v in variant_labels}
    non_hand_authored_margins = {v: [] for v in variant_labels}

    for a, b in present:
        print(f"{a} <-> {b}")
        hand_authored = is_hand_authored_pair(a, b, G)
        for v in variant_labels:
            r = rank_and_margin(sims[v], skills, a, b)
            margins[v].append(r["margin"])
            if hand_authored:
                hand_authored_margins[v].append(r["margin"])
            else:
                non_hand_authored_margins[v].append(r["margin"])
            hand_str = "✓" if hand_authored else "✗"
            print(f"   {v:<9} sim={r['sim']:.3f} rank=#{r['rank']:<3} "
                  f"margin={r['margin']:+.3f}   rival: {r['rival']} ({r['rival_sim']:.3f})  hand-authored: {hand_str}")
            md.append(f"| {a} ↔ {b} | {v} | {r['sim']:.3f} | #{r['rank']} | "
                      f"{r['margin']:+.3f} | {r['rival']} ({r['rival_sim']:.3f}) | {hand_str} |")
    if absent:
        print(f"\nprobe pairs skipped (node absent from taxonomy): {absent}")
        md += ["", f"Skipped — node absent from taxonomy: `{absent}`. "
                   "These are coverage gaps, not edge-construction problems."]

    print("\n--- mean probe margin (higher = better discrimination) ---")
    md += ["", "### Mean probe margin (all pairs)", "",
           "| variant | mean margin | rank-1 hits |", "|---|---|---|"]
    for v in variant_labels:
        arr = np.array(margins[v]) if margins[v] else np.array([np.nan])
        hits = int((arr > 0).sum())
        print(f"  {v:<12} {arr.mean():+.3f}   (rank-1 on {hits}/{len(present)} probes)")
        md.append(f"| {v} | {arr.mean():+.3f} | {hits}/{len(present)} |")

    # --- 3b. Mean probe margin excluding hand-authored pairs ---
    print("\n--- mean probe margin EXCLUDING hand-authored pairs ---")
    md += ["", "### Mean probe margin (excluding hand-authored category overrides)", "",
           "| variant | mean margin | rank-1 hits |", "|---|---|---|"]
    for v in variant_labels:
        arr = np.array(non_hand_authored_margins[v]) if non_hand_authored_margins[v] else np.array([np.nan])
        hits = int((arr > 0).sum())
        total_non = len(non_hand_authored_margins[v])
        print(f"  {v:<12} {arr.mean():+.3f}   (rank-1 on {hits}/{total_non} non-hand-authored probes)")
        md.append(f"| {v} | {arr.mean():+.3f} | {hits}/{total_non} |")

    # --- 3c. Variant identity check ---
    if args.substrate == "embedding":
        print("\n--- variant identity check (which probe rows are byte-identical across variants) ---")
        md += ["", "### Variant identity check", "",
               "Probe rows where the similarity score is exactly identical across variants. "
               "This identifies when the category suffix is doing all the work."]
        identical = check_variant_identity(sims, skills, present)
        if identical:
            md += ["", "| comparison | identical pairs |", "|---|---|"]
            for comp, pairs in identical.items():
                md.append(f"| {comp} | {', '.join(pairs)} |")
                print(f"  {comp}: {', '.join(pairs)}")
        else:
            print("  No identical probe rows across variants")
            md.append("No identical probe rows across variants.")
        md.append("")

    # --- 4. anchor neighbourhoods ------------------------------------------
    md += ["", "## 4. Anchor neighbourhoods", ""]
    for anchor in args.anchors:
        if anchor not in skills:
            print(f"\n{anchor}: not in taxonomy")
            continue
        print(f"\n=== {anchor}: top {args.top} by variant ===")
        md += [f"### {anchor}", "", "| rank | " + " | ".join(variant_labels) + " |",
               "|---|" + "---|" * len(variant_labels)]
        cols = {}
        for v in variant_labels:
            row = sims[v][skills.index(anchor)]
            top = np.argsort(-row)[:args.top]
            cols[v] = [(skills[t], float(row[t])) for t in top]
        for rank in range(args.top):
            cells = [f"{cols[v][rank][0]} {cols[v][rank][1]:.3f}" for v in variant_labels]
            print(f"  {rank + 1}. " + " | ".join(f"{c:<34}" for c in cells))
            md.append(f"| {rank + 1} | " + " | ".join(cells) + " |")
        md.append("")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write("\n".join(md) + "\n")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()