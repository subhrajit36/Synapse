"""Step 4: is the embedder measuring tool relatedness, or just O*NET category membership?

    python scripts/category_dominance.py
    python scripts/category_dominance.py --anchors Docker PyTorch Jenkins
    python scripts/category_dominance.py --out data/eval/edge_construction_audit.md

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

Text variants compared:
    bare     "Docker"
    onet     "Docker (Application server software)"     <- raw O*NET category
    override "PyTorch (Deep Learning Framework)"        <- current production path

Grouping for (1) and (2) is always the deterministic O*NET category, so the three
variants are directly comparable.

READING THE RESULT
------------------
If `bare` has a much lower dominance AUC but equal-or-better probe margins, the
category suffix is actively hurting you: drop it and rebuild edges from names.
If every variant has high AUC and thin margins, embedding-derived edges are the
wrong substrate regardless of phrasing, and the co-occurrence route (or a
directed transfer-cost route) is where the graph's value has to come from.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, "src")

from synapse.graph.build_graph import _embed_text, build_skill_graph  # noqa: E402

VARIANTS = ("bare", "onet", "override")

# Pairs a domain expert would call substitutable. Absent ones are skipped and
# reported - absence is itself a taxonomy-coverage finding.
PROBE_PAIRS = [
    ("Docker", "Kubernetes"),
    ("PyTorch", "TensorFlow"),
    ("TensorFlow", "Keras"),
    ("XGBoost", "LightGBM"),
    ("GitHub", "GitLab"),
    ("MySQL", "PostgreSQL"),
    ("React", "Angular"),
    ("Jenkins", "Apache Maven"),
    ("Amazon Web Services", "Microsoft Azure"),
    ("Apache Spark", "Apache Hadoop"),
]


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


# ----------------------------------------------------------------- embedding

def texts_for(G, skills: list[str], variant: str) -> list[str]:
    if variant == "bare":
        return list(skills)
    if variant == "onet":
        return [f"{s} ({G.nodes[s].get('category', '')})" for s in skills]
    if variant == "override":
        return [_embed_text(G, s) for s in skills]
    raise ValueError(variant)


def embed_all(G, skills, model_name):
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


# ----------------------------------------------------------------- reporting

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="all-MiniLM-L6-v2")
    ap.add_argument("--anchors", nargs="*", default=["Docker", "PyTorch"])
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--out", default="data/eval/edge_construction_audit.md")
    args = ap.parse_args()

    G = build_skill_graph()
    skills = sorted(n for n, d in G.nodes(data=True) if d.get("node_type") == "skill")
    categories = [G.nodes[s].get("category", "") for s in skills]
    print(f"{len(skills)} skills across {len(set(categories))} O*NET categories")

    sims = embed_all(G, skills, args.model)
    md = ["# Edge construction audit", "",
          f"`{len(skills)}` skills, `{len(set(categories))}` O*NET categories, "
          f"model `{args.model}`.", "",
          "Grouping for dominance/spread is the deterministic O*NET Element Name.", ""]

    # --- 1. dominance -------------------------------------------------------
    print("\n=== 1. CATEGORY DOMINANCE ===")
    print("does cosine similarity just predict 'same O*NET Element Name'?\n")
    hdr = f"{'variant':<10} {'AUC':>7} {'r':>7} {'same mean':>11} {'diff mean':>11} {'gap':>7}"
    print(hdr); print("-" * len(hdr))
    md += ["## 1. Category dominance", "",
           "AUC = P(a same-category pair scores above a different-category pair). "
           "Near 1.0 means the model is a category detector.", "",
           "| variant | AUC | point-biserial r | same-cat mean | diff-cat mean | gap |",
           "|---|---|---|---|---|---|"]
    for v in VARIANTS:
        d = dominance(sims[v], categories)
        gap = d["same_mean"] - d["diff_mean"]
        print(f"{v:<10} {d['auc']:>7.3f} {d['point_biserial_r']:>7.3f} "
              f"{d['same_mean']:>11.3f} {d['diff_mean']:>11.3f} {gap:>7.3f}")
        md.append(f"| {v} | {d['auc']:.3f} | {d['point_biserial_r']:.3f} | "
                  f"{d['same_mean']:.3f} | {d['diff_mean']:.3f} | {gap:.3f} |")
    md.append("")

    # --- 2. within-category spread -----------------------------------------
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
           "| pair | variant | sim | rank | margin | best rival |",
           "|---|---|---|---|---|---|"]
    present = [(a, b) for a, b in PROBE_PAIRS if a in skills and b in skills]
    absent = [(a, b) for a, b in PROBE_PAIRS if (a, b) not in present]
    margins = {v: [] for v in VARIANTS}
    for a, b in present:
        print(f"{a} <-> {b}")
        for v in VARIANTS:
            r = rank_and_margin(sims[v], skills, a, b)
            margins[v].append(r["margin"])
            print(f"   {v:<9} sim={r['sim']:.3f} rank=#{r['rank']:<3} "
                  f"margin={r['margin']:+.3f}   rival: {r['rival']} ({r['rival_sim']:.3f})")
            md.append(f"| {a} ↔ {b} | {v} | {r['sim']:.3f} | #{r['rank']} | "
                      f"{r['margin']:+.3f} | {r['rival']} ({r['rival_sim']:.3f}) |")
    if absent:
        print(f"\nprobe pairs skipped (node absent from taxonomy): {absent}")
        md += ["", f"Skipped — node absent from taxonomy: `{absent}`. "
                   "These are coverage gaps, not edge-construction problems."]

    print("\n--- mean probe margin (higher = better discrimination) ---")
    md += ["", "### Mean probe margin", "",
           "| variant | mean margin | rank-1 hits |", "|---|---|---|"]
    for v in VARIANTS:
        arr = np.array(margins[v]) if margins[v] else np.array([np.nan])
        hits = int((arr > 0).sum())
        print(f"  {v:<10} {arr.mean():+.3f}   (rank-1 on {hits}/{len(present)} probes)")
        md.append(f"| {v} | {arr.mean():+.3f} | {hits}/{len(present)} |")

    # --- 4. anchor neighbourhoods ------------------------------------------
    md += ["", "## 4. Anchor neighbourhoods", ""]
    for anchor in args.anchors:
        if anchor not in skills:
            print(f"\n{anchor}: not in taxonomy")
            continue
        print(f"\n=== {anchor}: top {args.top} by variant ===")
        md += [f"### {anchor}", "", "| rank | " + " | ".join(VARIANTS) + " |",
               "|---|" + "---|" * len(VARIANTS)]
        cols = {}
        for v in VARIANTS:
            row = sims[v][skills.index(anchor)]
            top = np.argsort(-row)[:args.top]
            cols[v] = [(skills[t], float(row[t])) for t in top]
        for rank in range(args.top):
            cells = [f"{cols[v][rank][0]} {cols[v][rank][1]:.3f}" for v in VARIANTS]
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