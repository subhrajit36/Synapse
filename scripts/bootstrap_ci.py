"""Bootstrap confidence intervals for Phase B metrics.

Resamples JDs with replacement (1000 iterations) and reports 95% CI on
bridge>weak, nDCG@10, P@5, and bridgeable-gap precision.
"""
from __future__ import annotations

import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, "src")

from synapse.eval.dataset import load
from synapse.eval.metrics import ndcg_at_k, pairwise_accuracy, precision_at_k
from synapse.matching.matcher import Matcher, ScoringParams
from synapse.graph.build_graph import build_skill_graph, add_semantic_edges, add_seed_edges


def compute_metrics(matcher, jds, params):
    """Compute all metrics for a list of JDs."""
    wins2 = total2 = 0
    ndcgs = []
    p5s = []
    for jd in jds:
        cands = {c["cand_id"]: c["skills"] for c in jd["candidates"]}
        grades = {c["cand_id"]: c["grade"] for c in jd["candidates"]}
        ranked = matcher.rank(jd["jd_skills"], cands, params=params)
        scores = {r.name: r.total for r in ranked}
        ranked_ids = [r.name for r in ranked]
        w, t = pairwise_accuracy(scores, grades, high=2, low=1)
        wins2 += w
        total2 += t
        ndcgs.append(ndcg_at_k(list(cands), scores, grades, 10))
        p5s.append(precision_at_k(ranked_ids, grades, 5))

    return {
        "bridge>weak": wins2 / total2 if total2 else 0.0,
        "nDCG@10": float(np.mean(ndcgs)) if ndcgs else 0.0,
        "P@5": float(np.mean(p5s)) if p5s else 0.0,
    }


def bootstrap_ci(metric_values, n_iter=1000, ci=0.95):
    """Bootstrap CI on a list of per-JD metric values."""
    if not metric_values:
        return (0.0, 0.0)
    metric_values = np.array(metric_values)
    n = len(metric_values)
    boots = []
    for _ in range(n_iter):
        idx = np.random.choice(n, n, replace=True)
        boots.append(np.mean(metric_values[idx]))
    boots = np.array(boots)
    alpha = (1 - ci) / 2
    return (float(np.percentile(boots, alpha * 100)),
            float(np.percentile(boots, (1 - alpha) * 100)))


def bootstrap_metrics(matcher, jds, params, n_iter=1000):
    """Bootstrap all metrics by resampling JDs."""
    jd_metrics = []
    for jd in jds:
        cands = {c["cand_id"]: c["skills"] for c in jd["candidates"]}
        grades = {c["cand_id"]: c["grade"] for c in jd["candidates"]}
        ranked = matcher.rank(jd["jd_skills"], cands, params=params)
        scores = {r.name: r.total for r in ranked}
        ranked_ids = [r.name for r in ranked]
        w, t = pairwise_accuracy(scores, grades, high=2, low=1)
        ndcg = ndcg_at_k(list(cands), scores, grades, 10)
        p5 = precision_at_k(ranked_ids, grades, 5)
        jd_metrics.append({
            "bridge>weak": (w, t),  # wins, total
            "nDCG@10": ndcg,
            "P@5": p5,
        })

    # Bootstrap by resampling JDs
    n = len(jd_metrics)
    boot_results = defaultdict(list)

    for _ in range(n_iter):
        idx = np.random.choice(n, n, replace=True)
        sample = [jd_metrics[i] for i in idx]

        # Aggregate bridge>weak
        total_w = sum(m["bridge>weak"][0] for m in sample)
        total_t = sum(m["bridge>weak"][1] for m in sample)
        boot_results["bridge>weak"].append(total_w / total_t if total_t else 0.0)

        boot_results["nDCG@10"].append(np.mean([m["nDCG@10"] for m in sample]))
        boot_results["P@5"].append(np.mean([m["P@5"] for m in sample]))

    cis = {}
    for metric, values in boot_results.items():
        values = np.array(values)
        alpha = 0.025
        ci_low = float(np.percentile(values, alpha * 100))
        ci_high = float(np.percentile(values, (1 - alpha) * 100))
        cis[metric] = (ci_low, ci_high)

    return cis


def main():
    # Load ORIGINAL Phase B graph (sentence-transformers, frozen)
    import pickle
    with open("data/skill_graph.pkl", "rb") as f:
        G = pickle.load(f)

    matcher = Matcher(G)

    train = load("v2", split="train")["jds"]
    heldout = load("v2", split="heldout")["jds"]

    # Phase B tuned config (from ablation.py output)
    params = ScoringParams(
        bridge_cutoff=0.7,
        bridge_credit_scale=2.0,
        unreachable_penalty=0.0,
        max_hops=2,
    )

    print(f"Computing bootstrap CIs ({1000} iterations) on HELDOUT ({len(heldout)} JDs)...")
    cis = bootstrap_metrics(matcher, heldout, params, n_iter=1000)

    # Point estimates
    point = compute_metrics(matcher, heldout, params)

    print(f"\n=== Embedding Arm - Bootstrap 95% CI (HELDOUT) ===")
    for metric in ["bridge>weak", "nDCG@10", "P@5"]:
        low, high = cis[metric]
        pt = point[metric]
        print(f"  {metric}: {pt:.3f} [{low:.3f}, {high:.3f}]")

    # Also run on train for comparison
    print(f"\nComputing on TRAIN ({len(train)} JDs)...")
    cis_train = bootstrap_metrics(matcher, train, params, n_iter=1000)
    point_train = compute_metrics(matcher, train, params)

    print(f"\n=== Embedding Arm - Bootstrap 95% CI (TRAIN) ===")
    for metric in ["bridge>weak", "nDCG@10", "P@5"]:
        low, high = cis_train[metric]
        pt = point_train[metric]
        print(f"  {metric}: {pt:.3f} [{low:.3f}, {high:.3f}]")

    # For categorical arm
    from synapse.graph.build_graph import build_categorical_graph
    G_cat = build_categorical_graph(verbose=False)
    matcher_cat = Matcher(G_cat)
    params_cat = ScoringParams(
        bridge_cutoff=0.5,
        bridge_credit_scale=2.0,
        unreachable_penalty=0.0,
        max_hops=1,
    )

    print(f"\n=== Categorical Arm - Bootstrap 95% CI (HELDOUT) ===")
    cis_cat = bootstrap_metrics(matcher_cat, heldout, params_cat, n_iter=1000)
    point_cat = compute_metrics(matcher_cat, heldout, params_cat)
    for metric in ["bridge>weak", "nDCG@10", "P@5"]:
        low, high = cis_cat[metric]
        pt = point_cat[metric]
        print(f"  {metric}: {pt:.3f} [{low:.3f}, {high:.3f}]")


if __name__ == "__main__":
    np.random.seed(42)
    main()
    