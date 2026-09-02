"""Phase B4: parameter sweep + ablation.

Protocol (this is the part that makes the result mean anything):

    1. Sweep ScoringParams on the TRAIN split only.
    2. Pick the best config by `bridge>weak` (ties broken by nDCG@10).
    3. Report that single config on the HELDOUT split, which was never consulted
       during selection.

Tuning on the reported split would make any "improvement" unfalsifiable - the
number would describe the search, not the method. The heldout row is the claim.

Ablation arms answer "which component earns its keep?" rather than "how high can
the number go": bridging on/off isolates the graph itself, uniform weights
isolate JD demand weighting, and max_hops isolates the bridging radius.
"""

from __future__ import annotations

import itertools
import pickle
import sys
from pathlib import Path

from synapse.eval.dataset import load
from synapse.eval.metrics import ndcg_at_k, pairwise_accuracy
from synapse.matching.matcher import Matcher, ScoringParams

GRAPH_PATH = "data/skill_graph.pkl"

# Grid. Chosen around the current defaults (cutoff .6, scale 1.0, penalty 0) so
# the baseline config is inside the search space and can win on its own merits.
GRID = {
    "bridge_cutoff": [0.40, 0.50, 0.60, 0.70],
    "bridge_credit_scale": [1.0, 1.5, 2.0],
    # Swept because `bridge_credit_scale` alone cannot express "reward bridges
    # strongly, but never above a direct match". Leaving it out is what let the
    # previous sweep select a config that ranked a candidate holding none of the
    # required skills above one holding all of them. 1.0 is included so the
    # uncapped behaviour stays inside the search space and can win on merit.
    #
    # Enabling this makes the sweep 324 configs instead of 108 and will select a
    # different config, so results stop being comparable to the reported ones.
    # `run_eval_arms.py` deliberately does NOT sweep it for that reason; this
    # grid is the one to use when a full re-sweep is intended.
    # None keeps the uncapped behaviour - the one that produced the published
    # numbers - inside the search space so it can win on merit.
    "max_bridge_credit": [None, 0.75, 0.90, 1.0],
    "unreachable_penalty": [0.0, 0.25, 0.5],
    "max_hops": [1, 2, None],
}


def evaluate_params(matcher: Matcher, jds: list[dict], params: ScoringParams) -> dict:
    """bridge>weak (pooled over pairs) and mean nDCG@10 for one config."""
    wins2 = total2 = 0
    ndcgs = []
    for jd in jds:
        cands = {c["cand_id"]: c["skills"] for c in jd["candidates"]}
        grades = {c["cand_id"]: c["grade"] for c in jd["candidates"]}
        scores = {r.name: r.total
                  for r in matcher.rank(jd["jd_skills"], cands, params=params)}
        w, t = pairwise_accuracy(scores, grades, high=2, low=1)
        wins2 += w
        total2 += t
        ndcgs.append(ndcg_at_k(list(cands), scores, grades, 10))
    return {
        "bridge>weak": wins2 / total2 if total2 else 0.0,
        "nDCG@10": sum(ndcgs) / len(ndcgs) if ndcgs else 0.0,
    }


def sweep(matcher: Matcher, jds: list[dict]) -> list[tuple[dict, dict]]:
    keys = list(GRID)
    results = []
    for combo in itertools.product(*(GRID[k] for k in keys)):
        cfg = dict(zip(keys, combo))
        m = evaluate_params(matcher, jds, ScoringParams(**cfg))
        results.append((cfg, m))
    results.sort(key=lambda r: (-r[1]["bridge>weak"], -r[1]["nDCG@10"]))
    return results


def ablations(matcher: Matcher, jds: list[dict], best: dict) -> dict[str, dict]:
    """B4 arms, each a one-factor change from the selected config."""
    arms = {
        "selected config": ScoringParams(**best),
        "no bridging (direct match only)": ScoringParams(**{**best, "enable_bridging": False}),
        "uniform weights (no JD demand)": ScoringParams(**{**best, "use_weights": False}),
        "max_hops = 1": ScoringParams(**{**best, "max_hops": 1}),
        "max_hops = 2": ScoringParams(**{**best, "max_hops": 2}),
        "default (pre-tuning)": ScoringParams(),
    }
    return {name: evaluate_params(matcher, jds, p) for name, p in arms.items()}


def _fmt(cfg: dict) -> str:
    return ", ".join(f"{k}={v}" for k, v in cfg.items())


def main():
    G = pickle.load(open(GRAPH_PATH, "rb"))
    matcher = Matcher(G)
    train = load("v2", split="train")["jds"]
    heldout = load("v2", split="heldout")["jds"]

    print(f"sweeping {len(list(itertools.product(*GRID.values())))} configs "
          f"on TRAIN ({len(train)} JDs)...")
    results = sweep(matcher, train)

    print("\n--- top 8 configs on TRAIN ---")
    for cfg, m in results[:8]:
        print(f"  bridge>weak={m['bridge>weak']:.3f}  nDCG={m['nDCG@10']:.3f}  {_fmt(cfg)}")

    best_cfg, best_train = results[0]
    default_train = evaluate_params(matcher, train, ScoringParams())
    best_heldout = evaluate_params(matcher, heldout, ScoringParams(**best_cfg))
    default_heldout = evaluate_params(matcher, heldout, ScoringParams())

    print(f"\nselected on train: {_fmt(best_cfg)}")
    print(f"  TRAIN   bridge>weak {default_train['bridge>weak']:.3f} -> "
          f"{best_train['bridge>weak']:.3f}")
    print(f"  HELDOUT bridge>weak {default_heldout['bridge>weak']:.3f} -> "
          f"{best_heldout['bridge>weak']:.3f}   "
          f"(nDCG {default_heldout['nDCG@10']:.3f} -> {best_heldout['nDCG@10']:.3f})")

    print("\n--- ablation on HELDOUT ---")
    abl = ablations(matcher, heldout, best_cfg)
    for name, m in abl.items():
        print(f"  {name:34s} bridge>weak={m['bridge>weak']:.3f}  nDCG={m['nDCG@10']:.3f}")

    md = [
        "# Phase B4 - Parameter sweep & ablation",
        "",
        f"Swept {len(results)} configs on the **train** split ({len(train)} JDs); "
        f"the winner is reported on the **heldout** split ({len(heldout)} JDs), "
        "which was not consulted during selection.",
        "",
        f"Selected config: `{_fmt(best_cfg)}`",
        "",
        "## Tuned vs default",
        "",
        "| split | config | bridge>weak | nDCG@10 |",
        "|---|---|---|---|",
        f"| train | default | {default_train['bridge>weak']:.3f} | {default_train['nDCG@10']:.3f} |",
        f"| train | tuned | {best_train['bridge>weak']:.3f} | {best_train['nDCG@10']:.3f} |",
        f"| heldout | default | {default_heldout['bridge>weak']:.3f} | {default_heldout['nDCG@10']:.3f} |",
        f"| **heldout** | **tuned** | **{best_heldout['bridge>weak']:.3f}** | "
        f"**{best_heldout['nDCG@10']:.3f}** |",
        "",
        "## Ablation (heldout)",
        "",
        "| arm | bridge>weak | nDCG@10 |",
        "|---|---|---|",
    ]
    md += [f"| {n} | {m['bridge>weak']:.3f} | {m['nDCG@10']:.3f} |" for n, m in abl.items()]
    md.append("")

    out = Path("src/synapse/eval/ABLATION.md")
    out.write_text("\n".join(md))
    print(f"\nwrote {out}")
    return best_cfg


if __name__ == "__main__":
    sys.path.insert(0, "src")
    main()
