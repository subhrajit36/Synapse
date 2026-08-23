"""Answer 'why won't A bridge to B?' with evidence instead of guesses.

    python scripts/diagnose_bridge.py
    python scripts/diagnose_bridge.py --pair PyTorch TensorFlow --control Docker Kubernetes
    python scripts/diagnose_bridge.py --legacy     # build the OLD graph, for comparison

There are four distinct ways bridging fails and they need opposite fixes:

    1. the node isn't in the graph          -> entity-linking / taxonomy problem
    2. the node is an orphan                -> the builder never linked it
    3. the edge exists but is too weak      -> min_sim vs bridge_cutoff mismatch
    4. the edge was crowded out of top-k    -> ranking budget problem

This script names which one you have. Run it before changing any constant.
"""

from __future__ import annotations

import argparse
import sys

sys.path.insert(0, "src")

from synapse.graph.build_graph import (  # noqa: E402
    CUSTOM_SKILLS, _embed_text, add_seed_edges, add_semantic_edges,
    audit_graph, build_skill_graph,
)
from synapse.matching.matcher import Matcher, ScoringParams  # noqa: E402


def rank_of(G, anchor: str, target: str, model_name="all-MiniLM-L6-v2"):
    """Where does `target` sit in `anchor`'s raw similarity ranking?

    This is the question top-k answers implicitly. If the true rank is > k, the
    edge was crowded out and no threshold change will bring it back.
    """
    from sentence_transformers import SentenceTransformer, util

    skills = [n for n, d in G.nodes(data=True) if d.get("node_type") == "skill"]
    if anchor not in skills or target not in skills:
        return None
    texts = [_embed_text(G, s) for s in skills]
    emb = SentenceTransformer(model_name).encode(texts, convert_to_tensor=True)
    sim = util.cos_sim(emb, emb)
    i = skills.index(anchor)
    row = sim[i].clone()
    row[i] = -1.0
    order = row.argsort(descending=True).tolist()
    ranked = [(skills[j], float(row[j])) for j in order]
    rank = next(r for r, (s, _) in enumerate(ranked, 1) if s == target)
    return rank, dict(ranked)[target], ranked[:10]


def describe_node(G, name: str) -> str:
    if name not in G:
        return f"  {name:<14} ABSENT from the graph"
    d = G.nodes[name]
    return (f"  {name:<14} type={d.get('node_type')} source={d.get('source', '?')}\n"
            f"  {'':<14} category      = {d.get('category', '-')!r}\n"
            f"  {'':<14} embed_category= {d.get('embed_category', '-')!r}\n"
            f"  {'':<14} embeds as     = {_embed_text(G, name)!r}")


def report(G, a: str, b: str, cutoff: float, k: int, check_rank: bool):
    print(f"\n{'=' * 68}\n  {a}  ->  {b}\n{'=' * 68}")
    print(describe_node(G, a))
    print(describe_node(G, b))

    if a not in G or b not in G:
        print("\n  VERDICT: missing node. Fix the taxonomy/CUSTOM_SKILLS, not the cutoff.")
        return

    if G.has_edge(a, b):
        d = G[a][b]
        sim = d.get("weight")
        print(f"\n  direct edge: relation={d.get('relation')} weight={sim} "
              f"source={d.get('edge_source', '-')}")
        if d.get("relation") == "similar":
            dist = 1 - sim
            verdict = "traversable" if dist <= cutoff else "PRESENT BUT UNTRAVERSABLE"
            print(f"  distance = 1 - {sim} = {dist:.3f}  vs bridge_cutoff {cutoff}"
                  f"  -> {verdict}")
    else:
        print("\n  no direct edge")

    m = Matcher(G, params=ScoringParams(bridge_cutoff=cutoff))
    print("\n" + m.debug_bridge([a], b))

    if check_rank:
        res = rank_of(G, a, b)
        if res:
            rank, sim, top = res
            print(f"\n  raw similarity rank of {b} among {a}'s neighbours: "
                  f"#{rank} (sim {sim:.3f}), top-k budget is {k}")
            if rank > k:
                print(f"  -> CROWDED OUT: rank {rank} > k={k}. The threshold pass "
                      f"(strong_sim) or a larger k is the fix, not bridge_cutoff.")
            print(f"  {a}'s true top 10:")
            for s, v in top:
                print(f"    {s:<32} {v:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", nargs=2, default=["PyTorch", "TensorFlow"])
    ap.add_argument("--control", nargs=2, default=["Docker", "Kubernetes"])
    ap.add_argument("--cutoff", type=float, default=0.6)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--legacy", action="store_true",
                    help="build the pre-fix graph (no category override, no "
                         "threshold pass, no seed edges) to reproduce the bug")
    ap.add_argument("--no-rank", action="store_true",
                    help="skip the rank check (it re-embeds, ~10s)")
    args = ap.parse_args()

    G = build_skill_graph()
    if args.legacy:
        print(">>> LEGACY graph: original behaviour, for comparison")
        G = add_semantic_edges(G, k=args.k, strong_sim=None, use_embed_category=False)
    else:
        G = add_semantic_edges(G, k=args.k)
        G = add_seed_edges(G)

    audit_graph(G, bridge_cutoff=args.cutoff)

    # Which CUSTOM_SKILLS collided with O*NET? That collision is the root of the
    # split-vocabulary problem, so name the members explicitly.
    collided = [i["skill"] for i in CUSTOM_SKILLS
                if G.nodes.get(i["skill"], {}).get("source") == "onet"]
    print(f"\nCUSTOM_SKILLS already present in O*NET (category came from O*NET): "
          f"{collided or 'none'}")

    report(G, *args.pair, cutoff=args.cutoff, k=args.k, check_rank=not args.no_rank)
    report(G, *args.control, cutoff=args.cutoff, k=args.k, check_rank=not args.no_rank)


if __name__ == "__main__":
    main()