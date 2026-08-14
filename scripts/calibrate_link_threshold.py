"""Calibrate the entity-linking similarity threshold with evidence, not a guess.

CLAUDE.md A2.3 says "hard-code a similarity threshold (e.g., 0.82)". That figure
assumes surface forms and canonical names look alike. They do not here: nodes are
O*NET strings like "Amazon Web Services AWS software". This script measures the
actual score distribution so the threshold is chosen from data.

Usage:
    python scripts/calibrate_link_threshold.py
    python scripts/calibrate_link_threshold.py --probes data/eval/link_probes.json

Probe file format - surfaces you have judged by hand:
    {"positives": {"docker": "Docker", "k8s": "Kubernetes"},
     "negatives": ["underwater basket weaving", "conflict resolution"]}

Positives are surfaces that SHOULD link to the given node; negatives are
surfaces that should link to nothing. The best threshold is the one that keeps
positives above it and negatives below it - and the gap between those two
distributions is the number worth reporting in RESULTS.md.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, "src")

from synapse.matching.entity_linker import EntityLinker  # noqa: E402

GRAPH_PATH = "data/skill_graph.pkl"

# Fallback probes so the script is runnable before a labeled file exists.
# Replace these with your own judgments; they are a starting point, not truth.
DEFAULT_PROBES = {
    "positives": {
        "docker": "Docker",
        "kubernetes": "Kubernetes",
        "python": "Python",
        "jenkins": "Jenkins",
        "terraform": "Terraform",
    },
    "negatives": [
        "stakeholder management",
        "underwater basket weaving",
        "team player",
        "excellent communication",
        "willingness to learn",
    ],
}


def load_graph(path: str):
    with open(path, "rb") as fh:
        return pickle.load(fh)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", default=GRAPH_PATH)
    parser.add_argument("--probes", default=None)
    parser.add_argument("--out", default="data/eval/link_threshold_report.md")
    args = parser.parse_args()

    probes = (
        json.loads(Path(args.probes).read_text(encoding="utf-8"))
        if args.probes
        else DEFAULT_PROBES
    )

    G = load_graph(args.graph)
    skills = [n for n, d in G.nodes(data=True) if d.get("node_type") == "skill"]
    node_texts = {
        n: f"{n} ({G.nodes[n].get('category', '')})".strip() for n in skills
    }

    # min_score=0 so every probe reports its raw best score.
    linker = EntityLinker(skills, min_score=0.0, node_texts=node_texts)

    rows: list[tuple[str, str, str, str, float, str]] = []
    pos_scores: list[float] = []
    neg_scores: list[float] = []

    for surface, expected in probes["positives"].items():
        r = linker.link(surface)
        correct = r.node == expected
        rows.append(("positive", surface, expected, r.node or "-", r.score, r.method))
        if r.method == "embedding":
            pos_scores.append(r.score if correct else 0.0)

    for surface in probes["negatives"]:
        r = linker.link(surface)
        rows.append(("negative", surface, "-", r.node or "-", r.score, r.method))
        if r.method == "embedding":
            neg_scores.append(r.score)

    lines = [
        "# Entity-linking threshold calibration",
        "",
        f"Nodes: {len(skills)} | alias hits and surface hits bypass the threshold entirely.",
        "",
        "| kind | surface | expected | linked | score | method |",
        "|---|---|---|---|---|---|",
    ]
    lines += [
        f"| {k} | {s} | {e} | {l} | {sc:.3f} | {m} |" for k, s, e, l, sc, m in rows
    ]

    if pos_scores and neg_scores:
        lo, hi = min(pos_scores), max(neg_scores)
        lines += [
            "",
            f"Lowest correct positive (embedding path): **{lo:.3f}**",
            f"Highest negative (embedding path): **{hi:.3f}**",
        ]
        if lo > hi:
            lines.append(
                f"\nSeparable. Any threshold in ({hi:.3f}, {lo:.3f}] works; "
                f"suggest **{(lo + hi) / 2:.2f}**."
            )
        else:
            lines.append(
                "\n**Not separable.** No threshold cleanly splits these probes. "
                "Add alias entries for the failing positives rather than lowering "
                "the threshold, which would admit the negatives too."
            )
    else:
        lines.append(
            "\nAll probes resolved deterministically (alias/surface). "
            "Add harder surfaces to exercise the embedding path."
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nWritten to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())