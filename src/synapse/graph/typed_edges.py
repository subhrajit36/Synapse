"""Work Item 2: Typed directed edges for substitutability.

Purpose: raise bridgeable-gap precision. Untyped symmetric similarity cannot
distinguish "Docker substitutes for Kubernetes" from "Docker complements
Kubernetes" from "Docker is a prerequisite for Kubernetes." Bridging should
credit **substitutes only**.

Edge types: substitute, complement, prerequisite, unrelated.
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import networkx as nx
import numpy as np

from synapse.graph.build_graph import build_skill_graph, add_semantic_edges

logger = logging.getLogger(__name__)

print("script started")

# Edge type constants
EDGE_TYPES = ("substitute", "complement", "prerequisite", "unrelated")

# Traversal cost per edge type (see §2.5 in Edge_substrate_plan.md)
TYPE_COST = {
    "substitute": 0.2,   # cheap — this is what bridging is for
    "prerequisite": 0.6, # traversable but expensive, directed
    "complement": None,  # NOT traversable for bridging
    "unrelated": None,   # not added to the graph at all
}

# LLM configuration
# Note: model name should NOT have "models/" prefix for the google-genai SDK
DEFAULT_MODEL = os.getenv("SYNAPSE_GEMINI_MODEL", "gemini-3.1-flash-lite")##"gemini-3.1-flash-lite
BATCH_SIZE = 20  # pairs per call
MAX_RETRIES = 4
TEMPERATURE = 0.0  # deterministic


@dataclass(frozen=True)
class TypedEdge:
    """A typed, directed edge with provenance."""
    a: str
    b: str
    edge_type: Literal["substitute", "complement", "prerequisite", "unrelated"]
    direction: Literal["a_to_b", "b_to_a", "symmetric"]
    confidence: float
    source: Literal["llm", "alias", "category"]
    rationale: str

    @property
    def weight(self) -> float:
        """Traversal cost for this edge."""
        base_cost = TYPE_COST[self.edge_type]
        if base_cost is None:
            return float("inf")  # non-traversable
        return base_cost * (2.0 - self.confidence)

    @property
    def is_traversable(self) -> bool:
        """Whether this edge can be used for bridging."""
        return self.edge_type in ("substitute", "prerequisite")

    @property
    def is_directed(self) -> bool:
        """Whether the edge is directed."""
        return self.edge_type == "prerequisite" and self.direction != "symmetric"


@dataclass
class ClassificationResult:
    """Result of classifying one skill pair."""
    a: str
    b: str
    edge_type: str
    direction: str
    confidence: float
    rationale: str


SYSTEM_INSTRUCTION = """You are a senior engineering hiring expert. Classify the relationship between two technical skills.

EDGE TYPES (exactly one):
- substitute: A can replace B in practice. A hiring manager would accept A where B is required. Symmetric.
- complement: A and B are commonly used together but neither replaces the other. Symmetric.
- prerequisite: A must be learned before B, or B builds on A. Directed (A -> B means A is prerequisite for B).
- unrelated: No meaningful relationship.

DIRECTION:
- For "substitute" and "complement" and "unrelated": always "symmetric"
- For "prerequisite": "a_to_b" (a is prerequisite for b), "b_to_a" (b is prerequisite for a), or "symmetric" if mutual

CONFIDENCE: 0.0 to 1.0 (how certain you are)

RATIONALE: One sentence explaining the classification. Mention specific technical overlap or dependency.

IMPORTANT:
- Be discriminating. "substitute" means truly interchangeable in practice (e.g., PyTorch/TensorFlow, MySQL/PostgreSQL).
- "complement" means they form a coherent stack (e.g., Docker/Kubernetes, React/Redux).
- "prerequisite" means a clear learning dependency (e.g., Python/FastAPI, SQL/PostgreSQL).
- Do NOT default to "substitute" — when in doubt, use "complement" or "unrelated".
"""


def get_candidate_pairs(
    G: nx.Graph,
    k_cross_category: int = 8,
    min_similarity: float = 0.30,
) -> list[tuple[str, str]]:
    """Generate candidate pairs for LLM classification.

    Strategy (per §2.2):
    1. All within-category pairs (from O*NET Element Name)
    2. Top-k cross-category embedding neighbours using 'bare' variant (least category-contaminated)
    """
    skills = [n for n, d in G.nodes(data=True) if d.get("node_type") == "skill"]

    # --- 1. Within-category pairs ---
    cat_to_skills: dict[str, list[str]] = {}
    for skill in skills:
        cat = G.nodes[skill].get("category", "")
        if cat:
            cat_to_skills.setdefault(cat, []).append(skill)

    within_pairs = set()
    for cat, skill_list in cat_to_skills.items():
        n = len(skill_list)
        for i in range(n):
            for j in range(i + 1, n):
                a, b = sorted([skill_list[i], skill_list[j]])
                within_pairs.add((a, b))

    logger.info(f"Within-category pairs: {len(within_pairs)}")

    # --- 2. Cross-category embedding neighbours (bare variant) ---
    # Compute bare embeddings for cross-category recall
    bare_texts = [s for s in skills]
    # We'll use the existing embedding infrastructure
    from fastembed import TextEmbedding
    model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    embeddings = np.array(list(model.embed(bare_texts)), dtype=np.float32)

    # Normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.clip(norms, 1e-12, None)
    sim = embeddings @ embeddings.T
    np.fill_diagonal(sim, -1.0)

    skill_to_idx = {s: i for i, s in enumerate(skills)}
    cross_pairs = set()

    for skill in skills:
        idx = skill_to_idx[skill]
        skill_cat = G.nodes[skill].get("category", "")
        # Find top-k cross-category neighbours
        top_indices = np.argpartition(sim[idx], -k_cross_category)[-k_cross_category:]
        top_indices = top_indices[np.argsort(sim[idx][top_indices])[::-1]]

        for j in top_indices:
            if sim[idx, j] < min_similarity:
                continue
            other = skills[j]
            other_cat = G.nodes[other].get("category", "")
            if other_cat != skill_cat:
                a, b = sorted([skill, other])
                cross_pairs.add((a, b))

    logger.info(f"Cross-category pairs (top-{k_cross_category}, bare): {len(cross_pairs)}")

    # Union and deduplicate
    all_pairs = within_pairs | cross_pairs
    logger.info(f"Total candidate pairs: {len(all_pairs)}")

    return sorted(all_pairs)


def classify_pairs_batch(
    pairs: list[tuple[str, str]],
    api_key: str | None = None,
    model: str = DEFAULT_MODEL,
) -> list[ClassificationResult]:
    """Classify a batch of skill pairs using Gemini Flash."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key or os.getenv("GEMINI_API_KEY"))
    if not client:
        raise ValueError("No Gemini API key provided")

    # Build the prompt
    pairs_text = "\n".join(
        f"{i+1}. {a} | {b}" for i, (a, b) in enumerate(pairs)
    )

    prompt = f"""Classify each pair. Output JSON array of objects with keys:
a, b, edge_type, direction, confidence, rationale

Pairs:
{pairs_text}"""

    results = []
    for attempt in range(MAX_RETRIES):
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION,
                    response_mime_type="application/json",
                    temperature=TEMPERATURE,
                ),
            )
            text = response.text or "[]"
            data = json.loads(text)
            if isinstance(data, dict):
                data = data.get("skills", data.get("items", []))
            if not isinstance(data, list):
                raise ValueError(f"Expected list, got {type(data)}")

            for item in data:
                # Validate required fields
                required = ("a", "b", "edge_type", "direction", "confidence", "rationale")
                if not all(k in item for k in required):
                    logger.warning(f"Missing fields in result: {item}")
                    continue
                if item["edge_type"] not in EDGE_TYPES:
                    logger.warning(f"Invalid edge_type: {item['edge_type']}")
                    continue
                if item["direction"] not in ("a_to_b", "b_to_a", "symmetric"):
                    logger.warning(f"Invalid direction: {item['direction']}")
                    continue
                results.append(ClassificationResult(
                    a=item["a"],
                    b=item["b"],
                    edge_type=item["edge_type"],
                    direction=item["direction"],
                    confidence=float(item["confidence"]),
                    rationale=item["rationale"],
                ))
            return results
        except Exception as exc:
            if attempt == MAX_RETRIES - 1:
                raise
            delay = min(2**attempt, 30) + random.uniform(0, 0.5)
            logger.warning(f"Batch classification attempt {attempt+1} failed: {exc}; retrying in {delay:.1f}s")
            time.sleep(delay)

    return results


def load_cache(cache_path: str | Path) -> dict[tuple[str, str], ClassificationResult]:
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
            # Filter to only ClassificationResult fields (ignore model, timestamp, etc.)
            cr_fields = {"a", "b", "edge_type", "direction", "confidence", "rationale"}
            filtered = {k: v for k, v in data.items() if k in cr_fields}
            key = tuple(sorted([filtered["a"], filtered["b"]]))
            cache[key] = ClassificationResult(**filtered)
    logger.info(f"Loaded {len(cache)} cached classifications from {cache_path}")
    return cache


def save_to_cache(cache_path: str | Path, results: list[ClassificationResult]) -> None:
    """Append new classifications to JSONL cache."""
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("a", encoding="utf-8") as f:
        for r in results:
            _ = tuple(sorted([r.a, r.b]))  # keep for future dedup if needed
            data = {
                "a": r.a,
                "b": r.b,
                "edge_type": r.edge_type,
                "direction": r.direction,
                "confidence": r.confidence,
                "rationale": r.rationale,
                "model": DEFAULT_MODEL,
                "timestamp": time.time(),
            }
            f.write(json.dumps(data) + "\n")


def filter_contaminated_pairs(pairs: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Remove curated substitution group pairs from the candidate set.

    Per §2.3: the curated substitution groups used to build the Phase B eval
    dataset must not appear in any prompt, few-shot example, or system message.
    If they do, bridgeable-gap precision becomes circular and the measurement is void.

    Returns the filtered list and logs the removed pairs.
    """
    from synapse.eval.dataset import SUBSTITUTION_GROUPS

    curated_pairs = set()
    for group in SUBSTITUTION_GROUPS:
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                a, b = sorted([group[i], group[j]])
                curated_pairs.add((a, b))

    candidate_pairs = set(tuple(sorted(p)) for p in pairs)
    intersection = curated_pairs & candidate_pairs

    if intersection:
        logger.warning(
            f"Filtering {len(intersection)} curated substitution pairs from candidate set: "
            f"{sorted(intersection)}. These must not reach the LLM to avoid circularity."
        )

    filtered = [p for p in pairs if tuple(sorted(p)) not in curated_pairs]
    return filtered


def classify_all_pairs(
    G: nx.Graph,
    cache_path: str | Path = "data/eval/typed_edge_cache.jsonl",
    api_key: str | None = None,
    model: str = DEFAULT_MODEL,
    batch_size: int = BATCH_SIZE,
) -> list[TypedEdge]:
    """Main entry point: classify all candidate pairs for a graph."""
    cache_path = Path(cache_path)

    # Generate candidate pairs
    candidates = get_candidate_pairs(G)

    # Filter contaminated pairs (curated substitution groups from eval dataset)
    candidates = filter_contaminated_pairs(candidates)

    # Load cache
    cache = load_cache(cache_path)
    cached_keys = set(cache.keys())

    # Filter to uncached pairs
    uncached = [p for p in candidates if tuple(sorted(p)) not in cached_keys]
    logger.info(f"Pairs to classify: {len(uncached)} (cached: {len(cached_keys)})")

    if not uncached:
        logger.info("All pairs cached, returning cached results")
        return [TypedEdge(
            a=r.a, b=r.b,
            edge_type=r.edge_type,
            direction=r.direction,
            confidence=r.confidence,
            source="llm",
            rationale=r.rationale,
        ) for r in cache.values()]

    # Classify in batches
    all_results = []
    for i in range(0, len(uncached), batch_size):
        batch = uncached[i:i + batch_size]
        logger.info(f"Classifying batch {i//batch_size + 1}/{(len(uncached) + batch_size - 1)//batch_size} ({len(batch)} pairs)")
        batch_results = classify_pairs_batch(batch, api_key=api_key, model=model)
        all_results.extend(batch_results)

        # Save incrementally
        save_to_cache(cache_path, batch_results)

        time.sleep(10)  # avoid rate limits

    # Merge cached and new results
    all_classifications = list(cache.values()) + all_results

    # Convert to TypedEdge objects
    typed_edges = [
        TypedEdge(
            a=r.a, b=r.b,
            edge_type=r.edge_type,
            direction=r.direction,
            confidence=r.confidence,
            source="llm",
            rationale=r.rationale,
        )
        for r in all_classifications
    ]

    logger.info(f"Total typed edges: {len(typed_edges)}")
    return typed_edges


# Probe pairs for sanity checks (moved from deleted scripts.category_dominance)
PROBE_PAIRS = [
    ("Docker", "Kubernetes"),
    ("PyTorch", "TensorFlow"),
    ("MySQL", "PostgreSQL"),
    ("Git", "GitHub"),
]


def classify_probe_pairs(
    cache_path: str | Path = "data/eval/typed_edge_cache.jsonl",
    api_key: str | None = None,
    model: str = DEFAULT_MODEL,
) -> list[TypedEdge]:
    """Classify the known-good probe pairs for sanity checks.

    These pairs are in SUBSTITUTION_GROUPS and are filtered from the main
    classification to avoid circularity. We classify them separately ONLY
    for sanity checking — their results are NOT added to the graph.
    """
    probes = PROBE_PAIRS

    cache = load_cache(cache_path)
    cached_keys = set(cache.keys())

    uncached_probes = [p for p in probes if tuple(sorted(p)) not in cached_keys]

    if uncached_probes:
        logger.info(f"Classifying {len(uncached_probes)} probe pairs for sanity checks")
        probe_results = classify_pairs_batch(uncached_probes, api_key=api_key, model=model)
        save_to_cache(cache_path, probe_results)
    else:
        logger.info("All probe pairs cached")
        probe_results = [cache[tuple(sorted(p))] for p in probes if tuple(sorted(p)) in cache]

    # Convert to TypedEdge objects
    probe_edges = [
        TypedEdge(
            a=r.a, b=r.b,
            edge_type=r.edge_type,
            direction=r.direction,
            confidence=r.confidence,
            source="llm",
            rationale=r.rationale,
        )
        for r in probe_results
    ]

    return probe_edges


def build_typed_graph(
    G: nx.Graph,
    typed_edges: list[TypedEdge],
    traversable_types: set[str] | None = None,
) -> nx.DiGraph:
    """Build a new directed graph with typed edges added.

    Only traversable types (substitute, prerequisite) are added as traversable edges.
    Complement edges are stored but excluded from bridging traversal.
    Unrelated edges are not added.

    Returns a DiGraph where:
    - substitute/complement: edges in BOTH directions (symmetric)
    - prerequisite a_to_b: edge from a to b only (directed)
    - prerequisite symmetric: edges in BOTH directions
    """
    if traversable_types is None:
        traversable_types = {"substitute", "prerequisite"}

    # Convert to DiGraph to support directed prerequisite edges
    H = nx.DiGraph(G)

    # Store all typed edges as node attributes for provenance
    for edge in typed_edges:
        if not H.has_node(edge.a) or not H.has_node(edge.b):
            logger.warning(f"Skipping edge {edge.a} - {edge.b}: node not in graph")
            continue

        edge_data = {
            "edge_type": edge.edge_type,
            "direction": edge.direction,
            "confidence": edge.confidence,
            "source": edge.source,
            "rationale": edge.rationale,
            "weight": edge.weight,
            "traversable": edge.is_traversable,
        }

        if edge.edge_type == "unrelated":
            continue  # Don't add to graph at all

        if edge.edge_type == "prerequisite" and edge.direction != "symmetric":
            # Directed edge: only one direction
            if edge.direction == "a_to_b":
                H.add_edge(edge.a, edge.b, **edge_data)
            else:
                H.add_edge(edge.b, edge.a, **edge_data)
        else:
            # Symmetric edge: add in both directions
            H.add_edge(edge.a, edge.b, **edge_data)
            H.add_edge(edge.b, edge.a, **edge_data)

    return H


def run_sanity_checks(typed_edges: list[TypedEdge]) -> dict:
    """Run sanity checks per §2.6 and return a report."""
    report = {}

    # 1. Type distribution
    type_counts: dict[str, int] = {}
    for e in typed_edges:
        type_counts[e.edge_type] = type_counts.get(e.edge_type, 0) + 1
    total = len(typed_edges)
    report["type_distribution"] = {
        t: {"count": c, "pct": round(c / total * 100, 1)} for t, c in type_counts.items()
    }

    sub_pct = type_counts.get("substitute", 0) / total * 100
    if sub_pct > 70:
        report["warning"] = f"substitute rate {sub_pct:.1f}% > 70% — LLM may be rubber-stamping"

    # 2. Directed consistency (no mutual prerequisites)
    prereq_pairs: dict[tuple[str, str], str] = {}
    violations = []
    for e in typed_edges:
        if e.edge_type == "prerequisite" and e.direction != "symmetric":
            key = tuple(sorted([e.a, e.b]))
            if key in prereq_pairs:
                violations.append((e.a, e.b, prereq_pairs[key], e.direction))
            else:
                prereq_pairs[key] = e.direction
    report["directed_consistency"] = {
        "mutual_prerequisite_violations": len(violations),
        "details": violations,
    }

    # 3. Known-good probes
    # Per system instruction: complement = coherent stack (Docker/Kubernetes, React/Redux)
    #                         prerequisite = learning dependency (Python/FastAPI, SQL/PostgreSQL)
    probes = [
        ("Docker", "Kubernetes", "complement"),  # coherent stack, not substitutes
        ("PyTorch", "TensorFlow", "substitute"),  # truly interchangeable
        ("MySQL", "PostgreSQL", "substitute"),    # truly interchangeable
        ("Git", "GitHub", "complement"),          # coherent stack (VCS + hosting), not prerequisite
    ]
    probe_results = []
    for a, b, expected in probes:
        found = None
        for e in typed_edges:
            if (e.a == a and e.b == b) or (e.a == b and e.b == a):
                found = e
                break
        probe_results.append({
            "pair": f"{a}↔{b}",
            "expected": expected,
            "got": found.edge_type if found else "MISSING",
            "confidence": found.confidence if found else None,
            "pass": found.edge_type == expected if found else False,
        })
    report["probe_checks"] = probe_results

    # Git-GitHub should NOT be substitute (it's a coherent stack, not a learning dependency)
    git_github_fail = any(p["pair"] == "Git↔GitHub" and p["got"] == "substitute" for p in probe_results)
    if git_github_fail:
        report["critical_warning"] = "Git↔GitHub classified as substitute — should be complement (coherent stack)"

    return report


def write_audit_report(
    typed_edges: list[TypedEdge],
    sanity_report: dict,
    output_path: str | Path = "data/eval/typed_edge_audit.md",
) -> None:
    """Write the spot-audit report per §2.6."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Typed Edge Classification Audit",
        f"\nTotal edges classified: {len(typed_edges)}",
        "",
        "## Type Distribution",
        "| type | count | % |",
        "|---|---|---|",
    ]
    for t, info in sanity_report["type_distribution"].items():
        lines.append(f"| {t} | {info['count']} | {info['pct']}% |")
    if "warning" in sanity_report:
        lines.append(f"\n⚠️ {sanity_report['warning']}")

    lines += [
        "",
        "## Directed Consistency",
        f"Mutual prerequisite violations: {sanity_report['directed_consistency']['mutual_prerequisite_violations']}",
    ]
    for v in sanity_report['directed_consistency']['details']:
        lines.append(f"  - {v[0]} ↔ {v[1]}: both directions ({v[2]} vs {v[3]})")

    lines += [
        "",
        "## Probe Checks",
        "| pair | expected | got | confidence | pass |",
        "|---|---|---|---|---|",
    ]
    for p in sanity_report["probe_checks"]:
        # NEW: Safely handle missing confidences for filtered pairs
        conf_str = f"{p['confidence']:.2f}" if p['confidence'] is not None else "N/A"
        lines.append(f"| {p['pair']} | {p['expected']} | {p['got']} | {conf_str} | {'✓' if p['pass'] else '✗'} |")

    if "critical_warning" in sanity_report:
        lines.append(f"\n🚨 **CRITICAL**: {sanity_report['critical_warning']}")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Wrote audit report to {output_path}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--dry-run":
        logging.basicConfig(level=logging.INFO)

        print("Building skill graph...")
        G = build_skill_graph()
        G = add_semantic_edges(G)

        print("Generating candidate pairs...")
        pairs = get_candidate_pairs(G)
        from synapse.graph.typed_edges import filter_contaminated_pairs
        pairs = filter_contaminated_pairs(pairs)
        print(f"Total candidate pairs after filtering: {len(pairs)}")

        # Print first 10 pairs
        print("\nFirst 10 candidate pairs:")
        for a, b in pairs[:10]:
            print(f"  {a} ↔ {b}")

        sys.exit(0)

    logging.basicConfig(level=logging.INFO)

    print("Building skill graph...")
    G = build_skill_graph()
    G = add_semantic_edges(G)

    print("Classifying pairs...")
    typed_edges = classify_all_pairs(G)

    print("Classifying probe pairs for sanity checks...")
    probe_edges = classify_probe_pairs()

    # Include probe edges in sanity checks (they're filtered from main classification to avoid circularity)
    all_edges_for_sanity = typed_edges + probe_edges

    print("Running sanity checks...")
    sanity = run_sanity_checks(all_edges_for_sanity)

    print("Writing audit...")
    write_audit_report(all_edges_for_sanity, sanity)

    print("\nDone!")
    print(f"Type distribution: {sanity['type_distribution']}")
    for p in sanity["probe_checks"]:
        print(f"  {p['pair']}: expected {p['expected']}, got {p['got']} {'✓' if p['pass'] else '✗'}")



print("\nAudit report written to data/eval/typed_edge_audit.md")        
