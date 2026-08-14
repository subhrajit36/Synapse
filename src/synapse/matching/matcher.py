"""Phase A3 + A4: graph-distance scoring with an explainable score object.

Changes from the previous version, all additive:

  * `match()` accepts either the old bare list of skills or a {skill: weight}
    mapping. Lists behave exactly as before (every weight 1.0), so `app.py` and
    the __main__ demos keep working untouched.
  * Bridging now records BOTH weighted distance and hop count, so Phase B4's
    `max_hops = 1 vs 2` ablation is runnable alongside the distance cutoff.
  * One multi-source Dijkstra per candidate replaces the old
    (missing x candidate) nested shortest-path loop.
  * Scoring constants live in `ScoringParams`, not inline literals - B4 varies
    that object rather than editing this file.
  * Returns a `MatchResult` exposing {total, direct_match_score, bridge_score,
    gap_penalty, matched_skills, bridged_skills, missing_skills} while still
    supporting r["score"] / r["matched"] / r["gaps"] indexing.

Defaults are chosen so `Matcher(G).match(jd, cand)` reproduces the old numbers
exactly. Any deviation in Phase B results is then attributable to a parameter
you changed on purpose.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Iterable, Mapping, Sequence

import networkx as nx

DEFAULT_BRIDGE_CUTOFF = 0.6


@dataclass(frozen=True)
class ScoringParams:
    """Every tunable in one place. Phase B4 sweeps instances of this."""

    bridge_cutoff: float | None = DEFAULT_BRIDGE_CUTOFF
    max_hops: int | None = None          # None = distance cutoff alone decides
    use_weights: bool = True             # False = the B4.1 uniform-weight arm
    enable_bridging: bool = True         # False = the B4.2 direct-match-only arm
    bridge_credit_scale: float = 1.0     # multiplier on (1 - distance) credit
    bridgeable_penalty: float = 0.0      # smaller penalty for a reachable gap
    unreachable_penalty: float = 0.0     # full penalty for a true gap
    proficiency_reference: float = 1.0   # proficiency at or above this = full credit
    min_proficiency_credit: float = 0.0

    def proficiency_credit(self, weight: float) -> float:
        """Map a candidate's proficiency weight to credit in [min, 1.0].

        At the default reference of 1.0, ordinary competence earns full credit and
        a 0.5 passing mention earns half. Above 1.0 is capped: extra proficiency in
        a skill the job needs once should not outscore holding a second skill.
        """
        if not self.use_weights:
            return 1.0
        return max(self.min_proficiency_credit, min(weight / self.proficiency_reference, 1.0))


@dataclass
class Gap:
    skill: str
    via: str | None
    distance: float
    hops: int | None
    bridgeable: bool
    demand: float = 1.0

    def __getitem__(self, key):  # legacy dict access (app.py reads g["skill"] etc.)
        return getattr(self, key)

    def get(self, key, default=None):
        return getattr(self, key, default)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class MatchResult:
    """Explainable score. NFR6: every component is inspectable."""

    total: float = 0.0
    direct_match_score: float = 0.0
    bridge_score: float = 0.0
    gap_penalty: float = 0.0
    total_demand: float = 0.0
    matched_skills: list[str] = field(default_factory=list)
    bridged_skills: list[Gap] = field(default_factory=list)
    missing_skills: list[Gap] = field(default_factory=list)
    name: str = ""

    @property
    def gaps(self) -> list[Gap]:
        """All unmet JD skills, bridgeable first, nearest first."""
        return sorted(self.bridged_skills + self.missing_skills, key=lambda g: g.distance)

    _LEGACY = {"score": "total", "matched": "matched_skills"}

    def __getitem__(self, key):
        if key == "gaps":
            return self.gaps
        return getattr(self, self._LEGACY.get(key, key))

    def __setitem__(self, key, value):
        setattr(self, self._LEGACY.get(key, key), value)

    def get(self, key, default=None):
        try:
            return self[key]
        except AttributeError:
            return default

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "total": self.total,
            "score": self.total,
            "direct_match_score": round(self.direct_match_score, 4),
            "bridge_score": round(self.bridge_score, 4),
            "gap_penalty": round(self.gap_penalty, 4),
            "total_demand": round(self.total_demand, 4),
            "matched_skills": self.matched_skills,
            "bridged_skills": [g.to_dict() for g in self.bridged_skills],
            "missing_skills": [g.to_dict() for g in self.missing_skills],
        }

    def explain(self) -> str:
        lines = [
            f"fit={self.total:.3f}  "
            f"(direct {self.direct_match_score:.2f} + bridge {self.bridge_score:.2f} "
            f"- penalty {self.gap_penalty:.2f}) / demand {self.total_demand:.2f}",
            f"  matched   : {', '.join(self.matched_skills) or '-'}",
        ]
        for g in self.bridged_skills:
            lines.append(f"  bridged   : {g.skill} <- {g.via} (d={g.distance:.2f}, {g.hops} hop)")
        for g in self.missing_skills:
            lines.append(f"  missing   : {g.skill}")
        return "\n".join(lines)


def _as_weight_map(skills: Mapping[str, float] | Iterable[str]) -> dict[str, float]:
    """Accept {skill: weight} or a bare iterable of skills (all weight 1.0)."""
    if isinstance(skills, Mapping):
        return {str(k): float(v) for k, v in skills.items()}
    return {str(s): 1.0 for s in skills}


class Matcher:
    """Score a candidate against a job's required skills using graph bridgeability."""

    def __init__(
        self,
        G: nx.Graph,
        bridge_cutoff: float = DEFAULT_BRIDGE_CUTOFF,
        params: ScoringParams | None = None,
    ) -> None:
        self.G = G
        # `bridge_cutoff` stays a positional arg for backwards compatibility; an
        # explicit params object wins if both are supplied.
        self.params = params or ScoringParams(bridge_cutoff=bridge_cutoff)
        self.bridge_cutoff = self.params.bridge_cutoff

        # Skill-only view; each edge's 'distance' = 1 - similarity.
        self.skill_graph = nx.Graph()
        for u, v, d in G.edges(data=True):
            if d.get("relation") == "similar":
                self.skill_graph.add_edge(u, v, distance=1 - d["weight"])

    # ------------------------------------------------------------------ reachability

    def _reachability(self, sources: Sequence[str]) -> tuple[dict, dict, dict]:
        """Weighted distance, hop count and via-node from a set of held skills.

        Weighted distance and hop count are computed separately on purpose: the
        cheapest weighted path is not always the shortest in hops, and B4.3 asks
        about hops specifically.
        """
        present = [s for s in sources if s in self.skill_graph]
        if not present:
            return {}, {}, {}

        cutoff = self.params.bridge_cutoff
        dist, paths = nx.multi_source_dijkstra(
            self.skill_graph, present, weight="distance", cutoff=cutoff
        )
        hop_cutoff = self.params.max_hops
        hops, _ = nx.multi_source_dijkstra(
            self.skill_graph, present, weight=None, cutoff=hop_cutoff
        )
        via = {node: path[0] for node, path in paths.items()}
        return dist, hops, via

    def _nearest_owned(self, skill: str, candidate_skills: Iterable[str]):
        """Kept for backwards compatibility with any external caller."""
        dist, _, via = self._reachability(list(candidate_skills))
        if skill not in dist:
            return None, float("inf")
        return via.get(skill), dist[skill]

    # ------------------------------------------------------------------ scoring

    def match(
        self,
        jd_skills: Mapping[str, float] | Iterable[str],
        candidate_skills: Mapping[str, float] | Iterable[str],
        params: ScoringParams | None = None,
    ) -> MatchResult:
        p = params or self.params
        jd = _as_weight_map(jd_skills)       # weight = how much the JD demands it
        cand = _as_weight_map(candidate_skills)  # weight = candidate proficiency

        total_demand = sum(jd.values()) if p.use_weights else float(len(jd))
        if not jd:
            return MatchResult()

        matched = [s for s in jd if s in cand]
        missing = [s for s in jd if s not in cand]

        direct = 0.0
        for skill in matched:
            demand = jd[skill] if p.use_weights else 1.0
            direct += demand * p.proficiency_credit(cand[skill])

        dist, hops, via = (
            self._reachability(list(cand)) if (p.enable_bridging and missing) else ({}, {}, {})
        )

        bridged: list[Gap] = []
        unreachable: list[Gap] = []
        bridge_score = 0.0
        penalty = 0.0

        for skill in missing:
            demand = jd[skill] if p.use_weights else 1.0
            d = dist.get(skill, float("inf"))
            h = hops.get(skill)
            within_distance = p.bridge_cutoff is None or d <= p.bridge_cutoff
            within_hops = p.max_hops is None or (h is not None and h <= p.max_hops)
            bridgeable = p.enable_bridging and d != float("inf") and within_distance and within_hops

            gap = Gap(
                skill=skill,
                via=via.get(skill),
                distance=d,
                hops=h,
                bridgeable=bridgeable,
                demand=demand,
            )
            if bridgeable:
                credit = max(0.0, 1 - d) * p.bridge_credit_scale
                bridge_score += demand * credit
                penalty += demand * p.bridgeable_penalty
                bridged.append(gap)
            else:
                penalty += demand * p.unreachable_penalty
                unreachable.append(gap)

        total = (direct + bridge_score - penalty) / total_demand if total_demand else 0.0

        return MatchResult(
            total=round(total, 3),
            direct_match_score=direct,
            bridge_score=bridge_score,
            gap_penalty=penalty,
            total_demand=total_demand,
            matched_skills=sorted(matched),
            bridged_skills=sorted(bridged, key=lambda g: g.distance),
            missing_skills=sorted(unreachable, key=lambda g: g.skill),
        )

    # ------------------------------------------------------------------- FR4 / FR5

    def get_bridgeable_gaps(
        self,
        candidate_skills: Mapping[str, float] | Iterable[str],
        jd_skills: Mapping[str, float] | Iterable[str],
        max_hops: int | None = None,
    ) -> dict[str, list[dict]]:
        """FR4. Argument order follows CLAUDE.md (candidate first)."""
        p = self.params if max_hops is None else ScoringParams(
            **{**asdict(self.params), "max_hops": max_hops}
        )
        r = self.match(jd_skills, candidate_skills, params=p)
        return {
            "bridgeable": [g.to_dict() for g in r.bridged_skills],
            "gaps": [g.to_dict() for g in r.missing_skills],
        }

    def rank(
        self,
        jd_skills: Mapping[str, float] | Iterable[str],
        candidates: Mapping[str, Mapping[str, float] | Iterable[str]],
        params: ScoringParams | None = None,
        top_k: int | None = None,
    ) -> list[MatchResult]:
        """FR5. Ties break on name so rankings are reproducible (NFR7)."""
        results = []
        for name, skills in candidates.items():
            r = self.match(jd_skills, skills, params=params)
            r.name = name
            results.append(r)
        results.sort(key=lambda r: (-r.total, r.name))
        return results[:top_k] if top_k else results


if __name__ == "__main__":
    import sys

    sys.path.insert(0, "src")
    from synapse.graph.build_graph import add_semantic_edges, build_skill_graph
    from synapse.matching.entity_linker import EntityLinker

    G = add_semantic_edges(build_skill_graph())
    skills = [n for n, d in G.nodes(data=True) if d["node_type"] == "skill"]
    node_texts = {n: f"{n} ({G.nodes[n].get('category', '')})" for n in skills}
    linker = EntityLinker(skills, node_texts=node_texts)

    jd = linker.link_many([
        ("Kubernetes", 1.5), ("Docker", 1.5), ("AWS", 1.5), ("Terraform", 1.0),
        ("Jenkins", 1.0), ("Python", 1.0), ("Linux", 1.0), ("Prometheus", 0.5),
    ], source_id="jd_devops").skills

    candidates_raw = {
        "Aisha (DevOps)": ["Kubernetes", "Docker", "AWS", "Terraform",
                           "Jenkins", "Python", "Linux", "Prometheus"],
        "Ravi (Backend)": ["Docker", "Jenkins", "AWS", "Python", "Git", "Linux", "Bash"],
        "Meera (Frontend)": ["React", "JavaScript", "CSS", "HTML", "Figma", "TypeScript"],
        "Sam (Data)": ["Python", "SQL", "TensorFlow", "Tableau", "R"],
    }
    candidates = {n: linker.link_many(raw).skills for n, raw in candidates_raw.items()}

    print("\n=== CANDIDATE RANKING for the DevOps role ===\n")
    for i, r in enumerate(Matcher(G).rank(jd, candidates), 1):
        print(f"{i}. {r.name}")
        print(r.explain(), "\n")