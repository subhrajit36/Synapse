import networkx as nx


class Matcher:
    """Score a candidate against a job's required skills using graph bridgeability."""

    def __init__(self, G, bridge_cutoff=0.6):
        self.G = G
        self.bridge_cutoff = bridge_cutoff
        # A skill-only graph where each edge's 'distance' = 1 - similarity.
        # We bridge gaps by walking skill<->skill 'similar' edges only.
        self.skill_graph = nx.Graph()
        for u, v, d in G.edges(data=True):
            if d.get("relation") == "similar":
                self.skill_graph.add_edge(u, v, distance=1 - d["weight"])

    def _nearest_owned(self, skill, candidate_skills):
        """Closest candidate skill to `skill`, and the weighted distance."""
        if skill not in self.skill_graph:
            return None, float("inf")
        best_skill, best_dist = None, float("inf")
        for cs in candidate_skills:
            if cs not in self.skill_graph:
                continue
            try:
                d = nx.shortest_path_length(
                    self.skill_graph, skill, cs, weight="distance")
            except nx.NetworkXNoPath:
                continue
            if d < best_dist:
                best_skill, best_dist = cs, d
        return best_skill, best_dist

    def match(self, jd_skills, candidate_skills):
        jd = set(jd_skills)
        cand = set(candidate_skills)
        matched = jd & cand
        missing = jd - cand

        gaps = []
        for m in missing:
            via, dist = self._nearest_owned(m, cand)
            gaps.append({
                "skill": m, "via": via, "distance": dist,
                "bridgeable": dist <= self.bridge_cutoff,
            })

        # Scoring: direct match = 1.0; bridgeable gap = partial credit (1 - distance).
        credit = len(matched)
        for g in gaps:
            if g["bridgeable"]:
                credit += max(0.0, 1 - g["distance"])
        score = credit / len(jd) if jd else 0.0

        return {
            "score": round(score, 3),
            "matched": sorted(matched),
            "gaps": sorted(gaps, key=lambda g: g["distance"]),
        }

    def rank(self, jd_skills, candidates):
        """Rank candidates for a job

        candidates: dict {name: [linked skill nodes]}
        returns: list of match results (each with 'name'), best fit first.
        """
        results = []
        for name, skills in candidates.items():
            r = self.match(jd_skills, skills)
            r["name"] = name
            results.append(r)
        results.sort(key=lambda r: r["score"], reverse=True)
        return results
    
if __name__ == "__main__":
    import sys
    sys.path.insert(0, "src")
    from synapse.graph.build_graph import build_skill_graph, add_semantic_edges
    from synapse.matching.entity_linker import EntityLinker

    G = build_skill_graph()
    G = add_semantic_edges(G)
    skills = [n for n, d in G.nodes(data=True) if d["node_type"] == "skill"]
    linker = EntityLinker(skills)

    # The job we're hiring for.
    jd_raw = ["Kubernetes", "Docker", "AWS", "Terraform",
              "Jenkins", "Python", "Linux", "Prometheus"]
    jd = linker.extract(jd_raw)

    # A pool of candidates (raw resume skills).
    candidates_raw = {
        "Aisha (DevOps)":   ["Kubernetes", "Docker", "AWS", "Terraform",
                             "Jenkins", "Python", "Linux", "Prometheus"],
        "Ravi (Backend)":   ["Docker", "Jenkins", "AWS", "Python", "Git", "Linux", "Bash"],
        "Meera (Frontend)": ["React", "JavaScript", "CSS", "HTML", "Figma", "TypeScript"],
        "Sam (Data)":       ["Python", "SQL", "TensorFlow", "Tableau", "R"],
    }
    candidates = {name: linker.extract(raw) for name, raw in candidates_raw.items()}

    ranked = Matcher(G).rank(jd, candidates)

    print("\n=== CANDIDATE RANKING for the DevOps role ===\n")
    for i, r in enumerate(ranked, 1):
        bridgeable = [g["skill"] for g in r["gaps"] if g["bridgeable"]]
        real_gaps = [g["skill"] for g in r["gaps"] if not g["bridgeable"]]
        print(f"{i}. {r['name']:18s} fit={r['score']:.3f}  "
              f"({len(r['matched'])}/{len(jd)} direct)")
        print(f"     bridgeable gaps: {bridgeable}")
        print(f"     real gaps:       {real_gaps}\n")
