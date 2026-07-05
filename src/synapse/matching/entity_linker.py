from sentence_transformers import SentenceTransformer, util


class EntityLinker:
    """Map free-text skill phrases onto graph skill nodes via embedding similarity."""

    def __init__(self, skill_names, model_name="all-MiniLM-L6-v2", min_score=0.5):
        self.model = SentenceTransformer(model_name)
        self.skills = list(skill_names)
        # Encode every graph skill ONCE, up front, so linking is just a lookup.
        self.node_emb = self.model.encode(self.skills, convert_to_tensor=True)
        self.min_score = min_score

    def link(self, phrase):
        """Best-matching graph node for a phrase, or None if below the threshold."""
        q = self.model.encode(phrase, convert_to_tensor=True)
        sims = util.cos_sim(q, self.node_emb)[0]     # similarity to every node
        j = int(sims.argmax())                        # index of the closest node
        score = sims[j].item()
        if score < self.min_score:
            return None, score                        # out-of-vocabulary
        return self.skills[j], score

    def extract(self, phrases):
        """Link many phrases; return the list of graph nodes we confidently matched."""
        matched = []
        for p in phrases:
            node, _ = self.link(p)
            if node is not None:
                matched.append(node)
        return matched


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "src")   # so we can import the graph builder
    from synapse.graph.build_graph import build_skill_graph

    G = build_skill_graph()
    skills = [n for n, d in G.nodes(data=True) if d["node_type"] == "skill"]
    linker = EntityLinker(skills)

    resume_skills = ["Docker", "Jenkins", "AWS", "Python", "Git",
                     "SQL", "distributed systems", "Photoshop"]
    print("Linking a candidate's skills to the graph:\n")
    for p in resume_skills:
        node, score = linker.link(p)
        status = f"{node}" if node else f"(dropped — best only {score:.2f})"
        print(f"  {p:22s} -> {status}")
