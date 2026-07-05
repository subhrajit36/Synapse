import re

# O*NET names are "Vendor Product ..."; strip these to get the recognizable core.
VENDOR_PREFIXES = (
    "microsoft ", "apache ", "oracle ", "ibm ", "google ", "amazon ", "adobe ",
    "atlassian ", "red hat ", "sap ", "esri ", "trimble ", "salesforce ", "cisco ",
    "the mathworks ", "splunk ", "teradata ", "grafana labs ",
)
DROP_SUFFIXES = (" software", " ci", " ide")

# Cores that are common English words -> cause false positives in prose. Skip them.
STOP_CORES = {
    "software", "database", "systems", "system", "server", "platform", "access",
    "development", "cloud", "framework", "enterprise", "studio", "data", "word",
    "analytics", "services", "web", "office", "management", "project",
    "teams", "statistical", "version control", "content", "reporting", "design",
}

# Surface form -> exact graph node, for acronyms/abbreviations we want pinned.
CANONICAL_ALIASES = {
    "aws": "Amazon Web Services AWS software",
    "sql": "Structured query language SQL",
    "css": "Cascading style sheets CSS",
    "html": "Hypertext markup language HTML",
    "json": "JavaScript Object Notation JSON",
    "k8s": "Kubernetes",
}


class SkillExtractor:
    """Find graph skills mentioned in free resume text (gazetteer matching)."""

    def __init__(self, skill_names):
        self.aliases = self._build_aliases(skill_names)

    @staticmethod
    def _core(name):
        c = name.lower().strip()
        for v in VENDOR_PREFIXES:
            if c.startswith(v):
                c = c[len(v):]
                break
        for s in DROP_SUFFIXES:
            if c.endswith(s):
                c = c[:-len(s)]
        return c.strip()

    def _build_aliases(self, skill_names):
        aliases = {}
        for name in skill_names:
            al = {name.lower()}
            core = self._core(name)
            if len(core) >= 3 and core not in STOP_CORES:
                al.add(core)
            aliases[name] = al
        for alias, node in CANONICAL_ALIASES.items():
            if node in aliases:
                aliases[node].add(alias)
        return aliases

    @staticmethod
    def _mentions(alias, text):
        # Word-boundary match that also respects +, #, . (so 'git' != 'github').
        return re.search(r"(?<![\w+#.])" + re.escape(alias) + r"(?![\w+#])", text) is not None

    def extract(self, text):
        t = text.lower()
        return sorted(node for node, al in self.aliases.items()
                      if any(self._mentions(a, t) for a in al))


if __name__ == "__main__":
    import sys, glob
    sys.path.insert(0, "src")
    from synapse.graph.build_graph import build_skill_graph
    from synapse.ingest.resume import read_resume

    G = build_skill_graph()
    skills = [n for n, d in G.nodes(data=True) if d["node_type"] == "skill"]
    extractor = SkillExtractor(skills)

    for path in sorted(glob.glob("data/samples/*.txt")):
        text = read_resume(open(path, "rb").read(), path)
        print(f"{path.split('/')[-1]:22s} -> {extractor.extract(text)}")
