"""Deterministic canonicalization layer (Phase A2, steps 3-4).

This module owns the *rules-based* half of entity linking. It runs before any
embedding similarity, because an alias we know is right should never be left to
a cosine score to rediscover.

Note on namespace: canonical node names come from O*NET's "Workplace Example"
column, so they look like "Amazon Web Services AWS software" rather than "AWS".
Everything here exists to bridge that gap between how humans write skills and
how O*NET names them.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict

logger = logging.getLogger(__name__)

# O*NET names are "Vendor Product ..."; strip these to get the recognizable core.
VENDOR_PREFIXES = (
    "microsoft ", "apache ", "oracle ", "ibm ", "google ", "amazon ", "adobe ",
    "atlassian ", "red hat ", "sap ", "esri ", "trimble ", "salesforce ", "cisco ",
    "the mathworks ", "splunk ", "teradata ", "grafana labs ",
)
DROP_SUFFIXES = (" software", " ci", " ide")

# Cores that are common English words -> cause false positives. Never index them.
STOP_CORES = {
    "software", "database", "systems", "system", "server", "platform", "access",
    "development", "cloud", "framework", "enterprise", "studio", "data", "word",
    "analytics", "services", "web", "office", "management", "project",
    "teams", "statistical", "version control", "content", "reporting", "design",
}

# Hand-maintained overrides: surface form -> exact canonical node name.
# Targets are validated against the live graph at construction time; a typo here
# should surface as a warning, not a silently dead entry.
ALIAS_TABLE: dict[str, str] = {
    "aws": "Amazon Web Services AWS software",
    "amazon web services": "Amazon Web Services AWS software",
    "sql": "Structured query language SQL",
    "css": "Cascading style sheets CSS",
    "html": "Hypertext markup language HTML",
    "json": "JavaScript Object Notation JSON",
    "k8s": "Kubernetes",
    "kubernetes": "Kubernetes",
    "js": "JavaScript",
    "node": "Node.js",
    "nodejs": "Node.js",
    "postgres": "PostgreSQL",
    "gcp": "Google Workspace software",  # Closest match in current graph
    "tf": "IBM Terraform",
    "terraform": "IBM Terraform",
}

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w+#.\- ]+")
_ACRONYM = re.compile(r"\b[A-Z][A-Z0-9]{1,5}\b")


def normalize(text: str) -> str:
    """Lowercase, strip stray punctuation, collapse whitespace.

    `+`, `#`, `.` and `-` survive so C++, C#, Node.js and CI-CD stay distinct.
    """
    return _WS.sub(" ", _PUNCT.sub(" ", text)).strip().lower()


def core_form(name: str) -> str:
    """Strip O*NET vendor prefixes and boilerplate suffixes."""
    c = normalize(name)
    for prefix in VENDOR_PREFIXES:
        if c.startswith(prefix):
            c = c[len(prefix):]
            break
    for suffix in DROP_SUFFIXES:
        if c.endswith(suffix):
            c = c[: -len(suffix)]
    return c.strip()


def embedded_acronyms(name: str) -> set[str]:
    """Pull acronyms O*NET buries inside long names ('... SQL' -> 'sql')."""
    return {
        a.lower()
        for a in _ACRONYM.findall(name)
        if a.lower() not in STOP_CORES and len(a) >= 2
    }


def build_surface_index(node_names: list[str]) -> dict[str, str]:
    """Map every unambiguous surface form to its canonical node.

    A surface that would map to two different nodes is dropped entirely rather
    than resolved arbitrarily: an unresolved skill is a visible problem, a
    wrongly-resolved one is an invisible corruption of every downstream score.
    """
    candidates: dict[str, set[str]] = defaultdict(set)
    for node in node_names:
        for surface in {normalize(node), core_form(node), *embedded_acronyms(node)}:
            if surface and surface not in STOP_CORES and len(surface) >= 2:
                candidates[surface].add(node)

    index: dict[str, str] = {}
    for surface, nodes in candidates.items():
        if len(nodes) == 1:
            index[surface] = next(iter(nodes))
        else:
            logger.debug("Ambiguous surface %r maps to %s; dropped", surface, sorted(nodes))
    return index


def validate_alias_table(
    node_names: list[str], alias_table: dict[str, str] | None = None
) -> dict[str, list[str]]:
    """Report alias entries whose target does not exist in the graph.

    Returns {"missing_targets": [...], "shadowed": [...]}. Call this at startup
    and log it: a dead alias is the kind of thing that silently stops working
    when the taxonomy is rebuilt.
    """
    table = ALIAS_TABLE if alias_table is None else alias_table
    node_set = set(node_names)
    missing = sorted(k for k, v in table.items() if v not in node_set)
    if missing:
        logger.warning(
            "%d alias target(s) absent from the graph and therefore dead: %s",
            len(missing), missing,
        )
    return {"missing_targets": missing}


def active_alias_table(
    node_names: list[str], alias_table: dict[str, str] | None = None
) -> dict[str, str]:
    """The alias table filtered down to entries that actually resolve."""
    table = ALIAS_TABLE if alias_table is None else alias_table
    node_set = set(node_names)
    return {normalize(k): v for k, v in table.items() if v in node_set}