# Synapse — A Relationship-Aware Talent Graph Agent

Traditional Applicant Tracking Systems rank candidates by **keyword overlap**: a candidate
with *Docker* and *distributed systems* but not *Kubernetes* is scored as a poor match for a
Kubernetes role — even though that gap is small and bridgeable.

**Synapse** reframes talent matching as **graph reasoning**. Skills, roles, and candidates are
nodes in a weighted knowledge graph; the system scores role fit using both semantic similarity
and **graph distance**, and explains every decision:

> *"Strong match — the candidate has Docker; the Kubernetes gap is one hop away and bridgeable."*

## How it works

```
Job description + candidate skills
        │  entity-link phrases → graph nodes (embeddings; handles "k8s" → Kubernetes)
        ▼
Skill Knowledge Graph  ── role→skill edges from O*NET
        │               └─ skill↔skill edges from k-NN over sentence embeddings
        ▼
Score: direct matches (full credit) + bridgeable gaps (partial credit by graph distance)
        ▼
Ranked candidates + explainable, traceable gaps
```

- **Graph vocabulary:** [O*NET 30.3](https://www.onetcenter.org/) (US Dept. of Labor), filtered
  to software roles (SOC family 15), keeping Hot / In-Demand technologies.
- **Skill adjacency:** a k-nearest-neighbor graph over context-enriched
  [sentence-transformers](https://www.sbert.net/) embeddings (`all-MiniLM-L6-v2`).
- **Matching:** weighted shortest-path "bridgeability" over skill↔skill edges.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash scripts/fetch_onet.sh          # downloads O*NET data (~110 MB, not committed)

python src/synapse/graph/build_graph.py       # build the knowledge graph
python src/synapse/matching/matcher.py        # run the example candidate/JD match
```

## Layout

```
src/synapse/
  graph/build_graph.py       # O*NET → role/skill graph + embedding skill edges
  matching/entity_linker.py  # free-text skill phrases → graph nodes
  matching/matcher.py        # candidate vs job-description scoring + gap explanation
notebooks/                   # step-by-step learning scripts
scripts/fetch_onet.sh        # re-download the O*NET dataset
```

## Status

MVP core working: knowledge graph (255 nodes, 3.2k edges), entity linking, and explainable
bridgeable-gap matching. Roadmap: multi-candidate ranking, web UI, PDF resume ingestion,
PMI co-occurrence edges, fairness/bias audit.

## Data & license

O*NET data © U.S. Department of Labor, Employment and Training Administration, used under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). O*NET® is a trademark of USDOL/ETA.
