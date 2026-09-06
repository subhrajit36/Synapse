# Synapse — Graph-Based Talent Matching

**Reframes resume-JD matching as graph reasoning over a skill knowledge graph**, not keyword overlap.

---

## What Synapse Does

| Traditional ATS | Synapse |
|-----------------|---------|
| "Candidate lacks *Kubernetes* → reject" | "Candidate has *Docker* → *Kubernetes* is 1 hop away (bridgeable gap)" |
| Black-box score | Explainable: matched skills, bridged gaps, true gaps, path lengths |
| Keyword matching | Graph distance + semantic similarity |

---

## Pipeline

```
Job Description + Candidate Skills
          │
          ▼
Entity Linker (FastEmbed + alias table) → Canonical O*NET skill nodes
          │
          ▼
Skill Knowledge Graph (213 skills, 55 O*NET categories)
  ├─ Role → Skill edges (O*NET "requires")
  └─ Skill ↔ Skill edges (embedding k-NN, cosine similarity)
          │
          ▼
Matcher (Weighted shortest-path)
  ├─ Direct matches: full credit
  ├─ Bridgeable gaps (≤2 hops): partial credit by distance
  └─ True gaps: penalty
          │
          ▼
Ranked candidates + Traceable gap explanations
```

---

## Key Achievements

| Phase | Milestone | Result |
|-------|-----------|--------|
| **A** | Intelligence core | Working end-to-end locally (NetworkX, Gemini Flash extraction) |
| **B** | Evaluation | **7-arm study with bootstrap CIs** — graph reasoning beats cosine-only by **+0.406** on bridge>weak |
| **C** | Productionization | **Ready to start** (FastEmbed → Neo4j AuraDB → FastMCP → Render) |

---

## The Edge Substrate Experiment (Phase B.2)

We tested **4 edge substrates** to find the best skill↔skill adjacency for bridging:

| Substrate | Method | bridge>weak (heldout) | Bridge Precision | Decision |
|-----------|--------|----------------------|------------------|----------|
| **Embedding** ✅ | `BAAI/bge-small-en-v1.5` cosine k-NN | **0.859** [0.761, 0.944] | 48.8% | **Production** |
| Categorical | O*NET Element Name cliques | 0.547 [0.395, 0.695] | 29.2% | Control |
| Typed (sub) | LLM-classified `substitute` only | 0.500 [0.357, 0.635] | **77.9%** | Research |
| Typed (sub+prereq) | + `prerequisite` edges | 0.731 [0.607, 0.848] | 29.8% | Research |

### Why Embedding Won

- **Only substrate meeting the falsification criterion**: `bridge>weak ≥ 0.80` on heldout
- **Beats strong baseline**: +0.406 over cosine-only (0.454) — not a forced mechanism probe
- **Reproducible**: Frozen arm reproduces original Phase B result exactly
- **FastEmbed ready**: CPU-only ONNX, no GPU, fits 512MB RAM

### What Typed Edges Achieved (Research Track)

- **77.9% bridge precision** vs 48.8% — LLM correctly identifies substitutes
- **AUC dropped 0.906 → 0.610** — edges carry signal beyond category membership
- **Failed ranking**: Sparsity (201 edges / 213 nodes) + directed prerequisites + complement exclusion = insufficient connectivity
- **Preserved for future**: Bidirectional prerequisites, node coverage expansion

---

## Bootstrap Confidence Intervals — Why They Matter

| Arm | bridge>weak (95% CI) | nDCG@10 (95% CI) |
|-----|----------------------|------------------|
| **Embedding** | **0.859** [0.761, 0.944] | 0.922 [0.875, 0.959] |
| Categorical | 0.547 [0.395, 0.695] | 0.880 [0.834, 0.923] |
| Typed (sub) | 0.500 [0.357, 0.635] | 0.904 [0.876, 0.930] |
| Cosine-only | 0.454 [0.000, 0.000] | 0.908 [0.000, 0.000] |

**Importance:** With only 15 heldout JDs, point estimates are noisy. The CIs prove:
- Embedding **clearly beats** categorical (CIs don't overlap)
- Embedding **clearly beats** cosine-only (lower bound 0.761 > 0.454)
- Typed edges **cannot claim superiority** (upper bound 0.635 < 0.80 target)

*Protocol: JD-level resampling, 1000 iterations, config swept per-arm on train split only.*

---

## Tech Stack (Locked)

| Layer | Tool |
|-------|------|
| Orchestration | LangGraph |
| LLM Extraction | Gemini Flash (Google AI Studio, free tier) |
| Embeddings | **FastEmbed** (`BAAI/bge-small-en-v1.5`, ONNX, CPU) |
| Graph (dev) | NetworkX + pickle |
| Graph (prod) | **Neo4j AuraDB Free** |
| API | **FastMCP** (HTTP/SSE) |
| Deployment | **Docker → Render Free Web Service** |

---

## Quickstart

```bash
# 1. Setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Build graph (dev: NetworkX pickle)
python -m synapse.graph.build_graph

# 3. Run evaluation (7 arms, bootstrap CIs)
python -m synapse.eval.run_eval_arms

# 4. See results
cat src/synapse/eval/RESULTS.md

# 5. Phase C3: ingest documents through the LangGraph pipeline
export GEMINI_API_KEY=...
python -m synapse.ingest.pipeline data/samples --doc-type resume -v
```

### Ingestion pipeline (Phase C3)

`synapse.ingest.pipeline` wires Reader → Extractor into a LangGraph graph over a
typed `IngestState`. Three things it buys over calling the extractor directly:

- **Retry as a node, not a `try/except`.** A 429 or 503 routes to a `backoff`
  node that waits and re-runs the chunk; a fatal error (bad key, quota) skips
  retrying entirely. The decision and its reason land in state, so a slow run is
  explainable after the fact.
- **One chunk per superstep.** Extraction advances a `cursor`, so the
  checkpointer saves progress between every Gemini call.
- **Resumable batches.** With `--checkpoint` (SQLite, default
  `data/checkpoints/ingest.sqlite`), a run killed during a rate-limit pause
  resumes at the chunk it stopped on instead of re-billing the whole document
  against the 15 RPM free tier. Each document gets its own checkpoint thread.

```bash
python -m synapse.ingest.pipeline data/raw \
    --doc-type resume --checkpoint data/checkpoints/ingest.sqlite --out data/extractions
# re-run the exact same command after an interruption to resume
```

### MCP server (Phase C4)

```bash
python -m synapse.mcp.server --warm -v          # HTTP on 127.0.0.1:8000/mcp
python -m synapse.mcp.server --transport sse    # SSE transport, same tools
python -m synapse.mcp.server --transport stdio  # for local MCP clients
curl localhost:8000/health                      # {"status":"ok",...}
```

Four tools, all read-only, all thin wrappers over `synapse.matching`:

| Tool | Purpose |
|---|---|
| `rank_candidates` | FR5 — rank a pool against a JD, with score components per candidate |
| `get_bridgeable_gaps` | FR4 — split missing skills into bridgeable (with the path) and real gaps |
| `explain_score` | NFR6 — one candidate's full derivation plus the linking trace |
| `graph_stats` | What graph is loaded, how it is configured; doubles as a warm-up call |

Skill names are canonicalized before scoring (`K8s` → `Kubernetes`). Surfaces that
reach no node come back in `jd_unresolved` / `unresolved_skills` rather than
being dropped, because an unresolved JD skill leaves the demand denominator and
would otherwise inflate the score silently. Unreachable gaps report
`distance: -1` with `reason: "no_path"` — MCP payloads are strict JSON and
infinity does not survive the wire.

`/health` deliberately does not touch the graph: Render's free tier cold-starts,
and a probe that unpickled 213 skills would report unhealthy while a healthy
instance was merely waking up. Graph readiness is `graph_stats`.

### Graph source: AuraDB (Phase C2/C6)

AuraDB is the system of record. The graph is materialised into NetworkX once at
first use, so the scoring path stays byte-identical to the one Phase B measured
— `Matcher` runs the same in-process Dijkstra either way. One query, ~1.2MB,
negligible against the 512MB budget.

```bash
NEO4J_URI=neo4j+s://<instance>.databases.neo4j.io
NEO4J_USERNAME=<from the Aura credentials file>
NEO4J_PASSWORD=<from the Aura credentials file>
NEO4J_DATABASE=<from the Aura credentials file>

SYNAPSE_GRAPH_SOURCE=neo4j    # production; fails startup if Aura is unreachable
SYNAPSE_GRAPH_SOURCE=pickle   # tests and offline work; never touches the network
```

There is deliberately **no automatic fallback**. A service that quietly serves a
stale local pickle when Aura is down violates NFR4 while reporting itself
healthy, so `neo4j` raises instead. `graph_stats` reports the active
`graph_source`, and the loader asserts the graph's shape (`SYNAPSE_EXPECTED_SKILLS`,
`SYNAPSE_EXPECTED_PAIRS`, defaulting to 213 / 15,459) so a wrong-shaped deploy
fails at startup rather than producing subtly wrong rankings.

Two things about the migrated data that will otherwise mislead you:

- **`SIMILAR` is stored in both directions** — 30,918 relationships for 15,459
  logical pairs. Compare `count_similar_pairs()` against a NetworkX edge count,
  not `count_edges()["SIMILAR"]`, or it looks like a 2× mismatch.
- **Two roles have no `REQUIRES` edges** (`... Occupations, All Other`), so roles
  must be loaded as nodes, not derived from the edge list.

Verified against the live instance: identical rankings and identical gap
analysis from both sources, 251 nodes / 17,877 edges each.

### Web UI (Phase D)

The same uvicorn process serves the MCP transport and a single static page — one
service, no build step, no Node.

```
GET  /              -> the page
GET  /api/jds       -> JDs in the versioned eval snapshot
POST /api/rank      -> ranked candidates + full MatchResult each (one round trip)
```

Pick a JD on the left, click a candidate on the right to expand its score into
`direct_match_score`, `bridge_score` and `gap_penalty`, its matched skills, its
bridged skills with the path that produced them (`← via <skill>, n hops,
distance d`), its unreachable gaps with a reason code, and the active
`ScoringParams`. Every number is read from `MatchResult`; nothing is computed in
the browser. Each candidate's ground-truth tier from the eval set is shown beside
its score, so the ranking can be checked against the labels.

---

## Known limitation — FR4 is not validated

**FR3 (ranking) is validated. FR4 (gap explanation) is not.** Treat the
bridgeable/gap distinction as unproven, and read the reported path distance and
hop count rather than the label.

The skill graph is **68.5% dense with a diameter of 2** — every skill is within two
hops of every other skill. At the shipped `max_hops=2`, 100% of missing skills
classify as "bridgeable" over random candidate/JD draws, and 0% as real gaps. The
previously reported 48.8% bridgeable-gap precision is what an unconditional
classifier scores when roughly half the answers happen to be "yes".

The cause is the graph build threshold, not the scoring: `strong_sim=0.60` sits
below the *median* all-pair cosine of 0.627, so two unrelated skills are expected
to clear it. Full analysis, the threshold table, and the FR3-vs-FR4 impact split
are in [`data/eval/GRAPH_DENSITY.md`](data/eval/GRAPH_DENSITY.md); the edge-substrate
comparison that selected the embedding graph is in
[`data/eval/EDGE_SUBSTRATE_STUDY.md`](data/eval/EDGE_SUBSTRATE_STUDY.md).

Two related caveats on the reported numbers:

- The bootstrap CIs in `RESULTS.md` predate a fix to `bootstrap_ci`, which had
  been reporting one ranker's interval for every ranker and none at all for the
  frozen baselines. The code is fixed; the numbers have not been re-run.
- `TUNED_PARAMS` now carries `max_bridge_credit=0.9`. Without that ceiling,
  `bridge_credit_scale=2.0` let a bridged skill out-earn a direct match, and a
  candidate holding none of a role's skills outranked one holding all of them.

---

## Project Structure

```
synapse/
├── data/eval/
│   ├── EDGE_SUBSTRATE_STUDY.md    # Complete experiment documentation
│   ├── v2/dataset.json            # 30 JDs × 18 candidates (540 pairs)
│   └── typed_edge_cache.jsonl     # LLM classification cache
├── src/synapse/
│   ├── ingest/
│   │   ├── reader.py             # Node 1: load + chunk
│   │   ├── extractor.py          # Node 2: Gemini structured extraction
│   │   └── pipeline.py           # Phase C3: LangGraph graph + checkpointing
│   ├── matching/        # EntityLinker + Matcher (scoring + gaps)
│   ├── mcp/
│   │   ├── engine.py             # Phase C4: loaded graph + typed tool contracts
│   │   ├── server.py             # Phase C4: FastMCP tools + Phase D routes
│   │   └── static/index.html     # Phase D: the score-decomposition page
│   ├── graph/
│   │   ├── build_graph.py        # Embedding + Categorical graphs
│   │   ├── migrate_to_neo4j.py   # Phase C2: NetworkX → Neo4j
│   │   └── typed_edges.py        # Research: LLM edge classification
│   └── eval/
│       ├── run_eval_arms.py      # 7-arm evaluation + bootstrap
│       ├── RESULTS.md            # Final numbers with CIs
│       └── ABLATION.md
├── CLAUDE.md              # Phase A→C implementation plan
├── Edge_substrate_plan.md # Phase B.2 experiment plan
├── requirements.txt
└── README.md
```


## License

O*NET data © U.S. Department of Labor, used under CC BY 4.0.