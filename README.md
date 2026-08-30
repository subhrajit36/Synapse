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
│   │   └── server.py             # Phase C4: FastMCP tools, HTTP/SSE transport
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