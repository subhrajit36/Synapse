# Synapse Knowledge Graph & Scoring Documentation

**Generated:** 2026-08-15  
**Purpose:** Internal reference — how the graph is built, what it represents, and how scoring works.  
**Git status:** Ignored (see `.gitignore`)

---

## 1. Knowledge Graph Construction

### 1.1 Source Data: O*NET

We use the **O*NET 30.3** taxonomy (US Department of Labor), specifically:

| File | Purpose |
|------|---------|
| `Occupation Data.txt` | Role definitions (title, SOC code, description) |
| `Software Skills.txt` | Skill-to-role mappings with metadata |

**Location:** `data/taxonomies/onet/db_30_3_text/` (git-ignored; re-download via `scripts/fetch_onet.sh`)

---

### 1.2 Graph Build Pipeline

```mermaid
flowchart TD
    A[O*NET Occupation Data] --> B[Filter SOC 15-xxxx<br/>Software/IT roles]
    C[O*NET Software Skills] --> D[Filter Hot Technology=Y<br/>or In Demand=Y]
    D --> E[Remove STOP_SKILLS<br/>(MS Office, etc.)]
    B --> F[Add Role Nodes<br/>node_type=role]
    E --> G[Add Skill Nodes<br/>node_type=skill, category=Element Name]
    F --> H[Create Graph]
    G --> H
    H --> I[Add role--skill edges<br/>relation=requires]
    I --> J[add_semantic_edges()]
    J --> K[Embed skills with<br/>SentenceTransformer]
    K --> L[Top-k cosine similarity<br/>k=5, min_sim=0.30]
    L --> M[Add skill-skill edges<br/>relation=similar, weight=similarity]
    M --> N[Final Graph<br/>skill_graph.pkl]
```

---

### 1.3 Graph Structure

#### Nodes

| Node Type | Attributes | Count (approx.) |
|-----------|------------|-----------------|
| **Role** | `node_type="role"`, `soc="15-xxxx"`, `Title` | ~25 |
| **Skill** | `node_type="skill"`, `category="Element Name"` (e.g., "Development environment software"), `Workplace Example` | ~200 |

#### Edges

| Relation | Direction | Weight | Meaning |
|----------|-----------|--------|---------|
| `requires` | Role → Skill | 1.0 (implicit) | Role requires this skill (from O*NET) |
| `similar` | Skill ↔ Skill | `cosine_sim ∈ [0.30, 1.0]` | Semantic similarity via embedding |

#### Example Subgraph

```
Software Developer (role)
    │ requires
    ▼
Docker (skill, category="Development environment software")
    │ similar (weight=0.87)
    ▼
Kubernetes (skill, category="Container orchestration software")
    │ similar (weight=0.82)
    ▼
Amazon Web Services AWS software (skill, category="Cloud platform software")
```

---

### 1.4 Key Build Decisions

| Decision | Rationale |
|----------|-----------|
| **SOC 15-xxxx only** | Focus on software/IT roles; avoids noise from unrelated occupations |
| **Hot Technology \|\| In Demand** | Keep only market-relevant tools; drops academic/legacy skills |
| **STOP_SKILLS exclusion** | Generic office tools (Word, PowerPoint) have high frequency but zero signal |
| **Context-enriched embeddings** | `"Docker (Development environment software)"` > `"Docker"` alone — category sharpens similarity |
| **Top-k (k=5) + min_sim=0.30** | Sparse graph; each skill connects to its 5 nearest neighbors above threshold |
| **Undirected similar edges** | Similarity is symmetric; enables bidirectional bridging |

---

### 1.5 Output Artifact

- **File:** `data/skill_graph.pkl` (git-ignored)
- **Format:** NetworkX `Graph` pickle
- **Reload:** `nx.read_gpickle("data/skill_graph.pkl")`

---

## 2. What the Graph Means

### 2.1 Semantic Bridgeability

The `similar` edges encode **skill proximity** — not just co-occurrence, but *semantic relatedness* via embeddings.

**Example:** A candidate knows `Docker`. The graph knows:
- `Docker` --0.87--> `Kubernetes`
- `Docker` --0.72--> `Amazon ECS`
- `Docker` --0.45--> `Jenkins` (CI/CD adjacency, not container runtime)

This enables **bridgeable gaps**: if a JD requires `Kubernetes` and the candidate has `Docker`, the gap is *bridgeable* (1 hop, high similarity) — the candidate can likely learn it quickly.

### 2.2 Distance Metric

For scoring, we convert similarity → **distance**:

```
distance = 1 - similarity
```

| Similarity | Distance | Interpretation |
|------------|----------|----------------|
| 0.90 | 0.10 | Near-identical (e.g., `K8s` ↔ `Kubernetes`) |
| 0.70 | 0.30 | Strongly related (e.g., `Docker` ↔ `Kubernetes`) |
| 0.50 | 0.50 | Moderately related (e.g., `React` ↔ `Vue.js`) |
| 0.30 | 0.70 | Weakly related (threshold floor) |

### 2.3 Dual View: Weighted Distance + Hop Count

The graph supports **two independent path metrics**:

```
                    weighted distance          hop count
Docker ──0.87──► Kubernetes ──0.82──► AWS
  │                                    │
  └── distance: 0.13 + 0.18 = 0.31     └── hops: 2
```

- **Weighted distance** = sum of (1 - similarity) along path → used for *bridge credit*
- **Hop count** = number of edges → used for *max_hops ablation* (B4.3)

---

## 3. Candidate Ranking & Scoring

### 3.1 Pipeline Overview

```mermaid
flowchart TD
    A[Raw Resume Text] --> B[reader.py: chunk_text<br/>400 words, 50 overlap]
    C[Raw JD Text] --> B
    B --> D[extractor.py: Gemini Flash<br/>structured output schema]
    D --> E[ExtractionResult:<br/>List[ExtractedSkill{skill, weight, context}]]
    E --> F[entity_linker.py: cascade resolution]
    F --> G[LinkedProfile:<br/>{canonical_node: proficiency_weight}]
    G --> H[Matcher.match()]
    H --> I[MatchResult:<br/>total, direct, bridge, penalty, gaps]
    I --> J[Matcher.rank()<br/>sorted by total desc]
```

---

### 3.2 Phase A1: Skill Extraction

**Input:** Plain text (resume or JD)  
**Output:** `ExtractionResult` with `ExtractedSkill` objects:

```python
@dataclass
class ExtractedSkill:
    skill: str          # free-text skill phrase
    weight: float       # proficiency ∈ [0.5, 1.5] (validated by Pydantic)
    context: str        # supporting sentence from source text
```

**Key behaviors:**
- Chunking: ~400 words with 50-word overlap (prevents boundary losses)
- Schema validation: LLM output MUST match schema; invalid → retry (max 3)
- Deduplication: `merge_skills()` collapses case variants, keeps highest weight + its context

---

### 3.3 Phase A2: Entity Linking (Canonicalization)

**Cascade resolution (deterministic order):**

```
┌─────────────────────────────────────────────────────────────┐
│  INPUT: "k8s"                                               │
├─────────────────────────────────────────────────────────────┤
│  1. ALIAS TABLE     → "Kubernetes" (score=1.0, METHOD_ALIAS)│
│     └─ Hand-maintained: k8s→Kubernetes, JS→JavaScript, etc. │
├─────────────────────────────────────────────────────────────┤
│  2. SURFACE INDEX   → "Kubernetes" (score=1.0, METHOD_SURFACE)│
│     └─ Normalized match: vendor-stripped, acronym-aware     │
├─────────────────────────────────────────────────────────────┤
│  3. EMBEDDING       → best cosine match (score≥min_score)   │
│     └─ FAIL: METHOD_UNRESOLVED, node=None, logged to JSONL  │
└─────────────────────────────────────────────────────────────┘
```

**Output:** `LinkedProfile` with:
- `skills: dict[canonical_node, proficiency_weight]` — weights preserved!
- `results: List[LinkResult]` — full provenance (method, score)
- `unresolved: List[LinkResult]` — logged for ontology expansion

---

### 3.4 Phase A3/A4: Graph Scoring

#### 3.4.1 Reachability Computation

For a candidate's canonical skill set `C = {s₁, s₂, ...}`, we run **one multi-source Dijkstra** on the skill graph:

```python
dist, hops, via = nx.multi_source_dijkstra(skill_graph, C, weight="distance")
# Also run unweighted for hop count
hops, _ = nx.multi_source_dijkstra(skill_graph, C, weight=None)
```

**Result per JD skill `j`:**
| Metric | Source |
|--------|--------|
| `distance` | Min weighted distance from any `c ∈ C` |
| `hops` | Min hop count from any `c ∈ C` |
| `via` | The specific candidate skill `c` that achieves the minimum |

---

#### 3.4.2 Gap Classification

For each JD skill `j` with demand weight `w_j`:

| Condition | Classification | Score Contribution |
|-----------|----------------|-------------------|
| `j ∈ C` | **Direct match** | `w_j × proficiency_credit(cand_weight)` |
| `j ∉ C` AND `distance ≤ bridge_cutoff` AND `hops ≤ max_hops` | **Bridgeable gap** | `+ w_j × (1 - distance) × bridge_credit_scale - w_j × bridgeable_penalty` |
| Otherwise | **True gap** | `- w_j × unreachable_penalty` |

**Proficiency credit function:**
```python
def proficiency_credit(cand_weight):
    if not use_weights: return 1.0
    return clamp(cand_weight / proficiency_reference, min_proficiency_credit, 1.0)
```
- Default: `proficiency_reference=1.0` → weight 1.0 = full credit, 0.5 = half credit
- Weights > reference are **capped at 1.0** (no runaway scores)

---

#### 3.4.3 Final Score Formula

```
total_demand = Σ w_j  (if use_weights) else count(JD skills)

direct_match_score  = Σ[w_j × proficiency_credit] for matched skills
bridge_score        = Σ[w_j × (1 - distance) × bridge_credit_scale] for bridged skills
gap_penalty         = Σ[w_j × bridgeable_penalty] for bridged + Σ[w_j × unreachable_penalty] for true gaps

total = (direct_match_score + bridge_score - gap_penalty) / total_demand
```

**Score range:** `[-unreachable_penalty, 1.0]` (typically `[0, 1]` with defaults)

---

#### 3.4.4 MatchResult — Explainable Output

```python
@dataclass
class MatchResult:
    total: float                    # final score
    direct_match_score: float       # component 1
    bridge_score: float             # component 2
    gap_penalty: float              # component 3
    total_demand: float             # denominator
    matched_skills: List[str]       # canonical nodes held
    bridged_skills: List[Gap]       # each: skill, via, distance, hops, demand
    missing_skills: List[Gap]       # true gaps
    name: str                       # candidate name (for ranking)
```

**Legacy dict access supported:** `r["score"]`, `r["matched"]`, `r["gaps"]`

---

### 3.5 Ranking (FR5)

```python
ranked = Matcher(G).rank(
    jd_skills=jd_profile.skills,          # {canonical_node: demand_weight}
    candidates={                          # name → {canonical_node: proficiency_weight}
        "Aisha": aisha_profile.skills,
        "Ravi":  ravi_profile.skills,
    },
    top_k=10
)
```

**Tie-breaking:** Alphabetical by name (deterministic, NFR7)

---

## 4. Ablation Knobs (Phase B4 Ready)

All tunables live in `ScoringParams` — **no code changes needed** for ablation:

| Parameter | Default | B4 Variant | Effect |
|-----------|---------|------------|--------|
| `use_weights` | `True` | `False` | Uniform weights (B4.1) |
| `enable_bridging` | `True` | `False` | Direct-match only (B4.2) |
| `max_hops` | `None` | `1` / `2` | Hop radius (B4.3) |
| `bridge_cutoff` | `0.6` | sweep | Distance threshold |
| `bridgeable_penalty` | `0.0` | `>0` | Small penalty for bridged gaps |
| `unreachable_penalty` | `0.0` | `>0` | Full penalty for true gaps |

---

## 5. Example: DevOps Role Scoring

**JD (demand weights):**
```
Kubernetes: 1.5, Docker: 1.5, AWS: 1.5, Terraform: 1.0,
Jenkins: 1.0, Python: 1.0, Linux: 1.0, Prometheus: 0.5
Total demand = 9.0
```

**Candidate: Ravi (Backend) — canonical skills after linking:**
```
Docker: 1.0, Jenkins: 1.0, AWS: 1.0, Python: 1.0, Linux: 1.0
```

**Graph paths (simplified):**
| JD Skill | In Candidate? | Distance | Via | Hops | Classification |
|----------|---------------|----------|-----|------|----------------|
| Kubernetes | No | 0.13 | Docker | 1 | Bridged |
| Docker | Yes | — | — | — | Direct |
| AWS | Yes | — | — | — | Direct |
| Terraform | No | 0.35 | AWS | 2 | Bridged |
| Jenkins | Yes | — | — | — | Direct |
| Python | Yes | — | — | — | Direct |
| Linux | Yes | — | — | — | Direct |
| Prometheus | No | 0.65 | Linux | 2 | True gap (dist > 0.6) |

**Score breakdown:**
```
direct_match_score = 1.5(Docker) + 1.5(AWS) + 1.0(Jenkins) + 1.0(Python) + 1.0(Linux) = 6.0
bridge_score       = 1.5×(1-0.13) + 1.0×(1-0.35) = 1.305 + 0.65 = 1.955
gap_penalty        = 0 (defaults)
total_demand       = 9.0

total = (6.0 + 1.955) / 9.0 = 0.884
```

**MatchResult.explain():**
```
fit=0.884  (direct 6.00 + bridge 1.96 - penalty 0.00) / demand 9.00
  matched   : AWS, Docker, Jenkins, Linux, Python
  bridged   : Kubernetes <- Docker (d=0.13, 1 hop)
              Terraform <- AWS (d=0.35, 2 hops)
  missing   : Prometheus
```

---

## 6. Files Reference

| File | Purpose |
|------|---------|
| `src/synapse/graph/build_graph.py` | Graph construction from O*NET |
| `src/synapse/matching/aliases.py` | Normalization, alias table, surface index |
| `src/synapse/matching/entity_linker.py` | Cascade linking, weight preservation |
| `src/synapse/matching/matcher.py` | Scoring, reachability, ranking |
| `src/synapse/ingest/reader.py` | Document reading + chunking |
| `src/synapse/ingest/extractor.py` | LLM extraction with schema validation |
| `src/synapse/ingest/schemas.py` | Pydantic models |
| `data/skill_graph.pkl` | Serialized graph (git-ignored) |
| `tests/test_*.py` | 56 unit tests (all passing) |

---

## 7. Phase C Migration Notes

| Component | Current | Phase C Target |
|-----------|---------|----------------|
| Embedder | `sentence-transformers` (torch) | `fastembed` (ONNX, CPU-only) |
| Graph storage | NetworkX + pickle | Neo4j AuraDB Free |
| Linker threshold | `DEFAULT_MIN_SCORE=0.60` | Calibrated via `scripts/calibrate_link_threshold.py` |
| Dynamic MERGE | Not implemented | Guarded by same canonicalization + threshold check |

---

*End of document.*