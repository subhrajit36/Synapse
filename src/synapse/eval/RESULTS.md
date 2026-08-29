# Phase B - Evaluation Results (Work Item 4: 7 Arms + Bootstrap CIs)

Dataset `v2` - 30 JDs, 540 pairs, seed 42, split train=15/heldout=15. Relevant = grade >= 2.

`bridge>weak` is the headline: pairwise accuracy on bridgeable(2) vs weak(1), the decision the graph exists to get right. By construction the weak tier has ~4x more exact JD-skill overlap than the bridgeable tier, so a bag-of-skills ranker must invert the correct order.

Each arm re-sweeps its config on the **train** split only; the reported heldout numbers are never consulted during selection. Bootstrap 95% CIs (JD-level resampling, 1000 iterations) are shown in brackets.

## Summary: All Arms on Heldout Split

| Arm | bridge>weak | nDCG@10 | P@5 | MRR | Bridge Precision |
|---|---|---|---|---|---|
| embedding | 0.859 [0.761, 0.944] | 0.922 [0.875, 0.959] | 0.960 [0.920, 1.000] | 0.933 [0.833, 1.000] | 0.488 [0.403, 0.578] |
| categorical | 0.547 [0.395, 0.695] | 0.880 [0.834, 0.923] | 0.787 [0.653, 0.907] | 1.000 [1.000, 1.000] | 0.292 [0.240, 0.345] |
| typed_sub | 0.500 [0.357, 0.635] | 0.904 [0.876, 0.930] | 0.680 [0.533, 0.827] | 1.000 [1.000, 1.000] | 0.779 [0.664, 0.894] |
| typed_sub_prereq | 0.731 [0.607, 0.848] | 0.914 [0.886, 0.941] | 0.893 [0.800, 0.973] | 1.000 [1.000, 1.000] | 0.298 [0.275, 0.321] |
| no_bridging | 0.000 [0.000, 0.000] | 0.819 [0.811, 0.825] | 0.373 [0.280, 0.467] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] |
| cosine-only | 0.454 [0.000, 0.000] | 0.908 [0.000, 0.000] | 0.667 [0.000, 0.000] | 1.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| cosine-only | 0.454 [0.000, 0.000] | 0.908 [0.000, 0.000] | 0.667 [0.000, 0.000] | 1.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |

## Arm: embedding — Current embedding substrate (frozen)

Selected config (train sweep): `bridge_cutoff=0.7, bridge_credit_scale=2.0, unreachable_penalty=0.0, max_hops=2`

### Train split (15 JDs)

| Ranker | bridge>weak | nDCG@10 | P@5 | MRR | bridgeable-gap precision |
|---|---|---|---|---|---|
| embedding | 0.739 [0.615, 0.848] | 0.923 [0.904, 0.942] | 0.907 [0.813, 0.973] | 1.000 [1.000, 1.000] | 0.478 [0.416, 0.547] |
| cosine-only | 0.459 [0.000, 0.000] | 0.913 [0.000, 0.000] | 0.667 [0.000, 0.000] | 1.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| tfidf | 0.002 [0.000, 0.000] | 0.824 [0.000, 0.000] | 0.427 [0.000, 0.000] | 1.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| no_bridging | 0.000 [0.615, 0.848] | 0.824 [0.904, 0.942] | 0.427 [0.813, 0.973] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] |

**Bridgeable-gap precision:** of 462 gaps labeled bridgeable, 47.8% bridge to a genuine substitute (curated ground truth, independent of the graph) [0.416, 0.547].

**Lift vs cosine-only:** bridge>weak +0.280, nDCG@10 +0.010, P@5 +0.240, MRR +0.000.

### Heldout split (15 JDs)

| Ranker | bridge>weak | nDCG@10 | P@5 | MRR | bridgeable-gap precision |
|---|---|---|---|---|---|
| embedding | 0.859 [0.761, 0.944] | 0.922 [0.875, 0.959] | 0.960 [0.920, 1.000] | 0.933 [0.833, 1.000] | 0.488 [0.403, 0.578] |
| cosine-only | 0.454 [0.000, 0.000] | 0.908 [0.000, 0.000] | 0.667 [0.000, 0.000] | 1.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| tfidf | 0.002 [0.000, 0.000] | 0.819 [0.000, 0.000] | 0.373 [0.000, 0.000] | 1.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| no_bridging | 0.000 [0.761, 0.944] | 0.819 [0.875, 0.959] | 0.373 [0.920, 1.000] | 1.000 [0.833, 1.000] | 0.000 [0.000, 0.000] |

**Bridgeable-gap precision:** of 455 gaps labeled bridgeable, 48.8% bridge to a genuine substitute (curated ground truth, independent of the graph) [0.403, 0.578].

**Lift vs cosine-only:** bridge>weak +0.406, nDCG@10 +0.014, P@5 +0.293, MRR -0.067.


## Arm: categorical — O*NET Element Name cliques only (control)

Selected config (train sweep): `bridge_cutoff=0.5, bridge_credit_scale=2.0, unreachable_penalty=0.0, max_hops=1`

### Train split (15 JDs)

| Ranker | bridge>weak | nDCG@10 | P@5 | MRR | bridgeable-gap precision |
|---|---|---|---|---|---|
| categorical | 0.422 [0.313, 0.519] | 0.867 [0.837, 0.894] | 0.720 [0.587, 0.840] | 1.000 [1.000, 1.000] | 0.347 [0.300, 0.412] |
| cosine-only | 0.459 [0.000, 0.000] | 0.913 [0.000, 0.000] | 0.667 [0.000, 0.000] | 1.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| tfidf | 0.002 [0.000, 0.000] | 0.824 [0.000, 0.000] | 0.427 [0.000, 0.000] | 1.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| no_bridging | 0.000 [0.313, 0.519] | 0.824 [0.837, 0.894] | 0.427 [0.587, 0.840] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] |

**Bridgeable-gap precision:** of 409 gaps labeled bridgeable, 34.7% bridge to a genuine substitute (curated ground truth, independent of the graph) [0.300, 0.412].

**Lift vs cosine-only:** bridge>weak -0.037, nDCG@10 -0.046, P@5 +0.053, MRR +0.000.

### Heldout split (15 JDs)

| Ranker | bridge>weak | nDCG@10 | P@5 | MRR | bridgeable-gap precision |
|---|---|---|---|---|---|
| categorical | 0.547 [0.395, 0.695] | 0.880 [0.834, 0.923] | 0.787 [0.653, 0.907] | 1.000 [1.000, 1.000] | 0.292 [0.240, 0.345] |
| cosine-only | 0.454 [0.000, 0.000] | 0.908 [0.000, 0.000] | 0.667 [0.000, 0.000] | 1.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| tfidf | 0.002 [0.000, 0.000] | 0.819 [0.000, 0.000] | 0.373 [0.000, 0.000] | 1.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| no_bridging | 0.000 [0.395, 0.695] | 0.819 [0.834, 0.923] | 0.373 [0.653, 0.907] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] |

**Bridgeable-gap precision:** of 479 gaps labeled bridgeable, 29.2% bridge to a genuine substitute (curated ground truth, independent of the graph) [0.240, 0.345].

**Lift vs cosine-only:** bridge>weak +0.094, nDCG@10 -0.028, P@5 +0.120, MRR +0.000.


## Arm: typed_sub — Typed edges: substitutes only

Selected config (train sweep): `bridge_cutoff=0.4, bridge_credit_scale=2.0, unreachable_penalty=0.0, max_hops=2`

### Train split (15 JDs)

| Ranker | bridge>weak | nDCG@10 | P@5 | MRR | bridgeable-gap precision |
|---|---|---|---|---|---|
| typed_sub | 0.517 [0.361, 0.678] | 0.913 [0.886, 0.942] | 0.773 [0.653, 0.907] | 1.000 [1.000, 1.000] | 0.670 [0.545, 0.819] |
| cosine-only | 0.459 [0.000, 0.000] | 0.913 [0.000, 0.000] | 0.667 [0.000, 0.000] | 1.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| tfidf | 0.002 [0.000, 0.000] | 0.824 [0.000, 0.000] | 0.427 [0.000, 0.000] | 1.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| no_bridging | 0.000 [0.361, 0.678] | 0.824 [0.886, 0.942] | 0.427 [0.653, 0.907] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] |

**Bridgeable-gap precision:** of 115 gaps labeled bridgeable, 67.0% bridge to a genuine substitute (curated ground truth, independent of the graph) [0.545, 0.819].

**Lift vs cosine-only:** bridge>weak +0.057, nDCG@10 -0.000, P@5 +0.107, MRR +0.000.

### Heldout split (15 JDs)

| Ranker | bridge>weak | nDCG@10 | P@5 | MRR | bridgeable-gap precision |
|---|---|---|---|---|---|
| typed_sub | 0.500 [0.357, 0.635] | 0.904 [0.876, 0.930] | 0.680 [0.533, 0.827] | 1.000 [1.000, 1.000] | 0.779 [0.664, 0.894] |
| cosine-only | 0.454 [0.000, 0.000] | 0.908 [0.000, 0.000] | 0.667 [0.000, 0.000] | 1.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| tfidf | 0.002 [0.000, 0.000] | 0.819 [0.000, 0.000] | 0.373 [0.000, 0.000] | 1.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| no_bridging | 0.000 [0.357, 0.635] | 0.819 [0.876, 0.930] | 0.373 [0.533, 0.827] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] |

**Bridgeable-gap precision:** of 140 gaps labeled bridgeable, 77.9% bridge to a genuine substitute (curated ground truth, independent of the graph) [0.664, 0.894].

**Lift vs cosine-only:** bridge>weak +0.046, nDCG@10 -0.004, P@5 +0.013, MRR +0.000.


## Arm: typed_sub_prereq — Typed edges: substitutes + prerequisites

Selected config (train sweep): `bridge_cutoff=0.7, bridge_credit_scale=2.0, unreachable_penalty=0.0, max_hops=None`

### Train split (15 JDs)

| Ranker | bridge>weak | nDCG@10 | P@5 | MRR | bridgeable-gap precision |
|---|---|---|---|---|---|
| typed_sub_prereq | 0.540 [0.376, 0.707] | 0.896 [0.866, 0.924] | 0.787 [0.680, 0.893] | 1.000 [1.000, 1.000] | 0.258 [0.216, 0.290] |
| cosine-only | 0.459 [0.000, 0.000] | 0.913 [0.000, 0.000] | 0.667 [0.000, 0.000] | 1.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| tfidf | 0.002 [0.000, 0.000] | 0.824 [0.000, 0.000] | 0.427 [0.000, 0.000] | 1.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| no_bridging | 0.000 [0.376, 0.707] | 0.824 [0.866, 0.924] | 0.427 [0.680, 0.893] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] |

**Bridgeable-gap precision:** of 391 gaps labeled bridgeable, 25.8% bridge to a genuine substitute (curated ground truth, independent of the graph) [0.216, 0.290].

**Lift vs cosine-only:** bridge>weak +0.081, nDCG@10 -0.017, P@5 +0.120, MRR +0.000.

### Heldout split (15 JDs)

| Ranker | bridge>weak | nDCG@10 | P@5 | MRR | bridgeable-gap precision |
|---|---|---|---|---|---|
| typed_sub_prereq | 0.731 [0.607, 0.848] | 0.914 [0.886, 0.941] | 0.893 [0.800, 0.973] | 1.000 [1.000, 1.000] | 0.298 [0.275, 0.321] |
| cosine-only | 0.454 [0.000, 0.000] | 0.908 [0.000, 0.000] | 0.667 [0.000, 0.000] | 1.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| tfidf | 0.002 [0.000, 0.000] | 0.819 [0.000, 0.000] | 0.373 [0.000, 0.000] | 1.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| no_bridging | 0.000 [0.607, 0.848] | 0.819 [0.886, 0.941] | 0.373 [0.800, 0.973] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] |

**Bridgeable-gap precision:** of 527 gaps labeled bridgeable, 29.8% bridge to a genuine substitute (curated ground truth, independent of the graph) [0.275, 0.321].

**Lift vs cosine-only:** bridge>weak +0.278, nDCG@10 +0.006, P@5 +0.227, MRR +0.000.


## Arm: no_bridging — Embedding graph, bridging disabled (floor)

Selected config (train sweep): `bridge_cutoff=0.4, bridge_credit_scale=1.0, unreachable_penalty=0.0, max_hops=1, enable_bridging=False`

### Train split (15 JDs)

| Ranker | bridge>weak | nDCG@10 | P@5 | MRR | bridgeable-gap precision |
|---|---|---|---|---|---|
| no_bridging | 0.000 [0.000, 0.000] | 0.824 [0.819, 0.829] | 0.427 [0.347, 0.507] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] |
| cosine-only | 0.459 [0.000, 0.000] | 0.913 [0.000, 0.000] | 0.667 [0.000, 0.000] | 1.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| tfidf | 0.002 [0.000, 0.000] | 0.824 [0.000, 0.000] | 0.427 [0.000, 0.000] | 1.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |

**Lift vs cosine-only:** bridge>weak -0.459, nDCG@10 -0.089, P@5 -0.240, MRR +0.000.

### Heldout split (15 JDs)

| Ranker | bridge>weak | nDCG@10 | P@5 | MRR | bridgeable-gap precision |
|---|---|---|---|---|---|
| no_bridging | 0.000 [0.000, 0.000] | 0.819 [0.811, 0.825] | 0.373 [0.280, 0.467] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] |
| cosine-only | 0.454 [0.000, 0.000] | 0.908 [0.000, 0.000] | 0.667 [0.000, 0.000] | 1.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| tfidf | 0.002 [0.000, 0.000] | 0.819 [0.000, 0.000] | 0.373 [0.000, 0.000] | 1.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |

**Lift vs cosine-only:** bridge>weak -0.454, nDCG@10 -0.089, P@5 -0.293, MRR +0.000.


## Arm: cosine-only — Cosine baseline (strong)

### Train split (15 JDs)

| Ranker | bridge>weak | nDCG@10 | P@5 | MRR |
|---|---|---|---|---|
| cosine-only | 0.459 | 0.913 | 0.667 | 1.000 |
| tfidf | 0.002 | 0.824 | 0.427 | 1.000 |

**Lift vs cosine-only:** bridge>weak +0.000, nDCG@10 +0.000, P@5 +0.000, MRR +0.000.

### Heldout split (15 JDs)

| Ranker | bridge>weak | nDCG@10 | P@5 | MRR |
|---|---|---|---|---|
| cosine-only | 0.454 | 0.908 | 0.667 | 1.000 |
| tfidf | 0.002 | 0.819 | 0.373 | 1.000 |

**Lift vs cosine-only:** bridge>weak +0.000, nDCG@10 +0.000, P@5 +0.000, MRR +0.000.


## Arm: tfidf — TF-IDF baseline (weak)

### Train split (15 JDs)

| Ranker | bridge>weak | nDCG@10 | P@5 | MRR |
|---|---|---|---|---|
| tfidf | 0.002 | 0.824 | 0.427 | 1.000 |
| cosine-only | 0.459 | 0.913 | 0.667 | 1.000 |

**Lift vs cosine-only:** bridge>weak -0.457, nDCG@10 -0.089, P@5 -0.240, MRR +0.000.

### Heldout split (15 JDs)

| Ranker | bridge>weak | nDCG@10 | P@5 | MRR |
|---|---|---|---|---|
| tfidf | 0.002 | 0.819 | 0.373 | 1.000 |
| cosine-only | 0.454 | 0.908 | 0.667 | 1.000 |

**Lift vs cosine-only:** bridge>weak -0.452, nDCG@10 -0.089, P@5 -0.293, MRR +0.000.


## Interpretation (with §4.4 Corrections)

### 1. The 0.000 floor is by construction (Mechanism Probe)

`no_bridging` scores 0.000 on the boundary decision — indistinguishable from TF-IDF (0.002). This is **not a performance comparison**; it is a **mechanism probe**. The dataset was built so the weak tier has ~4× more exact JD-skill overlap than the bridgeable tier, which forces any bag-of-skills ranker to invert the correct order. The 0.000 → 0.859 gap confirms the graph traversal mechanism activates on exactly the decision it was designed for. The 'beats a real baseline' claim is carried by `cosine-only` at 0.454 (not forced by construction), which Synapse improves on by +0.406.

### 2. FR3 (Ranking) and FR4 (Gap Classification) are distinct claims

**Ranking works:** `bridge>weak` = 0.859 [CI: ...] on heldout. The graph ranks bridgeable candidates above weak ones reliably.

**Gap classification does not:** Bridgeable-gap precision = 48.8% (455 bridges). Roughly half the gaps labeled bridgeable connect to something outside the curated substitution group. They coexist because ranking only needs bridge *counts* to correlate with relevance; it survives individual bridges being wrong. FR4 is the user-facing surface and must not shelter under FR3's number.

### 3. `unreachable_penalty=0.0` is under-reported

The sweep zeroed one of the three scoring terms specified in A4 (direct, bridge, penalty). The reported config has `unreachable_penalty=0.0`, meaning true gaps incur no penalty — only bridgeable gaps get a small penalty (`bridgeable_penalty`). This should be stated as a limitation with the same candour applied to the JD-demand-weighting finding. Also check whether `bridge_credit_scale=2.0` sits at the edge of the swept range — if so, the optimum may lie outside the grid and the reported config is a boundary artifact.

### 4. Substrate comparison

**Categorical control** (AUC=1.0 by construction): bridge>weak = 0.547. If this matches the embedding arm within CI, the lift belongs to the O*NET taxonomy, not the embedder — the embedder must be removed from edge construction.

**Typed (sub only)**: bridge>weak = 0.500. **Typed (sub + prereq)**: bridge>weak = 0.731. If typed_sub_prereq does not materially exceed typed_sub, prerequisite traversal is not earning its keep and should be dropped.

## Limitations (stated deliberately)

- **The dataset is synthetic.** Relevance is known by construction from curated substitution groups, not from recruiter judgement on real resumes. It is designed to be non-circular (ground truth never reads the graph) and adversarial to keyword matching, but it is not a field study.
- **Bridgeable-gap precision is ~49%.** Roughly half the gaps labeled bridgeable connect to something outside the curated substitution group. The metric is strict — it credits only exact group members, not broader learnable adjacency — but this is still the weakest number in the report and the clearest target for better edge construction.
- **MRR dips slightly** (0.933 vs 1.000): rewarding bridges strongly occasionally lifts a bridgeable candidate above a strong one. Acceptable here, since every relevant candidate still surfaces early, but it is a real cost of the tuned configuration.
- **JD demand weighting did not help** on this benchmark (see ABLATION.md); the uniform-weight arm scores marginally higher. Reported rather than quietly dropped.
- **`unreachable_penalty=0.0` in the selected config** means true gaps are not penalized — the scoring function relies entirely on positive credit for matches and bridges. This is a boundary artifact of the grid search; the optimum may lie outside the swept range.
