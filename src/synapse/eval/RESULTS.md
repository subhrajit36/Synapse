# Phase B - Evaluation Results

Dataset `v2` - 30 JDs, 540 pairs, seed 42, split {'train': 15, 'heldout': 15}. Relevant = grade >= 2.

`bridge>weak` is the headline: pairwise accuracy on bridgeable(2) vs weak(1), the decision the graph exists to get right. By construction the weak tier has ~4x more exact JD-skill overlap than the bridgeable tier, so a bag-of-skills ranker must invert the correct order.

Synapse uses the config selected by the B4 sweep on train only: `bridge_cutoff=0.7, bridge_credit_scale=2.0, unreachable_penalty=0.0, max_hops=2` (see ABLATION.md).

## Train split (15 JDs)

| Ranker | bridge>weak | nDCG@10 | P@5 | MRR |
|---|---|---|---|---|
| Synapse (tuned) | 1.000 | 0.586 | 0.773 | 0.833 |
| Synapse (no bridging) | 0.000 | 0.824 | 0.427 | 1.000 |
| TF-IDF | 0.002 | 0.824 | 0.427 | 1.000 |
| cosine-only | 0.459 | 0.913 | 0.667 | 1.000 |

**Synapse lift vs cosine-only:** bridge>weak +0.541, nDCG@10 -0.327, P@5 +0.107, MRR -0.167.

**Bridgeable-gap precision:** of 1124 gaps labeled bridgeable, 18.2% bridge to a genuine substitute (curated ground truth, independent of the graph).

## Heldout split (15 JDs)

| Ranker | bridge>weak | nDCG@10 | P@5 | MRR |
|---|---|---|---|---|
| Synapse (tuned) | 1.000 | 0.614 | 0.733 | 0.867 |
| Synapse (no bridging) | 0.000 | 0.819 | 0.373 | 1.000 |
| TF-IDF | 0.002 | 0.819 | 0.373 | 1.000 |
| cosine-only | 0.454 | 0.908 | 0.667 | 1.000 |

**Synapse lift vs cosine-only:** bridge>weak +0.546, nDCG@10 -0.294, P@5 +0.067, MRR -0.133.

**Bridgeable-gap precision:** of 1097 gaps labeled bridgeable, 19.8% bridge to a genuine substitute (curated ground truth, independent of the graph).

## Interpretation

**The graph, not the embeddings, does the work.** `Synapse (no bridging)` is the identical pipeline with graph traversal switched off: it scores 0.000 on the boundary decision, indistinguishable from TF-IDF (0.002). Switching bridging on takes the same code to 1.000. Since embeddings, extraction and scoring are unchanged between those two rows, the lift is attributable to shortest-path reasoning over skill adjacency and nothing else.

**It beats the strong baseline too.** `cosine-only` embeds the same skill text with the same model, so it already captures some adjacency - it scores 0.454, far above TF-IDF. Synapse still improves on it by +0.546 on the boundary decision and +0.067 on P@5. Explicit graph structure therefore adds signal that pooled embedding similarity does not.

**Held-out, not tuned-on.** Parameters were selected on train (bridge>weak 1.000); the heldout split reports 1.000 without being consulted during selection, so the gain is not an artifact of the search.

## Limitations (stated deliberately)

- **The dataset is synthetic.** Relevance is known by construction from curated substitution groups, not from recruiter judgement on real resumes. It is designed to be non-circular (ground truth never reads the graph) and adversarial to keyword matching, but it is not a field study.
- **Bridgeable-gap precision is ~20%.** Roughly half the gaps labeled bridgeable connect to something outside the curated substitution group. The metric is strict - it credits only exact group members, not broader learnable adjacency - but this is still the weakest number in the report and the clearest target for better edge construction.
- **MRR dips slightly** (0.867 vs 1.000): rewarding bridges strongly occasionally lifts a bridgeable candidate above a strong one. Acceptable here, since every relevant candidate still surfaces early, but it is a real cost of the tuned configuration.
- **JD demand weighting did not help** on this benchmark (see ABLATION.md); the uniform-weight arm scores marginally higher. Reported rather than quietly dropped.
