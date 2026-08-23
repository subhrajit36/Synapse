# Phase B4 - Parameter sweep & ablation

Swept 108 configs on the **train** split (15 JDs); the winner is reported on the **heldout** split (15 JDs), which was not consulted during selection.

Selected config: `bridge_cutoff=0.7, bridge_credit_scale=2.0, unreachable_penalty=0.0, max_hops=2`

## Tuned vs default

| split | config | bridge>weak | nDCG@10 |
|---|---|---|---|
| train | default | 0.330 | 0.888 |
| train | tuned | 0.739 | 0.923 |
| heldout | default | 0.417 | 0.905 |
| **heldout** | **tuned** | **0.859** | **0.922** |

## Ablation (heldout)

| arm | bridge>weak | nDCG@10 |
|---|---|---|
| selected config | 0.859 | 0.922 |
| no bridging (direct match only) | 0.000 | 0.819 |
| uniform weights (no JD demand) | 0.883 | 0.919 |
| max_hops = 1 | 0.846 | 0.926 |
| max_hops = 2 | 0.859 | 0.922 |
| default (pre-tuning) | 0.417 | 0.905 |
