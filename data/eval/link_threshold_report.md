# Entity-linking threshold calibration

Nodes: 213 | alias hits and surface hits bypass the threshold entirely.

| kind | surface | expected | linked | score | method |
|---|---|---|---|---|---|
| positive | docker | Docker | Docker | 1.000 | surface |
| positive | kubernetes | Kubernetes | Kubernetes | 1.000 | alias |
| positive | python | Python | Python | 1.000 | surface |
| positive | jenkins | Jenkins | Jenkins CI | 1.000 | surface |
| positive | terraform | Terraform | IBM Terraform | 1.000 | surface |
| negative | stakeholder management | - | Oracle Primavera Enterprise Project Portfolio Management | 0.380 | embedding |
| negative | underwater basket weaving | - | Rust programming language | 0.188 | embedding |
| negative | team player | - | Microsoft Teams | 0.426 | embedding |
| negative | excellent communication | - | Mistral | 0.200 | embedding |
| negative | willingness to learn | - | Scikit-learn | 0.239 | embedding |

All probes resolved deterministically (alias/surface). Add harder surfaces to exercise the embedding path.
