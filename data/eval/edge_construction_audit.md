# Edge construction audit

`213` skills, `55` O*NET categories, model `all-MiniLM-L6-v2`.

Grouping for dominance/spread is the deterministic O*NET Element Name.

## 1. Category dominance

AUC = P(a same-category pair scores above a different-category pair). Near 1.0 means the model is a category detector.

| variant | AUC | point-biserial r | same-cat mean | diff-cat mean | gap |
|---|---|---|---|---|---|
| bare | 0.618 | 0.111 | 0.213 | 0.149 | 0.064 |
| onet | 0.948 | 0.436 | 0.506 | 0.233 | 0.272 |
| override | 0.906 | 0.404 | 0.486 | 0.229 | 0.257 |

## 2. Within-category spread (`override`)

Low std means the tool name adds nothing once the suffix is fixed.

| category | n | mean | std | min | max |
|---|---|---|---|---|---|
| Development environment software | 21 | 0.399 | 0.148 | -0.006 | 0.888 |
| Web platform development software | 18 | 0.479 | 0.121 | 0.201 | 0.791 |
| Data base user interface and query software | 17 | 0.490 | 0.181 | -0.045 | 0.859 |
| Object or component oriented development software | 12 | 0.432 | 0.157 | 0.115 | 0.768 |
| Operating system software | 12 | 0.555 | 0.141 | 0.300 | 0.885 |
| Data base management system software | 11 | 0.558 | 0.076 | 0.390 | 0.769 |
| Program testing software | 8 | 0.524 | 0.084 | 0.421 | 0.778 |
| Analytical or scientific software | 8 | 0.429 | 0.204 | -0.029 | 0.835 |
| Application server software | 7 | 0.456 | 0.069 | 0.371 | 0.612 |
| Graphics or photo imaging software | 6 | 0.667 | 0.119 | 0.491 | 0.853 |
| Cloud-based management software | 6 | 0.531 | 0.076 | 0.419 | 0.754 |
| Computer aided design CAD software | 6 | 0.688 | 0.101 | 0.578 | 0.891 |

## 3. Probe pairs

`margin` = sim(a,b) − best rival similarity. Small margin = the 'correct' neighbour is indistinguishable from an unrelated one.

| pair | variant | sim | rank | margin | best rival |
|---|---|---|---|---|---|
| Docker ↔ Kubernetes | bare | 0.315 | #8 | -0.080 | Qualys Cloud Platform (0.395) |
| Docker ↔ Kubernetes | onet | 0.542 | #1 | +0.020 | Atlassian Bitbucket (0.521) |
| Docker ↔ Kubernetes | override | 0.542 | #1 | +0.020 | Atlassian Bitbucket (0.521) |
| PyTorch ↔ TensorFlow | bare | 0.486 | #1 | +0.031 | NumPy (0.456) |
| PyTorch ↔ TensorFlow | onet | 0.399 | #19 | -0.274 | NumPy (0.674) |
| PyTorch ↔ TensorFlow | override | 0.736 | #1 | +0.064 | Keras (0.673) |
| TensorFlow ↔ Keras | bare | 0.519 | #1 | +0.032 | PyTorch (0.486) |
| TensorFlow ↔ Keras | onet | 0.547 | #2 | -0.025 | The MathWorks MATLAB (0.572) |
| TensorFlow ↔ Keras | override | 0.796 | #1 | +0.059 | PyTorch (0.736) |
| XGBoost ↔ LightGBM | bare | 0.534 | #1 | +0.092 | Adobe XD (0.442) |
| XGBoost ↔ LightGBM | onet | 0.825 | #1 | +0.074 | CatBoost (0.751) |
| XGBoost ↔ LightGBM | override | 0.825 | #1 | +0.074 | CatBoost (0.751) |
| GitHub ↔ GitLab | bare | 0.473 | #5 | -0.287 | Git (0.759) |
| GitHub ↔ GitLab | onet | 0.600 | #3 | -0.025 | Git (0.626) |
| GitHub ↔ GitLab | override | 0.600 | #3 | -0.025 | Git (0.626) |
| MySQL ↔ PostgreSQL | bare | 0.547 | #1 | +0.018 | NoSQL (0.529) |
| MySQL ↔ PostgreSQL | onet | 0.594 | #9 | -0.096 | NoSQL (0.691) |
| MySQL ↔ PostgreSQL | override | 0.594 | #9 | -0.096 | NoSQL (0.691) |
| Apache Spark ↔ Apache Hadoop | bare | 0.602 | #1 | +0.014 | PySpark (0.588) |
| Apache Spark ↔ Apache Hadoop | onet | 0.618 | #2 | -0.021 | PySpark (0.639) |
| Apache Spark ↔ Apache Hadoop | override | 0.618 | #2 | -0.021 | PySpark (0.639) |

Skipped — node absent from taxonomy: `[('React', 'Angular'), ('Jenkins', 'Apache Maven'), ('Amazon Web Services', 'Microsoft Azure')]`. These are coverage gaps, not edge-construction problems.

### Mean probe margin

| variant | mean margin | rank-1 hits |
|---|---|---|
| bare | -0.026 | 5/7 |
| onet | -0.050 | 2/7 |
| override | +0.011 | 4/7 |

## 4. Anchor neighbourhoods

### Docker

| rank | bare | onet | override |
|---|---|---|---|
| 1 | Qualys Cloud Platform 0.395 | Kubernetes 0.542 | Kubernetes 0.542 |
| 2 | WordPress 0.379 | Atlassian Bitbucket 0.521 | Atlassian Bitbucket 0.521 |
| 3 | Git 0.364 | Spring Boot 0.476 | Spring Boot 0.476 |
| 4 | Atlassian Bitbucket 0.357 | GitHub 0.450 | GitHub 0.450 |
| 5 | Grafana Labs Grafana Cloud 0.340 | GitLab 0.449 | GitLab 0.449 |
| 6 | Portswigger BurP Suite 0.335 | Microsoft Windows Server 0.438 | Microsoft Windows Server 0.438 |
| 7 | Bash 0.333 | Red Hat OpenShift 0.433 | Red Hat OpenShift 0.433 |
| 8 | Kubernetes 0.315 | Bash 0.379 | Bash 0.379 |

### PyTorch

| rank | bare | onet | override |
|---|---|---|---|
| 1 | TensorFlow 0.486 | NumPy 0.674 | TensorFlow 0.736 |
| 2 | NumPy 0.456 | PySpark 0.625 | Keras 0.673 |
| 3 | Python 0.430 | Yardi software 0.622 | LightGBM 0.419 |
| 4 | PySpark 0.428 | pandas 0.574 | XGBoost 0.399 |
| 5 | Scikit-learn 0.416 | GraphQL 0.567 | CatBoost 0.334 |
| 6 | Ruby 0.394 | IBM DB2 0.529 | BERT 0.324 |
| 7 | DistilBERT 0.394 | Microsoft Access 0.515 | Scikit-learn 0.320 |
| 8 | Keras 0.384 | Structured query language SQL 0.515 | Mistral 0.311 |

