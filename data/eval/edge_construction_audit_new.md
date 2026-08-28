# Edge construction audit

`213` skills, `55` O*NET categories, model `BAAI/bge-small-en-v1.5`.

Grouping for dominance/spread is the deterministic O*NET Element Name.

## 1. Category dominance

AUC = P(a same-category pair scores above a different-category pair). Near 1.0 means the model is a category detector.

| variant | AUC | point-biserial r | same-cat mean | diff-cat mean | gap |
|---|---|---|---|---|---|
| bare | 0.657 | 0.130 | 0.610 | 0.570 | 0.040 |
| onet | 0.931 | 0.394 | 0.753 | 0.628 | 0.125 |
| override | 0.895 | 0.367 | 0.744 | 0.626 | 0.118 |

## 2. Within-category spread (`override`)

Low std means the tool name adds nothing once the suffix is fixed.

| category | n | mean | std | min | max |
|---|---|---|---|---|---|
| Development environment software | 21 | 0.706 | 0.070 | 0.459 | 0.932 |
| Web platform development software | 18 | 0.733 | 0.054 | 0.620 | 0.870 |
| Data base user interface and query software | 17 | 0.773 | 0.077 | 0.475 | 0.914 |
| Object or component oriented development software | 12 | 0.754 | 0.075 | 0.554 | 0.870 |
| Operating system software | 12 | 0.705 | 0.074 | 0.589 | 0.939 |
| Data base management system software | 11 | 0.800 | 0.032 | 0.736 | 0.896 |
| Program testing software | 8 | 0.713 | 0.035 | 0.662 | 0.821 |
| Analytical or scientific software | 8 | 0.707 | 0.082 | 0.546 | 0.901 |
| Application server software | 7 | 0.732 | 0.034 | 0.663 | 0.827 |
| Graphics or photo imaging software | 6 | 0.839 | 0.033 | 0.784 | 0.906 |
| Cloud-based management software | 6 | 0.794 | 0.035 | 0.743 | 0.877 |
| Computer aided design CAD software | 6 | 0.850 | 0.036 | 0.807 | 0.922 |

## 3. Probe pairs

`margin` = sim(a,b) − best rival similarity. Small margin = the 'correct' neighbour is indistinguishable from an unrelated one.

| pair | variant | sim | rank | margin | best rival |
|---|---|---|---|---|---|
| Docker ↔ Kubernetes | bare | 0.699 | #9 | -0.044 | Portswigger BurP Suite (0.743) |
| Docker ↔ Kubernetes | onet | 0.770 | #1 | +0.007 | GitHub (0.763) |
| Docker ↔ Kubernetes | override | 0.770 | #1 | +0.007 | GitHub (0.763) |
| PyTorch ↔ TensorFlow | bare | 0.725 | #2 | -0.000 | PySpark (0.725) |
| PyTorch ↔ TensorFlow | onet | 0.727 | #24 | -0.122 | PySpark (0.848) |
| PyTorch ↔ TensorFlow | override | 0.861 | #1 | +0.044 | Keras (0.817) |
| TensorFlow ↔ Keras | bare | 0.820 | #1 | +0.094 | Scikit-learn (0.726) |
| TensorFlow ↔ Keras | onet | 0.736 | #6 | -0.040 | SAS (0.777) |
| TensorFlow ↔ Keras | override | 0.912 | #1 | +0.050 | PyTorch (0.861) |
| XGBoost ↔ LightGBM | bare | 0.748 | #1 | +0.031 | Adobe XD (0.717) |
| XGBoost ↔ LightGBM | onet | 0.876 | #1 | +0.020 | CatBoost (0.856) |
| XGBoost ↔ LightGBM | override | 0.876 | #1 | +0.020 | CatBoost (0.856) |
| GitHub ↔ GitLab | bare | 0.766 | #2 | -0.059 | Git (0.824) |
| GitHub ↔ GitLab | onet | 0.827 | #1 | +0.027 | Git (0.800) |
| GitHub ↔ GitLab | override | 0.827 | #1 | +0.027 | Git (0.800) |
| MySQL ↔ PostgreSQL | bare | 0.836 | #1 | +0.042 | NoSQL (0.794) |
| MySQL ↔ PostgreSQL | onet | 0.864 | #1 | +0.019 | Oracle PL/SQL (0.846) |
| MySQL ↔ PostgreSQL | override | 0.864 | #1 | +0.019 | Oracle PL/SQL (0.846) |
| Apache Spark ↔ Apache Hadoop | bare | 0.865 | #1 | +0.017 | Apache Hive (0.847) |
| Apache Spark ↔ Apache Hadoop | onet | 0.872 | #1 | +0.021 | Apache Hive (0.851) |
| Apache Spark ↔ Apache Hadoop | override | 0.872 | #1 | +0.021 | Apache Hive (0.851) |

Skipped — node absent from taxonomy: `[('React', 'Angular'), ('Jenkins', 'Apache Maven'), ('Amazon Web Services', 'Microsoft Azure')]`. These are coverage gaps, not edge-construction problems.

### Mean probe margin

| variant | mean margin | rank-1 hits |
|---|---|---|
| bare | +0.012 | 4/7 |
| onet | -0.010 | 5/7 |
| override | +0.027 | 7/7 |

## 4. Anchor neighbourhoods

### Docker

| rank | bare | onet | override |
|---|---|---|---|
| 1 | Portswigger BurP Suite 0.743 | Kubernetes 0.770 | Kubernetes 0.770 |
| 2 | Linux 0.742 | GitHub 0.763 | GitHub 0.763 |
| 3 | GitHub 0.733 | Node.js 0.752 | Node.js 0.752 |
| 4 | Node.js 0.718 | GitLab 0.740 | GitLab 0.740 |
| 5 | Ansible software 0.715 | Red Hat OpenShift 0.740 | Red Hat OpenShift 0.740 |
| 6 | Bash 0.711 | Drupal 0.734 | Drupal 0.734 |
| 7 | Apache Hive 0.702 | Apache Tomcat 0.734 | Apache Tomcat 0.734 |
| 8 | Oracle Cloud software 0.701 | Microsoft Windows Server 0.731 | Microsoft Windows Server 0.731 |

### PyTorch

| rank | bare | onet | override |
|---|---|---|---|
| 1 | PySpark 0.725 | PySpark 0.848 | TensorFlow 0.861 |
| 2 | TensorFlow 0.725 | NumPy 0.825 | Keras 0.817 |
| 3 | Python 0.685 | Yardi software 0.799 | LightGBM 0.776 |
| 4 | Ruby 0.676 | pandas 0.798 | CatBoost 0.771 |
| 5 | MEDITECH software 0.673 | GraphQL 0.790 | XGBoost 0.765 |
| 6 | Scikit-learn 0.662 | IBM DB2 0.784 | Snowflake 0.714 |
| 7 | XGBoost 0.653 | Oracle Database 0.779 | Optuna 0.702 |
| 8 | Metasploit 0.651 | Prometheus 0.779 | BERT 0.701 |

