# Edge construction audit

`213` skills, `55` O*NET categories, substrate `typed`.

Grouping for dominance/spread is the deterministic O*NET Element Name.

## 1. Category dominance

AUC = P(a same-category pair scores above a different-category pair). Near 1.0 means the substrate is a category detector.

| variant | AUC | point-biserial r | same-cat mean | diff-cat mean | gap |
|---|---|---|---|---|---|
| typed | 0.610 | 0.337 | 0.121 | 0.003 | 0.119 |

## 3. Probe pairs

`margin` = sim(a,b) − best rival similarity. Small margin = the 'correct' neighbour is indistinguishable from an unrelated one.

| pair | variant | sim | rank | margin | best rival | hand-authored |
|---|---|---|---|---|---|---|
| Docker ↔ Kubernetes | typed | 0.000 | #3 | -0.192 | Linux (0.192) | ✗ |
| PyTorch ↔ TensorFlow | typed | 0.731 | #1 | +0.500 | Python (0.231) | ✓ |
| TensorFlow ↔ Keras | typed | 0.154 | #6 | -0.577 | PyTorch (0.731) | ✓ |
| XGBoost ↔ LightGBM | typed | 0.731 | #1 | +0.731 | AJAX (0.000) | ✓ |
| GitHub ↔ GitLab | typed | 0.731 | #1 | +0.013 | Apache Subversion SVN (0.718) | ✗ |
| MySQL ↔ PostgreSQL | typed | 0.731 | #1 | +0.000 | Microsoft SQL Server (0.731) | ✗ |
| Apache Spark ↔ Apache Hadoop | typed | 0.000 | #4 | -0.705 | Apache Hive (0.705) | ✗ |

Skipped — node absent from taxonomy: `[('React', 'Angular'), ('Jenkins', 'Apache Maven'), ('Amazon Web Services', 'Microsoft Azure')]`. These are coverage gaps, not edge-construction problems.

### Mean probe margin (all pairs)

| variant | mean margin | rank-1 hits |
|---|---|---|
| typed | -0.033 | 3/7 |

### Mean probe margin (excluding hand-authored category overrides)

| variant | mean margin | rank-1 hits |
|---|---|---|
| typed | -0.221 | 1/4 |

## 4. Anchor neighbourhoods

### Docker

| rank | typed |
|---|---|
| 1 | Linux 0.192 |
| 2 | UNIX Shell 0.154 |
| 3 | Adobe After Effects 0.000 |
| 4 | AJAX 0.000 |
| 5 | Adobe Illustrator 0.000 |
| 6 | Adobe InDesign 0.000 |
| 7 | Adobe Photoshop 0.000 |
| 8 | Adobe Acrobat 0.000 |

### PyTorch

| rank | typed |
|---|---|
| 1 | TensorFlow 0.731 |
| 2 | Python 0.231 |
| 3 | NumPy 0.192 |
| 4 | AJAX 0.000 |
| 5 | Adobe Illustrator 0.000 |
| 6 | Adobe Acrobat 0.000 |
| 7 | Adobe After Effects 0.000 |
| 8 | Adobe Creative Cloud software 0.000 |

