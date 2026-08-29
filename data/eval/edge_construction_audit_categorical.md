# Edge construction audit

`213` skills, `55` O*NET categories, substrate `categorical`.

Grouping for dominance/spread is the deterministic O*NET Element Name.

## 1. Category dominance

AUC = P(a same-category pair scores above a different-category pair). Near 1.0 means the substrate is a category detector.

| variant | AUC | point-biserial r | same-cat mean | diff-cat mean | gap |
|---|---|---|---|---|---|
| categorical | 1.000 | 0.998 | 0.500 | 0.000 | 0.500 |

## 3. Probe pairs

`margin` = sim(a,b) − best rival similarity. Small margin = the 'correct' neighbour is indistinguishable from an unrelated one.

| pair | variant | sim | rank | margin | best rival | hand-authored |
|---|---|---|---|---|---|---|
| Docker ↔ Kubernetes | categorical | 0.500 | #1 | +0.000 | Atlassian Bitbucket (0.500) | ✗ |
| PyTorch ↔ TensorFlow | categorical | 0.500 | #1 | +0.000 | Amazon Elastic Compute Cloud EC2 (0.500) | ✓ |
| TensorFlow ↔ Keras | categorical | 0.500 | #1 | +0.000 | ESRI ArcGIS ArcPy (0.500) | ✓ |
| XGBoost ↔ LightGBM | categorical | 0.500 | #1 | +0.000 | CatBoost (0.500) | ✓ |
| GitHub ↔ GitLab | categorical | 0.500 | #1 | +0.000 | Atlassian Bitbucket (0.500) | ✗ |
| MySQL ↔ PostgreSQL | categorical | 0.000 | #11 | -0.500 | Amazon DynamoDB (0.500) | ✗ |
| Apache Spark ↔ Apache Hadoop | categorical | 0.000 | #5 | -0.500 | Alteryx software (0.500) | ✗ |

Skipped — node absent from taxonomy: `[('React', 'Angular'), ('Jenkins', 'Apache Maven'), ('Amazon Web Services', 'Microsoft Azure')]`. These are coverage gaps, not edge-construction problems.

### Mean probe margin (all pairs)

| variant | mean margin | rank-1 hits |
|---|---|---|
| categorical | -0.143 | 0/7 |

### Mean probe margin (excluding hand-authored category overrides)

| variant | mean margin | rank-1 hits |
|---|---|---|
| categorical | -0.250 | 0/4 |

## 4. Anchor neighbourhoods

### Docker

| rank | categorical |
|---|---|
| 1 | Atlassian Bitbucket 0.500 |
| 2 | Kubernetes 0.500 |
| 3 | GitLab 0.500 |
| 4 | GitHub 0.500 |
| 5 | Red Hat OpenShift 0.500 |
| 6 | Spring Boot 0.500 |
| 7 | Adobe Illustrator 0.000 |
| 8 | AJAX 0.000 |

### PyTorch

| rank | categorical |
|---|---|
| 1 | Amazon Elastic Compute Cloud EC2 0.500 |
| 2 | Amazon Web Services AWS software 0.500 |
| 3 | Amazon Redshift 0.500 |
| 4 | Microsoft SQL Server 0.500 |
| 5 | Microsoft Access 0.500 |
| 6 | Keras 0.500 |
| 7 | IBM DB2 0.500 |
| 8 | GraphQL 0.500 |

