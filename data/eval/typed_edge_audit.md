# Typed Edge Classification Audit

Total edges classified: 1931

## Type Distribution
| type | count | % |
|---|---|---|
| complement | 852 | 44.1% |
| prerequisite | 151 | 7.8% |
| unrelated | 727 | 37.6% |
| substitute | 201 | 10.4% |

## Directed Consistency
Mutual prerequisite violations: 0

## Probe Checks
| pair | expected | got | confidence | pass |
|---|---|---|---|---|
| Docker↔Kubernetes | complement | complement | 0.95 | ✓ |
| PyTorch↔TensorFlow | substitute | substitute | 0.95 | ✓ |
| MySQL↔PostgreSQL | substitute | substitute | 0.95 | ✓ |
| Git↔GitHub | complement | complement | 1.00 | ✓ |