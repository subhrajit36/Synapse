"""Phase B3: ranking metrics. Thin wrappers with fixed conventions so every
ranker is scored identically.

'Relevant' = grade >= 2 (strong or bridgeable) - the candidates a recruiter would
actually want surfaced. Weak (1) and irrelevant (0) are non-relevant for P@K/MRR.
nDCG uses the full graded scale (0-3), so it rewards correct *ordering*, not just
set membership.

`pairwise_accuracy` exists because the aggregate metrics above are dominated by
the easy decisions (a strong candidate outranking an irrelevant one). It isolates
ONE decision - does the ranker put a grade-A candidate above a grade-B one - so
the bridgeable(2)-vs-weak(1) boundary, which is the entire premise of the graph,
is measured directly instead of being averaged away.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import ndcg_score

RELEVANT = 2


def precision_at_k(ranked_ids: list[str], grades: dict[str, int], k: int,
                   relevant: int = RELEVANT) -> float:
    top = ranked_ids[:k]
    return sum(grades[c] >= relevant for c in top) / len(top) if top else 0.0


def mrr(ranked_ids: list[str], grades: dict[str, int], relevant: int = RELEVANT) -> float:
    for i, c in enumerate(ranked_ids, 1):
        if grades[c] >= relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(cand_ids: list[str], scores: dict[str, float],
              grades: dict[str, int], k: int) -> float:
    y_true = np.array([[grades[c] for c in cand_ids]])
    y_score = np.array([[scores[c] for c in cand_ids]])
    return float(ndcg_score(y_true, y_score, k=k))


def pairwise_accuracy(scores: dict[str, float], grades: dict[str, int],
                      high: int, low: int) -> tuple[int, int]:
    """Over all (high-grade, low-grade) pairs, how often is the high one scored
    above the low one? Ties count as half a win - a ranker that cannot separate
    them is neither right nor wrong, and scoring ties as losses would flatter
    whichever ranker happens to break ties by name.

    Returns (wins*2, n_pairs*2) as integers so callers can pool across JDs
    without weighting a JD with few pairs the same as one with many.
    """
    hi = [c for c, g in grades.items() if g == high]
    lo = [c for c, g in grades.items() if g == low]
    wins2 = 0
    for a in hi:
        for b in lo:
            if scores[a] > scores[b]:
                wins2 += 2
            elif scores[a] == scores[b]:
                wins2 += 1
    return wins2, 2 * len(hi) * len(lo)
