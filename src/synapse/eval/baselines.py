"""Phase B2: graph-free baselines that isolate what the skill graph adds.

Two naive rankers score a candidate against a JD WITHOUT any graph reasoning:

  TfidfBaseline  - each skill is an atomic term; TF-IDF vectors + cosine. This is
                   the keyword-ATS model: a swapped skill (Docker for Kubernetes)
                   shares no term, so a bridgeable candidate scores near a weak one.
  CosineBaseline - embed the JD's and candidate's skill bags as text with the SAME
                   model the graph is built from, then cosine. This isolates graph
                   STRUCTURE: identical embeddings, minus shortest-path bridging.
                   If Synapse beats this, the graph (not just embeddings) is why.

Both expose .rank(jd_skills, candidates) -> [(cand_id, score)] sorted descending
-- the same shape the harness gets from Matcher.rank -- so B3 scores all three
rankers through one code path.
"""

from __future__ import annotations

from typing import Iterable, Mapping

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def _skills(x: Mapping[str, float] | Iterable[str]) -> list[str]:
    """Accept {skill: weight} or a bare list; baselines ignore weights by design
    (they represent naive approaches that have no notion of skill demand)."""
    return list(x.keys()) if isinstance(x, Mapping) else list(x)


class TfidfBaseline:
    name = "tfidf"

    def rank(self, jd_skills, candidates: Mapping[str, object]):
        jd = _skills(jd_skills)
        ids = list(candidates)
        docs = [jd] + [_skills(candidates[c]) for c in ids]
        # analyzer=identity -> each canonical skill is one term (not split into
        # words), so this measures skill overlap, not incidental word overlap.
        vec = TfidfVectorizer(analyzer=lambda s: s)
        matrix = vec.fit_transform(docs)
        sims = cosine_similarity(matrix[0:1], matrix[1:]).ravel()
        return sorted(zip(ids, sims.tolist()), key=lambda t: -t[1])


class CosineBaseline:
    name = "cosine"

    def __init__(self, model=None, model_name: str = "all-MiniLM-L6-v2"):
        # Same embedder the graph is built from, so the only difference vs Synapse
        # is the graph structure -- a clean isolation of the graph's contribution.
        from sentence_transformers import SentenceTransformer
        self.model = model or SentenceTransformer(model_name)

    @staticmethod
    def _text(skills: list[str]) -> str:
        return ", ".join(skills)

    def rank(self, jd_skills, candidates: Mapping[str, object]):
        jd = _skills(jd_skills)
        ids = list(candidates)
        texts = [self._text(jd)] + [self._text(_skills(candidates[c])) for c in ids]
        emb = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        sims = emb[1:] @ emb[0]            # normalized -> dot product is cosine
        return sorted(zip(ids, sims.tolist()), key=lambda t: -t[1])


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "src")
    from synapse.eval.dataset import load

    ds = load("v1")
    jd = ds["jds"][0]
    cands = {c["cand_id"]: c["skills"] for c in jd["candidates"]}
    grade = {c["cand_id"]: c["grade"] for c in jd["candidates"]}

    print(f"JD [{jd['jd_id']}]: {list(jd['jd_skills'])}\n")
    for baseline in (TfidfBaseline(), CosineBaseline()):
        ranked = baseline.rank(jd["jd_skills"], cands)
        print(f"--- {baseline.name} top 6 (grade in brackets) ---")
        for cid, score in ranked[:6]:
            print(f"  {score:5.3f}  [g{grade[cid]}]  {cid}")
        print()
