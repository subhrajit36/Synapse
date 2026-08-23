"""Phase B1: discriminating evaluation dataset with relevance known by construction.

Why synthetic: there is no public (resume, JD, graded-relevance) corpus for this
task, and hand-labelling enough pairs for stable nDCG is infeasible here. So we
GENERATE candidates at controlled relevance tiers, where the ground-truth grade
is fixed at construction time.

--- What makes this eval credible ---

1. NO CIRCULARITY. Ground truth must not come from the graph under test. If
   "bridgeable" candidates were built by reading the graph's own neighbours and
   then graded relevant, we'd only be checking that the graph reproduces itself.
   Instead, substitutes come from CURATED, human-judged groups (Docker/Kubernetes/
   OpenShift, Linux/UNIX/RHEL, Tableau/Power BI, ...). Those encode real-world
   interchangeability, independent of any graph edge.

2. IT DISCRIMINATES (v2). The first version was saturated: every tier kept most
   of its exact JD skills, so a keyword baseline scored ~0.99 and the graph was
   never needed - a ceiling effect that could not distinguish any ranker.
   v2 makes the bridgeable-vs-weak boundary adversarial to keyword matching:

       bridgeable (grade 2) : substitutes EVERY substitutable JD skill
                              -> ~0-1 exact JD skills
       weak       (grade 1) : keeps 2-3 exact JD anchor skills + off-domain filler
                              -> ~2-3 exact JD skills

   So a bag-of-skills ranker sees *more* overlap on the WRONG candidate and must
   invert the correct order. This is not rigged against the baselines: the cosine
   baseline embeds the same skill text and is free to recognise the substitution.
   It only removes the free win that exact-token overlap was getting.

3. IT IS SPLIT. JDs are partitioned train / heldout (B1.4). Parameters are swept
   on train only; the reported number is heldout. Tuning on the reported set
   would make any "improvement" unfalsifiable.

Reproducible (NFR7): fixed seed, versioned output under data/eval/<version>/.
Skill strings are canonical graph node names, so scoring is tested without an
entity-linking confound.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

# --- External ground truth: human-judged interchangeable skills (NOT graph edges) ---
# Each list is a set of tools a hiring manager would accept in place of one another.
SUBSTITUTION_GROUPS = [
    ["Docker", "Kubernetes", "Red Hat OpenShift"],                      # containers
    ["Amazon Web Services AWS software", "Microsoft Azure software",
     "Oracle Cloud software"],                                          # cloud
    ["GitHub", "GitLab", "Atlassian Bitbucket"],                        # repo hosting
    ["Linux", "UNIX", "Red Hat Enterprise Linux"],                      # unix-like OS
    ["Bash", "UNIX Shell", "Shell script"],                             # shell
    ["Ansible software", "Puppet", "Chef"],                             # config mgmt
    ["PyTorch", "TensorFlow", "Keras"],                                 # deep learning
    ["XGBoost", "LightGBM", "CatBoost"],                                # boosting
    ["MySQL", "PostgreSQL", "Oracle Database"],                         # relational db
    ["MongoDB", "Apache Cassandra", "Redis"],                           # nosql
    ["React", "Vue.js"],                                                # frontend fw
    ["Apache Spark", "Apache Hadoop"],                                  # big data
    ["Tableau", "Microsoft Power BI"],                                  # BI
    ["Structured query language SQL", "Oracle PL/SQL", "Transact-SQL"],  # sql dialect
]

# Per domain: `subs` are group members (a JD takes at most one per group, and a
# bridgeable candidate swaps them); `anchors` belong to no group, so a weak
# candidate can hold them exactly without secretly becoming bridgeable.
DOMAINS = {
    "devops": {
        "subs": ["Docker", "Amazon Web Services AWS software", "GitHub", "Linux",
                 "Bash", "Ansible software"],
        "anchors": ["Jenkins CI", "Prometheus", "Git", "Splunk Enterprise"],
    },
    "ml": {
        "subs": ["PyTorch", "XGBoost", "Apache Spark"],
        "anchors": ["Python", "pandas", "NumPy"],
    },
    "frontend": {
        "subs": ["React", "GitHub"],
        "anchors": ["JavaScript", "TypeScript", "Cascading style sheets CSS",
                    "Hypertext markup language HTML", "Figma"],
    },
    "backend": {
        "subs": ["PostgreSQL", "MongoDB", "Docker"],
        "anchors": ["Python", "Spring Boot", "Node.js", "RESTful API", "Django"],
    },
    "data": {
        "subs": ["Apache Spark", "Tableau", "Structured query language SQL", "MySQL"],
        "anchors": ["Python", "pandas", "Apache Airflow"],
    },
}

GRADE = {"strong": 3, "bridgeable": 2, "weak": 1, "irrelevant": 0}
TIER_COUNTS = {"strong": 4, "bridgeable": 6, "weak": 6, "irrelevant": 4}  # 20 / JD
JDS_PER_DOMAIN = 6
N_JD_SUBS = 3        # substitutable skills per JD (each from a different group)
N_JD_ANCHORS = 3     # non-substitutable skills per JD
HELDOUT_FRACTION = 0.5

# skill -> substitutes / group id
_SUBS: dict[str, list[str]] = {}
_GID: dict[str, int] = {}
for _i, _grp in enumerate(SUBSTITUTION_GROUPS):
    for _s in _grp:
        _SUBS[_s] = [x for x in _grp if x != _s]
        _GID[_s] = _i


def _all_skills() -> set[str]:
    return {s for d in DOMAINS.values() for s in d["subs"] + d["anchors"]}


def _forbidden(jd_skills) -> set[str]:
    """JD skills plus their substitutes. Weak/irrelevant candidates must avoid
    these, or they would secretly be bridgeable and mislabelled."""
    bad = set(jd_skills)
    for s in jd_skills:
        bad.update(_SUBS.get(s, []))
    return bad


def _off_domain(domain: str, forbidden: set[str]) -> list[str]:
    pool = {s for d, cfg in DOMAINS.items() if d != domain
            for s in cfg["subs"] + cfg["anchors"]}
    return sorted(pool - forbidden)


def _make_jd(rng, domain) -> dict[str, float]:
    """One substitutable skill per group (so a bridge always exists) + anchors."""
    cfg = DOMAINS[domain]
    by_group: dict[int, list[str]] = {}
    for s in cfg["subs"]:
        by_group.setdefault(_GID[s], []).append(s)
    groups = rng.sample(list(by_group), min(N_JD_SUBS, len(by_group)))
    subs = [rng.choice(by_group[g]) for g in groups]
    anchors = rng.sample(cfg["anchors"], min(N_JD_ANCHORS, len(cfg["anchors"])))
    skills = subs + anchors
    core = set(rng.sample(skills, min(2, len(skills))))     # a couple of must-haves
    return {s: (1.5 if s in core else 1.0) for s in sorted(skills)}


def _jd_parts(jd) -> tuple[list[str], list[str]]:
    subs = [s for s in jd if s in _SUBS]
    anchors = [s for s in jd if s not in _SUBS]
    return subs, anchors


def _strong(rng, jd, domain) -> list[str]:
    """Holds the JD outright, plus maybe an extra."""
    cand = set(jd)
    extras = [s for s in DOMAINS[domain]["anchors"] if s not in cand]
    cand |= set(rng.sample(extras, min(len(extras), rng.randint(0, 1))))
    return sorted(cand)


def _bridgeable(rng, jd, domain) -> list[str] | None:
    """Adjacent-but-not-identical: substitutes EVERY substitutable JD skill and
    keeps at most one anchor. Low exact overlap by construction."""
    subs, anchors = _jd_parts(jd)
    if not subs:
        return None
    cand = {rng.choice(_SUBS[s]) for s in subs}                 # all swapped
    if anchors and rng.random() < 0.5:
        cand.add(rng.choice(anchors))                           # at most one anchor
    extras = [s for s in DOMAINS[domain]["anchors"] if s not in jd]
    cand |= set(rng.sample(extras, min(len(extras), rng.randint(0, 1))))
    return sorted(cand)


def _weak(rng, jd, domain) -> list[str]:
    """Adversarial to keyword matching: keeps MORE exact JD skills than the
    bridgeable tier (2-3 anchors), but is not actually a fit - the rest of the
    profile is off-domain and no JD substitutable skill is covered at all."""
    subs, anchors = _jd_parts(jd)
    forb = _forbidden(jd)
    kept = rng.sample(anchors, min(len(anchors), rng.randint(2, 3))) if anchors else []
    fillers = _off_domain(domain, forb)
    fill = rng.sample(fillers, min(len(fillers), rng.randint(3, 4)))
    return sorted(set(kept) | set(fill))


def _irrelevant(rng, jd, domain) -> list[str]:
    pool = _off_domain(domain, _forbidden(jd))
    return sorted(rng.sample(pool, min(len(pool), rng.randint(4, 6))))


_BUILDERS = {"strong": _strong, "bridgeable": _bridgeable,
             "weak": _weak, "irrelevant": _irrelevant}


def generate(version: str = "v2", seed: int = 42) -> dict:
    rng = random.Random(seed)
    jds = []
    for domain in DOMAINS:
        for n in range(JDS_PER_DOMAIN):
            jd_id = f"{domain}_{n}"
            jd = _make_jd(rng, domain)
            candidates = []
            for tier, count in TIER_COUNTS.items():
                seen: set[tuple[str, ...]] = set()
                made = attempts = 0
                while made < count and attempts < count * 8:
                    attempts += 1
                    skills = _BUILDERS[tier](rng, jd, domain)
                    if not skills:
                        break
                    key = tuple(skills)
                    if key in seen:            # avoid duplicate profiles
                        continue
                    seen.add(key)
                    candidates.append({
                        "cand_id": f"{jd_id}__{tier}_{made}",
                        "tier": tier,
                        "grade": GRADE[tier],
                        "skills": skills,
                        "exact_overlap": len(set(skills) & set(jd)),
                    })
                    made += 1
            jds.append({"jd_id": jd_id, "domain": domain, "jd_skills": jd,
                        "candidates": candidates})

    # Train / heldout split (B1.4): tune on train, report heldout.
    order = list(range(len(jds)))
    rng.shuffle(order)
    n_heldout = int(len(jds) * HELDOUT_FRACTION)
    heldout = set(order[:n_heldout])
    for i, jd in enumerate(jds):
        jd["split"] = "heldout" if i in heldout else "train"

    dataset = {
        "version": version,
        "seed": seed,
        "n_jds": len(jds),
        "n_pairs": sum(len(j["candidates"]) for j in jds),
        "grade_scale": GRADE,
        "splits": {s: sum(1 for j in jds if j["split"] == s) for s in ("train", "heldout")},
        "jds": jds,
    }

    out_dir = Path("data/eval") / version
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "dataset.json").write_text(json.dumps(dataset, indent=2))

    body = json.dumps(dataset["jds"], sort_keys=True).encode()
    (out_dir / "manifest.json").write_text(json.dumps({
        "version": version, "seed": seed,
        "n_jds": dataset["n_jds"], "n_pairs": dataset["n_pairs"],
        "splits": dataset["splits"],
        "content_sha256": hashlib.sha256(body).hexdigest()[:16],
        "substitution_groups": SUBSTITUTION_GROUPS,
    }, indent=2))
    return dataset


def load(version: str = "v2", split: str | None = None) -> dict:
    ds = json.loads((Path("data/eval") / version / "dataset.json").read_text())
    if split:
        ds = {**ds, "jds": [j for j in ds["jds"] if j.get("split") == split]}
        ds["n_jds"] = len(ds["jds"])
        ds["n_pairs"] = sum(len(j["candidates"]) for j in ds["jds"])
    return ds


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v2")
    ap.add_argument("--seed", type=int, default=42)
    ds = generate(**vars(ap.parse_args()))

    from collections import Counter
    tiers = Counter(c["tier"] for j in ds["jds"] for c in j["candidates"])
    print(f"dataset {ds['version']}: {ds['n_jds']} JDs, {ds['n_pairs']} pairs "
          f"(seed {ds['seed']}) split={ds['splits']}")
    print("candidates per tier:", dict(tiers))

    print("\nmean EXACT JD-skill overlap by tier "
          "(bridgeable must sit BELOW weak for the test to discriminate):")
    for tier in GRADE:
        vals = [c["exact_overlap"] for j in ds["jds"] for c in j["candidates"]
                if c["tier"] == tier]
        print(f"  grade {GRADE[tier]} {tier:11s}: {sum(vals)/len(vals):.2f}")

    ex = ds["jds"][0]
    print(f"\nexample JD [{ex['jd_id']}]: {list(ex['jd_skills'])}")
    for tier in GRADE:
        c = next((c for c in ex["candidates"] if c["tier"] == tier), None)
        if c:
            print(f"  g{c['grade']} {tier:11s} (overlap {c['exact_overlap']}): {c['skills']}")
