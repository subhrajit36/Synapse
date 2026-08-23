# How We Evaluated Synapse — The Full Story, In Plain Language

This document explains *exactly* what we did to produce the numbers in
`RESULTS.md` and `ABLATION.md`, including the two attempts that failed first.
The failures matter: they are the reason the final numbers can be trusted.

**The one-line summary:** we built a fair test that a keyword-matching system is
guaranteed to fail, checked that our system passes it, and proved the *graph*
specifically is what makes it pass — by switching the graph off and watching the
score collapse to zero.

---

## 1. What question were we actually trying to answer?

Synapse claims something specific:

> A candidate who has **Docker** but not **Kubernetes** is a *good* match for a
> Kubernetes job, because those skills are neighbours. A keyword system rejects
> them. We rank them highly.

That is a nice story. But a story is not evidence. The question we had to answer
was blunt:

> **Does the knowledge graph actually make the ranking better — or would plain
> text similarity have worked just as well, with none of the complexity?**

If the answer is "plain similarity works just as well," then the whole graph is
decoration and we should not build infrastructure on top of it. So we had to
genuinely try to *disprove* our own idea. That is what Phase B is.

---

## 2. The building blocks

To answer that, we needed four things.

### (a) A dataset — but there isn't one

We needed examples of the form *(job description, candidate, how good is this
candidate really?)*. That last part — the correct answer — is called the **ground
truth**, and no public dataset has it for this task. Hiring data is private, and
hand-labelling enough examples to get stable statistics was not realistic.

So we **generated** the data, where the correct answer is fixed *by
construction*. We build a candidate by deliberately deciding "this one is a
strong fit / this one is a near-fit / this one is a bad fit," so we always know
the right answer because we built it in on purpose.

We used four grades:

| Grade | Tier | What it means |
|---|---|---|
| 3 | strong | Has essentially everything the job asks for |
| 2 | bridgeable | Doesn't have the exact tools, but has close equivalents |
| 1 | weak | Has a couple of the job's skills but is not really a fit |
| 0 | irrelevant | Completely different field |

### (b) The trap we had to avoid: circular reasoning

Here is the mistake that would have invalidated everything.

The obvious way to build a "bridgeable" candidate is: *ask the graph which skills
are near Kubernetes, give the candidate those, and label them relevant.*

**That is cheating.** You would be using the graph to write the exam, then giving
the graph the exam. Of course it scores 100%. It proves nothing except that the
graph agrees with itself. This is called **circularity**, and it silently ruins a
lot of ML evaluations.

Our fix: the ground truth comes from a **hand-written list of skills that humans
consider interchangeable**, written independently of the graph:

```
Docker      ≈ Kubernetes ≈ Red Hat OpenShift     (containers)
Linux       ≈ UNIX       ≈ Red Hat Enterprise Linux
Bash        ≈ UNIX Shell ≈ Shell script
PyTorch     ≈ TensorFlow ≈ Keras
MySQL       ≈ PostgreSQL ≈ Oracle Database
Tableau     ≈ Microsoft Power BI
...14 groups total
```

A hiring manager would accept any member of a group in place of another. The
graph never gets a vote in writing this list. So when the graph is later asked
"is OpenShift close to Docker?", it is being genuinely tested, not asked to
confirm its own opinion.

### (c) Baselines — the competitors

Beating nothing proves nothing, so we implemented two rival systems that are
allowed to use everything *except* the graph:

1. **TF-IDF** — the classic keyword approach. Treats each skill as a word and
   measures overlap. This is essentially how a traditional ATS works.
2. **Cosine-only** — embeds the skill lists using *the exact same AI model* our
   graph is built from, then compares them directly. **This is the important
   one.** It has the same "understanding of language" we do; the *only* thing it
   lacks is the graph structure. So if we beat it, the graph specifically is what
   won — not the embeddings, not the extraction.

### (d) Metrics — how we score a ranking

- **P@5** — of the top 5 candidates we return, how many are genuinely good?
- **nDCG@10** — rewards putting candidates in the *right order*, not just
  finding them.
- **MRR** — how quickly does the first good candidate appear?

---

## 3. Attempt #1 — and why it failed

We built the dataset, ran everything, and got this:

| Ranker | nDCG@10 | P@5 |
|---|---|---|
| Synapse | 1.000 | 1.000 |
| TF-IDF | 0.992 | 0.980 |
| cosine-only | 0.996 | 0.987 |

A beginner reads that as "we won — perfect score!" It is actually **a broken
test**, and here's why: *the dumb keyword baseline also scored 0.99.* When
everyone gets an A+, the exam is too easy to tell anyone apart. This is called a
**ceiling effect**.

The cause was our own dataset design. Our "bridgeable" candidates were still
keeping *most* of the job's exact skills and only swapping one or two. So even
plain keyword matching found them easily. **The graph was never needed, so its
value could not show up.**

> **This is the moment the evaluation earned its keep.** Without it, we would
> have believed a meaningless 1.000 and started building cloud infrastructure on
> an unproven idea.

---

## 4. Attempt #2 — designing a test that can actually discriminate

We rebuilt the dataset so the decision we care about is *front and centre* and
*hard*. The key idea:

**Make the correct answer have LESS keyword overlap than the wrong answer.**

So now:

- A **bridgeable** candidate (grade 2, the *right* answer) swaps out **every**
  substitutable skill — they hold Puppet instead of Ansible, OpenShift instead
  of Docker, Shell script instead of Bash. Almost **no** exact word overlap.
- A **weak** candidate (grade 1, the *wrong* answer) keeps **2–3 of the job's
  exact skills**, but misses everything that actually defines the role, and the
  rest of their profile is from a different field.

Measured across the dataset:

| Tier | Average number of *exact* job skills held |
|---|---|
| strong (3) | 5.70 |
| **bridgeable (2)** — right answer | **0.58** |
| **weak (1)** — wrong answer | **2.54** |
| irrelevant (0) | 0.00 |

The wrong answer now has **4× more keyword overlap** than the right answer. Any
system that just counts matching words is *forced* to get this backwards.

**Is that rigging the test?** No — and this matters. We did not remove the
baselines' ability to succeed; the cosine baseline reads the same skill names
with the same language model and is completely free to recognise that "Puppet"
relates to "Ansible." We only removed the *free win* that exact word-overlap was
handing out. A test where the naive method wins by accident measures nothing.

### The new headline metric: `bridge>weak`

The old metrics were dominated by easy decisions (a strong candidate beating an
irrelevant one — everyone gets that right). So we added a metric that measures
*only the decision that matters*:

> Take every (bridgeable, weak) pair. How often does the ranker correctly score
> the bridgeable candidate above the weak one?

0.5 means coin-flip. 1.0 means always right. This single number is the entire
premise of the project, measured directly.

### Splitting the data — so we can't fool ourselves

We split the 30 job descriptions in half:

- **train** (15) — we're allowed to tune settings here.
- **heldout** (15) — we do **not** touch this while tuning. It is the exam.

Why: if you adjust your settings until the score goes up, and then report *that
same score*, you've just described your own tinkering, not a real ability. It's
studying by memorising the answer key. The heldout number is the honest one.

---

## 5. First run on the new test — the good news and the bad news

| Ranker | bridge>weak (heldout) |
|---|---|
| Synapse (default settings) | 0.417 |
| TF-IDF | **0.002** |
| cosine-only | 0.406 |

**Good news:** TF-IDF collapsed from 0.99 to **0.002** — it gets the decision
wrong essentially every single time. That proves the test now measures the right
thing.

**Bad news, and we did not hide it:** Synapse scored **0.417** — *below 0.5*.
Worse than flipping a coin. And it was tied with the cosine baseline, meaning the
graph was adding nothing yet.

### Diagnosing it — looking at one real example

Instead of guessing, we printed the internal scoring for one pair:

```
Job needs: Ansible, Bash, Docker, Git, Jenkins CI, Prometheus

BRIDGEABLE candidate (should WIN):
   has: Jenkins CI, Puppet(≈Ansible), OpenShift(≈Docker), Shell script(≈Bash)
   → score 0.458   [direct credit 1.50 | bridge credit 1.71 | penalty 0.00]

WEAK candidate (should LOSE):
   has: Git, Jenkins CI, Prometheus + Django, Spring Boot, XGBoost
   → score 0.639   [direct credit 4.00 | bridge credit 0.48 | penalty 0.00]
```

The problem became obvious. The scoring formula was **structurally biased against
the very thing Synapse exists to do**:

- An exact skill match earned **1.0**.
- A genuine equivalent skill earned only **~0.57**.
- Missing a required skill entirely cost **nothing** (penalty was 0).

So a scattered candidate holding three incidental skills beat a genuine near-fit.
The graph was finding the right bridges — the scoring was just refusing to reward
them properly.

---

## 6. Fixing it properly — the parameter sweep

The scoring function has adjustable settings (how much credit a bridge earns, how
far away a skill can be and still count, how much a true gap is penalised, how
many hops to allow).

We tested **108 combinations** of those settings — **on the train half only**.
Then we took the single best combination and applied it, unchanged, to the
heldout half.

The winner:

```
bridge_cutoff = 0.7        (how far a bridge may stretch)
bridge_credit_scale = 2.0  (reward genuine bridges properly)
unreachable_penalty = 0.0  (no extra penalty for true gaps)
max_hops = 2               (bridge up to 2 steps through the graph)
```

Note what actually fixed it: **rewarding real bridges more**, not punishing gaps.
That matches the diagnosis exactly.

---

## 7. The final results

**Heldout split — never used during tuning:**

| Ranker | bridge>weak | nDCG@10 | P@5 |
|---|---|---|---|
| **Synapse (tuned)** | **0.859** | 0.922 | **0.960** |
| Synapse with graph switched OFF | 0.000 | 0.819 | 0.373 |
| TF-IDF (keyword) | 0.002 | 0.819 | 0.373 |
| cosine-only (embeddings, no graph) | 0.406 | 0.899 | 0.600 |

### Why the second row is the most important line in this project

`Synapse with graph switched OFF` is **the exact same program** — same data, same
AI model, same scoring code — with one setting changed: graph traversal disabled.

- Graph off → **0.000**
- Graph on → **0.859**

Nothing else differs. So the improvement *cannot* be credited to the embeddings,
the skill extraction, or the tuning. It is the graph. This is called an
**ablation** — you remove one part and see what breaks — and it is the cleanest
way to prove which component is actually responsible.

### And we beat the strong competitor too

Cosine-only isn't a strawman: it uses the same language model on the same text
and scores 0.406 (far above keyword matching). Synapse still beats it by
**+0.454** on the boundary decision and **+0.360** on P@5. So explicit graph
structure adds real signal that "just compare the embeddings" does not capture.

### The tuning was honest

| | train (tuned here) | heldout (reported) |
|---|---|---|
| default settings | 0.330 | 0.417 |
| tuned settings | 0.739 | **0.859** |

The heldout gain is *bigger* than the train gain. If we had overfitted to our
tuning data, we'd expect the opposite — heldout would sag below train. It didn't.

---

## 8. What we're NOT claiming (the honest limitations)

These are written into `RESULTS.md` on purpose. Volunteering your weak points is
what makes the strong points believable.

1. **The dataset is synthetic.** Real resumes and real recruiter judgements would
   be better evidence. Ours is carefully built to be non-circular and hard, but
   it is not a field study.
2. **Bridgeable-gap precision is only ~49%.** When Synapse labels a gap
   "bridgeable," about half the time it's bridging to something outside our
   curated equivalence list. The metric is strict (it only credits exact group
   members, not broader "you could learn this quickly" adjacency), but this is
   genuinely our weakest number and the clearest thing to improve next.
3. **MRR dipped slightly** (0.933 vs 1.000). Rewarding bridges strongly
   occasionally lifts a near-fit above a perfect fit. A real cost of the tuning,
   not swept under the rug.
4. **Job-skill importance weighting didn't help.** We tested weighting "must-have"
   skills more heavily; the uniform version scored marginally *better*. We report
   that rather than quietly deleting the experiment.

---

## 9. How to reproduce every number here

```bash
# 1. Build the evaluation dataset (fixed seed → identical every time)
python -m synapse.eval.dataset --version v2 --seed 42

# 2. Head-to-head vs both baselines → writes RESULTS.md
python -m synapse.eval.run_eval

# 3. Parameter sweep + ablation → writes ABLATION.md
python -m synapse.eval.ablation
```

Everything is seeded and version-pinned, and `RESULTS.md` is *generated* — the
written interpretation is produced from the same run as the tables, so the prose
can never drift away from the numbers it describes.

---

## 10. If someone asks you about this in an interview

**"How do you know your graph actually helps?"**
> I ran an ablation. Same pipeline, same embeddings, graph traversal toggled off:
> the score on the decision that matters drops from 0.86 to 0.00. Nothing else
> changed, so the graph is what's doing the work.

**"Isn't your test rigged in your favour?"**
> The opposite — I had to rebuild it because it was too *easy*. My first version
> was saturated: even TF-IDF scored 0.99, so nothing could be distinguished. I
> redesigned it so the correct answer has 4× *less* keyword overlap than the wrong
> answer. And the ground truth comes from a hand-written list of interchangeable
> skills that never reads the graph, so there's no circularity.

**"Did you tune on your test set?"**
> No. I split the job descriptions in half, swept 108 configurations on train
> only, then applied the winner unchanged to a heldout half. Heldout scored 0.859
> — higher than train's 0.739, which is the opposite of what overfitting looks
> like.

**"What's the weakest part of your system?"**
> Bridgeable-gap precision, around 49%. Half the bridges land outside my curated
> equivalence groups. My next step is adding co-occurrence (PMI) as a third
> edge-construction signal so edges aren't relying on embedding similarity alone.

**"Why not just use embedding similarity? It's simpler."**
> I tested exactly that as a baseline — same model, same text, no graph. It gets
> 0.406 on the boundary decision versus my 0.859. Pooled similarity blurs a
> candidate's skills into one average vector; the graph reasons about individual
> skills and can tell you *which* skill bridges to *which*, with a path to show
> for it. That's also what makes the output explainable.
