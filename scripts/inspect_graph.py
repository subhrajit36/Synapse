#!/usr/bin/env python3
"""
inspect_graph.py — audit and visualise the Synapse skill graph.

Reads a pickled NetworkX graph (default: data/skill_graph.pkl), prints a
structural report to stdout, and writes a self-contained interactive HTML
viewer (no CDN, no network) you can open in a browser.

Usage
-----
    python scripts/inspect_graph.py
    python scripts/inspect_graph.py --graph data/skill_graph.pkl --out reports/graph.html
    python scripts/inspect_graph.py --around "Programming" --hops 2
    python scripts/inspect_graph.py --mode random --max-nodes 250 --seed 7

Modes for the rendered subgraph:
    hubs    (default) highest-degree nodes and the edges induced among them
    around  ego-network of --around out to --hops
    random  uniform random node sample

Everything else in the report is computed over the *full* graph.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import random
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

try:
    import networkx as nx
except ImportError:  # pragma: no cover
    sys.exit("networkx is required: pip install networkx")


WEIGHT_KEY_CANDIDATES = ("weight", "w", "cost", "distance", "similarity", "score")
LABEL_KEY_CANDIDATES = ("name", "label", "title", "canonical", "skill")
TYPE_KEY_CANDIDATES = ("type", "kind", "layer", "category", "node_type", "element_type")
REL_KEY_CANDIDATES = ("relation", "rel", "type", "edge_type", "kind")


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------

def load_graph(path: Path):
    """Unpickle a graph. Tolerates a graph stored inside a dict wrapper."""
    with path.open("rb") as fh:
        obj = pickle.load(fh)

    if isinstance(obj, nx.Graph):
        return obj

    if isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(value, nx.Graph):
                print(f"  note: graph found under pickle key {key!r}")
                return value

    raise TypeError(
        f"{path} unpickled to {type(obj).__name__}, not a networkx graph. "
        "Point --graph at the file that holds the graph object."
    )


def detect_key(sample_dicts, candidates):
    """Return the first candidate key present in a sample of attribute dicts."""
    seen = Counter()
    for d in sample_dicts:
        seen.update(d.keys())
    for cand in candidates:
        if seen.get(cand):
            return cand
    return None


# --------------------------------------------------------------------------
# profiling
# --------------------------------------------------------------------------

def attribute_coverage(items, total):
    """items: iterable of attr dicts -> {key: (count, pct, example)}"""
    counts = Counter()
    examples = {}
    for d in items:
        for k, v in d.items():
            counts[k] += 1
            if k not in examples:
                examples[k] = v
    out = {}
    for k, c in counts.most_common():
        ex = examples[k]
        if isinstance(ex, (list, tuple)) and len(ex) > 6:
            ex = f"<{type(ex).__name__} len={len(ex)}>"
        else:
            ex = str(ex)
            if len(ex) > 48:
                ex = ex[:45] + "..."
        out[k] = (c, 100.0 * c / total if total else 0.0, ex)
    return out


def normalise(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(label).lower())


def hop_profile(G, sources, cutoff=4):
    """Average share of the graph reachable at each hop distance."""
    buckets = defaultdict(list)
    for s in sources:
        lengths = nx.single_source_shortest_path_length(G, s, cutoff=cutoff)
        reach = Counter(lengths.values())
        n = G.number_of_nodes() - 1
        if n <= 0:
            continue
        cumulative = 0
        for h in range(1, cutoff + 1):
            cumulative += reach.get(h, 0)
            buckets[h].append(100.0 * cumulative / n)
    return {h: statistics.mean(v) for h, v in sorted(buckets.items()) if v}


def path_divergence(G, sources, weight_key):
    """
    How often the cheapest-weighted path uses more hops than the
    fewest-hop path. Quantifies why hops and weighted distance need
    separate traversals.
    """
    if not weight_key:
        return None
    differ = same = 0
    extra_hops = []
    for s in sources:
        try:
            _, paths = nx.single_source_dijkstra(G, s, weight=weight_key)
        except Exception:
            continue
        bfs = nx.single_source_shortest_path_length(G, s)
        for target, path in paths.items():
            if target == s:
                continue
            w_hops = len(path) - 1
            b_hops = bfs.get(target)
            if b_hops is None:
                continue
            if w_hops > b_hops:
                differ += 1
                extra_hops.append(w_hops - b_hops)
            else:
                same += 1
    total = differ + same
    if not total:
        return None
    return {
        "pairs": total,
        "divergent": differ,
        "pct": 100.0 * differ / total,
        "mean_extra_hops": statistics.mean(extra_hops) if extra_hops else 0.0,
    }


def profile(G, sample=60, seed=13):
    rng = random.Random(seed)
    n, m = G.number_of_nodes(), G.number_of_edges()
    nodes = list(G.nodes())

    node_attrs = [d for _, d in G.nodes(data=True)]
    edge_attrs = [d for _, _, d in G.edges(data=True)]

    weight_key = detect_key(edge_attrs, WEIGHT_KEY_CANDIDATES)
    label_key = detect_key(node_attrs, LABEL_KEY_CANDIDATES)
    type_key = detect_key(node_attrs, TYPE_KEY_CANDIDATES)
    rel_key = detect_key(edge_attrs, REL_KEY_CANDIDATES)
    if rel_key == weight_key:
        rel_key = None

    degrees = dict(G.degree())
    deg_values = sorted(degrees.values())

    # weights
    weights = []
    if weight_key:
        weights = [d[weight_key] for d in edge_attrs
                   if isinstance(d.get(weight_key), (int, float))]

    # components
    if G.is_directed():
        comps = sorted((len(c) for c in nx.weakly_connected_components(G)), reverse=True)
        comp_kind = "weakly connected"
    else:
        comps = sorted((len(c) for c in nx.connected_components(G)), reverse=True)
        comp_kind = "connected"

    # label hygiene
    labels = [str(G.nodes[x].get(label_key, x)) if label_key else str(x) for x in nodes]
    name_lengths = [len(x) for x in labels]
    word_counts = [len(x.split()) for x in labels]
    collisions = defaultdict(list)
    for lab in labels:
        collisions[normalise(lab)].append(lab)
    dupes = {k: v for k, v in collisions.items() if len(set(v)) > 1}

    # reachability sampling on the largest component
    if n:
        if G.is_directed():
            big = max(nx.weakly_connected_components(G), key=len)
        else:
            big = max(nx.connected_components(G), key=len)
        big = list(big)
        sources = rng.sample(big, min(sample, len(big)))
    else:
        big, sources = [], []

    return {
        "n_nodes": n,
        "n_edges": m,
        "directed": G.is_directed(),
        "multigraph": G.is_multigraph(),
        "density": nx.density(G) if n > 1 else 0.0,
        "self_loops": nx.number_of_selfloops(G),
        "weight_key": weight_key,
        "label_key": label_key,
        "type_key": type_key,
        "rel_key": rel_key,
        "node_attrs": attribute_coverage(node_attrs, n),
        "edge_attrs": attribute_coverage(edge_attrs, m),
        "node_types": Counter(d.get(type_key) for d in node_attrs) if type_key else None,
        "edge_rels": Counter(d.get(rel_key) for d in edge_attrs) if rel_key else None,
        "weights": weights,
        "components": comps,
        "component_kind": comp_kind,
        "largest_component_pct": 100.0 * comps[0] / n if comps and n else 0.0,
        "isolated": [x for x in nodes if degrees[x] == 0][:20],
        "n_isolated": sum(1 for x in nodes if degrees[x] == 0),
        "n_leaf": sum(1 for v in deg_values if v == 1),
        "degrees": deg_values,
        "hubs": sorted(degrees.items(), key=lambda kv: -kv[1])[:15],
        "name_len_mean": statistics.mean(name_lengths) if name_lengths else 0,
        "name_len_max": max(name_lengths) if name_lengths else 0,
        "word_count_mean": statistics.mean(word_counts) if word_counts else 0,
        "multiword_pct": 100.0 * sum(1 for w in word_counts if w > 1) / n if n else 0,
        "duplicate_groups": list(dupes.items())[:10],
        "n_duplicate_groups": len(dupes),
        "hops": hop_profile(G, sources) if sources else {},
        "divergence": path_divergence(G, sources[:20], weight_key) if sources else None,
        "sample_size": len(sources),
    }


# --------------------------------------------------------------------------
# console report
# --------------------------------------------------------------------------

def bar(pct, width=28, fill="\u2588", empty="\u00b7"):
    filled = int(round(width * min(pct, 100) / 100))
    return fill * filled + empty * (width - filled)


def pct_of(values, q):
    if not values:
        return 0
    k = (len(values) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    if lo == hi:
        return values[int(k)]
    return values[lo] + (values[hi] - values[lo]) * (k - lo)


def report(p):
    W = 66
    def rule(title=""):
        if title:
            print(f"\n\u2500\u2500 {title} " + "\u2500" * max(0, W - len(title) - 4))
        else:
            print("\u2500" * W)

    rule()
    kind = "directed" if p["directed"] else "undirected"
    kind += " multigraph" if p["multigraph"] else " graph"
    print(f"  {p['n_nodes']:,} nodes   {p['n_edges']:,} edges   {kind}")
    print(f"  density {p['density']:.5f}   self-loops {p['self_loops']}")
    rule()

    rule("node attributes")
    for k, (c, pc, ex) in p["node_attrs"].items():
        print(f"  {k:<18} {pc:5.1f}%  e.g. {ex}")
    if not p["node_attrs"]:
        print("  (none — nodes carry no attributes)")

    rule("edge attributes")
    for k, (c, pc, ex) in p["edge_attrs"].items():
        print(f"  {k:<18} {pc:5.1f}%  e.g. {ex}")
    if not p["edge_attrs"]:
        print("  (none — edges carry no attributes)")

    if p["node_types"]:
        rule(f"node breakdown by '{p['type_key']}'")
        for t, c in p["node_types"].most_common(12):
            print(f"  {str(t):<28} {c:>7,}  {bar(100*c/p['n_nodes'], 20)}")

    if p["edge_rels"]:
        rule(f"edge breakdown by '{p['rel_key']}'")
        for t, c in p["edge_rels"].most_common(12):
            print(f"  {str(t):<28} {c:>7,}  {bar(100*c/p['n_edges'], 20)}")

    ws = sorted(p["weights"])
    if ws:
        rule(f"edge weight '{p['weight_key']}'")
        print(f"  min {min(ws):.4f}   p50 {pct_of(ws,.5):.4f}   "
              f"p90 {pct_of(ws,.9):.4f}   max {max(ws):.4f}")
        if min(ws) >= 0 and max(ws) <= 1:
            print("  ! all weights in [0,1]. Dijkstra minimises weight, so this only")
            print("    reads as a cost if smaller means 'more related'. If these are")
            print("    similarities, traversal is currently walking the weakest edges.")
    elif p["weight_key"] is None:
        print("\n  ! no numeric edge weight found — every edge costs 1 to Dijkstra")

    rule("connectivity")
    print(f"  {len(p['components'])} {p['component_kind']} component(s)")
    print(f"  largest holds {p['largest_component_pct']:.1f}% of nodes "
          f"({p['components'][0] if p['components'] else 0:,})")
    if len(p["components"]) > 1:
        print(f"  next largest: {p['components'][1:6]}")
    print(f"  isolated nodes: {p['n_isolated']}   degree-1 nodes: {p['n_leaf']}")
    if p["isolated"]:
        print(f"  e.g. {', '.join(str(x)[:30] for x in p['isolated'][:4])}")

    d = p["degrees"]
    if d:
        rule("degree")
        print(f"  min {d[0]}   p50 {pct_of(d,.5):.0f}   p90 {pct_of(d,.9):.0f}   "
              f"p99 {pct_of(d,.99):.0f}   max {d[-1]}")
        print("  top hubs:")
        for node, deg in p["hubs"][:8]:
            print(f"    {deg:>5}  {str(node)[:52]}")

    if p["hops"]:
        rule(f"reachability  (BFS from {p['sample_size']} sampled nodes)")
        print("  share of graph reachable within h hops, averaged:")
        for h, v in p["hops"].items():
            print(f"    h\u2264{h}   {v:6.2f}%  {bar(v)}")
        h1 = p["hops"].get(1, 0)
        h2 = p["hops"].get(2, 0)
        if h2 > 60:
            print("\n  ! at max_hops=2 most of the graph is 'bridgeable' — the label")
            print("    stops discriminating. Consider max_hops=1, or gate bridging on")
            print("    weighted path cost rather than hop count alone.")
        elif h1 < 0.5 and h2 < 5:
            print("\n  ! the graph is sparse enough that 2-hop bridging will fire rarely")

    dv = p["divergence"]
    if dv:
        rule("hop vs weighted path")
        print(f"  {dv['pct']:.1f}% of {dv['pairs']:,} sampled pairs: the cheapest-weighted")
        print(f"  path is longer in hops than the fewest-hop path "
              f"(+{dv['mean_extra_hops']:.2f} hops avg)")
        print("  -> hop count and weighted distance need separate traversals")

    rule("label hygiene")
    print(f"  mean name length {p['name_len_mean']:.1f} chars "
          f"(max {p['name_len_max']})")
    print(f"  mean words per name {p['word_count_mean']:.2f}   "
          f"multi-word {p['multiword_pct']:.1f}%")
    print(f"  case/punctuation-collapsed duplicate groups: {p['n_duplicate_groups']}")
    for _, variants in p["duplicate_groups"][:5]:
        print(f"    {' | '.join(sorted(set(variants))[:4])}")
    if p["word_count_mean"] > 3:
        print("  ! long multi-word canonical names — a single cosine threshold tuned")
        print("    on short skill strings will not transfer. Calibrate empirically.")
    rule()


# --------------------------------------------------------------------------
# subgraph selection
# --------------------------------------------------------------------------

def resolve_node(G, query, label_key):
    if query in G:
        return query
    q = normalise(query)
    exact, partial = [], []
    for node in G.nodes():
        lab = str(G.nodes[node].get(label_key, node)) if label_key else str(node)
        nl = normalise(lab)
        if nl == q:
            exact.append(node)
        elif q and q in nl:
            partial.append(node)
    if exact:
        return exact[0]
    if partial:
        print(f"  note: '{query}' matched {len(partial)} nodes, using {partial[0]!r}")
        return partial[0]
    raise SystemExit(f"No node matching {query!r}. Try a substring of a real node name.")


def pick_subgraph(G, mode, around, hops, max_nodes, seed, label_key):
    rng = random.Random(seed)
    if mode == "around":
        if not around:
            raise SystemExit("--mode around needs --around 'some skill name'")
        seed_node = resolve_node(G, around, label_key)
        ego = nx.ego_graph(G.to_undirected(as_view=True), seed_node, radius=hops)
        keep = list(ego.nodes())
        if len(keep) > max_nodes:
            deg = dict(G.degree(keep))
            keep = [seed_node] + [x for x in sorted(keep, key=lambda k: -deg[k])
                                  if x != seed_node][:max_nodes - 1]
        return G.subgraph(keep).copy(), seed_node

    if mode == "random":
        keep = rng.sample(list(G.nodes()), min(max_nodes, G.number_of_nodes()))
        return G.subgraph(keep).copy(), None

    deg = dict(G.degree())
    keep = [n for n, _ in sorted(deg.items(), key=lambda kv: -kv[1])[:max_nodes]]
    return G.subgraph(keep).copy(), None


def serialise(SG, G, label_key, weight_key, type_key, seed_node):
    full_deg = dict(G.degree())
    nodes = []
    index = {}
    for i, n in enumerate(SG.nodes()):
        index[n] = i
        label = str(SG.nodes[n].get(label_key, n)) if label_key else str(n)
        nodes.append({
            "id": i,
            "label": label,
            "deg": SG.degree(n),
            "fullDeg": full_deg.get(n, 0),
            "type": str(SG.nodes[n].get(type_key)) if type_key else None,
            "seed": n == seed_node,
        })
    links = []
    for u, v, d in SG.edges(data=True):
        w = d.get(weight_key) if weight_key else None
        links.append({
            "s": index[u],
            "t": index[v],
            "w": float(w) if isinstance(w, (int, float)) else None,
        })
    return {"nodes": nodes, "links": links}


# --------------------------------------------------------------------------
# html
# --------------------------------------------------------------------------

HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
  :root{
    --ink:#0e141b; --panel:#141d27; --rule:#243244; --line:#31455c;
    --text:#c6d3e1; --dim:#71879e; --faint:#48596c;
    --teal:#78d3c6; --amber:#f0b45f; --coral:#e0796f; --violet:#9b9ae0;
    --mono: ui-monospace, "SF Mono", "Cascadia Mono", Menlo, Consolas, monospace;
  }
  *{box-sizing:border-box}
  html,body{margin:0;height:100%}
  body{
    background:var(--ink); color:var(--text); font-family:var(--mono);
    font-size:13px; line-height:1.5; -webkit-font-smoothing:antialiased;
  }
  .wrap{display:grid; grid-template-columns:300px 1fr; height:100vh}
  aside{
    border-right:1px solid var(--rule); overflow-y:auto; padding:22px 20px 40px;
    background:linear-gradient(180deg,#111a23,#0e141b 40%);
  }
  main{position:relative; display:flex; flex-direction:column; min-width:0}

  .eyebrow{
    font-size:10px; letter-spacing:.24em; text-transform:uppercase;
    color:var(--faint); margin:0 0 6px;
  }
  h1{
    font-size:19px; font-weight:600; letter-spacing:-.01em; margin:0 0 2px;
    color:#eaf1f8;
  }
  .sub{color:var(--dim); font-size:11px; margin:0 0 24px}

  .block{margin-bottom:24px}
  .block h2{
    font-size:10px; letter-spacing:.2em; text-transform:uppercase;
    color:var(--faint); font-weight:500; margin:0 0 10px;
    padding-bottom:6px; border-bottom:1px solid var(--line);
  }
  .kv{display:flex; justify-content:space-between; gap:10px; padding:2px 0}
  .kv span:first-child{color:var(--dim)}
  .kv span:last-child{color:var(--text); font-variant-numeric:tabular-nums}
  .warn{
    color:var(--amber); font-size:11px; line-height:1.45; margin-top:10px;
    padding-left:10px; border-left:2px solid var(--amber);
  }

  /* signature: the hop range-finder */
  .hop{margin:2px 0 9px}
  .hop-top{display:flex; justify-content:space-between; font-size:11px}
  .hop-top b{font-weight:600; color:var(--teal); font-variant-numeric:tabular-nums}
  .track{height:5px; background:var(--line); margin-top:4px; position:relative; overflow:hidden}
  .fill{position:absolute; inset:0 auto 0 0; background:var(--teal)}
  .hop:nth-child(3) .fill{background:var(--amber)}
  .hop:nth-child(4) .fill{background:var(--coral)}
  .hop:nth-child(5) .fill{background:var(--violet)}

  .bar{display:flex; align-items:center; gap:8px; padding:1px 0}
  .bar i{
    height:8px; background:var(--faint); display:block; flex:none; min-width:1px;
  }
  .bar em{font-style:normal; color:var(--dim); font-size:11px;
          overflow:hidden; text-overflow:ellipsis; white-space:nowrap}

  header{
    display:flex; align-items:center; gap:14px; flex-wrap:wrap;
    padding:14px 20px; border-bottom:1px solid var(--rule);
  }
  header .t{font-size:12px; color:var(--dim)}
  header .t b{color:var(--text); font-weight:600}
  .legend{display:flex; gap:14px; margin-left:auto; font-size:11px; color:var(--dim)}
  .legend i{display:inline-block; width:8px; height:8px; border-radius:50%;
            margin-right:5px; vertical-align:middle}
  input[type=search]{
    background:var(--panel); border:1px solid var(--rule); color:var(--text);
    font-family:var(--mono); font-size:12px; padding:5px 9px; width:190px;
  }
  input[type=search]:focus{outline:2px solid var(--teal); outline-offset:1px}
  button{
    background:transparent; border:1px solid var(--rule); color:var(--dim);
    font-family:var(--mono); font-size:11px; padding:5px 10px; cursor:pointer;
  }
  button:hover{color:var(--text); border-color:var(--faint)}
  button:focus-visible{outline:2px solid var(--teal); outline-offset:1px}

  #stage{flex:1; position:relative; min-height:0}
  canvas{display:block; width:100%; height:100%; cursor:grab}
  canvas.drag{cursor:grabbing}
  #tip{
    position:absolute; pointer-events:none; background:#0b1118ee;
    border:1px solid var(--rule); padding:7px 10px; font-size:11px;
    max-width:280px; opacity:0; transition:opacity .1s; z-index:5;
  }
  #tip b{display:block; color:#eaf1f8; font-size:12px; margin-bottom:3px;
         white-space:normal}
  #tip span{color:var(--dim)}
  footer{
    padding:8px 20px; border-top:1px solid var(--rule); color:var(--faint);
    font-size:11px; display:flex; gap:18px; flex-wrap:wrap;
  }
  @media (max-width:820px){
    .wrap{grid-template-columns:1fr; height:auto}
    aside{border-right:0; border-bottom:1px solid var(--rule)}
    #stage{height:70vh}
  }
  @media (prefers-reduced-motion:reduce){ *{transition:none !important} }
</style>
</head>
<body>
<div class="wrap">
  <aside>
    <p class="eyebrow">skill graph audit</p>
    <h1>__NAME__</h1>
    <p class="sub">__STAMP__</p>
    <div id="panel"></div>
  </aside>
  <main>
    <header>
      <span class="t"><b id="shown">0</b> nodes shown &middot; <span id="shownE">0</span> edges &middot; <span id="pickmode"></span></span>
      <input type="search" id="find" placeholder="find a skill" aria-label="Find a skill">
      <button id="freeze">pause layout</button>
      <button id="reset">re-centre</button>
      <span class="legend">
        <span><i style="background:var(--teal)"></i>node</span>
        <span><i style="background:var(--amber)"></i>hub</span>
        <span><i style="background:var(--coral)"></i>focus</span>
      </span>
    </header>
    <div id="stage">
      <canvas id="cv"></canvas>
      <div id="tip"></div>
    </div>
    <footer>
      <span>drag a node to pin &middot; double-click to release</span>
      <span>scroll to zoom &middot; drag background to pan</span>
      <span>node size = degree in the full graph</span>
    </footer>
  </main>
</div>
<script>
const DATA = __DATA__;
const META = __META__;

/* ---------- side panel ---------- */
(function panel(){
  const el = document.getElementById('panel');
  const num = n => n.toLocaleString();
  let h = '';

  h += '<div class="block"><h2>shape</h2>';
  for (const [k,v] of META.shape) h += `<div class="kv"><span>${k}</span><span>${v}</span></div>`;
  h += '</div>';

  if (META.hops.length){
    h += '<div class="block"><h2>reach by hop</h2>';
    for (const [hop,pct] of META.hops){
      h += `<div class="hop"><div class="hop-top"><span>h &le; ${hop}</span><b>${pct.toFixed(1)}%</b></div>`
         + `<div class="track"><div class="fill" style="width:${Math.min(pct,100)}%"></div></div></div>`;
    }
    h += `<p class="sub" style="margin:8px 0 0">share of the graph a candidate can reach from one held skill</p>`;
    if (META.hopWarn) h += `<p class="warn">${META.hopWarn}</p>`;
    h += '</div>';
  }

  if (META.components.length){
    h += '<div class="block"><h2>components</h2>';
    const max = META.components[0];
    for (const c of META.components.slice(0,6)){
      h += `<div class="bar"><i style="width:${Math.max(2,120*c/max)}px"></i><em>${num(c)} nodes</em></div>`;
    }
    if (META.components.length>6) h += `<div class="kv"><span>+ ${META.components.length-6} more</span><span></span></div>`;
    h += '</div>';
  }

  if (META.hubs.length){
    h += '<div class="block"><h2>hubs</h2>';
    const max = META.hubs[0][1];
    for (const [name,deg] of META.hubs){
      h += `<div class="bar"><i style="width:${Math.max(2,70*deg/max)}px;background:var(--amber)"></i>`
         + `<em title="${esc(name)}">${deg} &nbsp;${esc(name)}</em></div>`;
    }
    h += '</div>';
  }

  if (META.notes.length){
    h += '<div class="block"><h2>flags</h2>';
    for (const n of META.notes) h += `<p class="warn">${n}</p>`;
    h += '</div>';
  }
  el.innerHTML = h;
})();
function esc(s){return String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));}

/* ---------- force layout on canvas ---------- */
const cv = document.getElementById('cv'), ctx = cv.getContext('2d');
const tip = document.getElementById('tip'), stage = document.getElementById('stage');
const N = DATA.nodes, L = DATA.links;
document.getElementById('shown').textContent = N.length.toLocaleString();
document.getElementById('shownE').textContent = L.length.toLocaleString();
document.getElementById('pickmode').textContent = META.mode;

const maxDeg = Math.max(1, ...N.map(n=>n.fullDeg));
const R = n => 3 + 7*Math.sqrt(n.fullDeg/maxDeg);
const adj = N.map(()=>[]);
L.forEach(l=>{adj[l.s].push(l.t); adj[l.t].push(l.s);});

let W=0,H=0,dpr=Math.min(devicePixelRatio||1,2);
function size(){
  W=stage.clientWidth; H=stage.clientHeight;
  cv.width=W*dpr; cv.height=H*dpr; ctx.setTransform(dpr,0,0,dpr,0,0);
}
size(); addEventListener('resize',()=>{size();});

const rand=(s=>()=>((s=s*16807%2147483647)/2147483647))(42);
N.forEach(n=>{
  const a=rand()*Math.PI*2, r=Math.min(W,H)*0.35*Math.sqrt(rand());
  n.x=W/2+Math.cos(a)*r; n.y=H/2+Math.sin(a)*r; n.vx=0; n.vy=0; n.fx=null;
});

const byDeg = N.slice().sort((a,b)=>b.fullDeg-a.fullDeg);
let alpha=1, running=true, view={k:1,x:0,y:0}, hover=null, focus=null, dragNode=null, fitted=0;

function fitView(pad=70){
  if(!N.length) return;
  let x0=Infinity,y0=Infinity,x1=-Infinity,y1=-Infinity;
  for(const n of N){ x0=Math.min(x0,n.x); y0=Math.min(y0,n.y); x1=Math.max(x1,n.x); y1=Math.max(y1,n.y); }
  const w=Math.max(x1-x0,1), h=Math.max(y1-y0,1);
  const k=Math.max(0.2, Math.min(2.2,(W-pad*2)/w, (H-pad*2)/h));
  view.k=k; view.x=W/2-(x0+x1)/2*k; view.y=H/2-(y0+y1)/2*k;
}
const K = Math.sqrt((W*H)/Math.max(N.length,1)) * 0.85;

function step(){
  if(!running||alpha<0.005) return;
  alpha *= 0.994;
  // repulsion (grid-bucketed)
  const cell=K, buckets=new Map();
  N.forEach((n,i)=>{
    const key=((n.x/cell)|0)+':'+((n.y/cell)|0);
    if(!buckets.has(key)) buckets.set(key,[]);
    buckets.get(key).push(i);
  });
  N.forEach((n,i)=>{
    const cx=(n.x/cell)|0, cy=(n.y/cell)|0;
    for(let dx=-1;dx<=1;dx++) for(let dy=-1;dy<=1;dy++){
      const b=buckets.get((cx+dx)+':'+(cy+dy)); if(!b) continue;
      for(const j of b){
        if(j<=i) continue;
        const o=N[j]; let ax=n.x-o.x, ay=n.y-o.y;
        let d2=ax*ax+ay*ay; if(d2===0){ax=rand()-0.5;ay=rand()-0.5;d2=0.01;}
        if(d2>cell*cell*4) continue;
        const f=(K*K)/d2*alpha*1.0, d=Math.sqrt(d2);
        const ux=ax/d*f, uy=ay/d*f;
        n.vx+=ux; n.vy+=uy; o.vx-=ux; o.vy-=uy;
      }
    }
  });
  // springs
  for(const l of L){
    const a=N[l.s], b=N[l.t];
    const dx=b.x-a.x, dy=b.y-a.y, d=Math.hypot(dx,dy)||0.01;
    const f=(d-K*0.75)/d*alpha*0.06;
    const ux=dx*f, uy=dy*f;
    a.vx+=ux; a.vy+=uy; b.vx-=ux; b.vy-=uy;
  }
  // gravity + integrate
  for(const n of N){
    if(n.fx!==null){ n.x=n.fx; n.y=n.fy; n.vx=n.vy=0; continue; }
    n.vx+=(W/2-n.x)*0.0009*alpha; n.vy+=(H/2-n.y)*0.0009*alpha;
    n.vx*=0.86; n.vy*=0.86;
    n.x+=Math.max(-30,Math.min(30,n.vx));
    n.y+=Math.max(-30,Math.min(30,n.vy));
  }
}

const css = v => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const C = {teal:css('--teal'), amber:css('--amber'), coral:css('--coral'),
           line:css('--line'), dim:css('--dim'), text:css('--text'), rule:css('--rule')};

function draw(){
  ctx.clearRect(0,0,W,H);
  ctx.save(); ctx.translate(view.x,view.y); ctx.scale(view.k,view.k);
  const lit = focus!==null ? new Set([focus,...adj[focus]]) : null;

  ctx.lineWidth = 1/view.k;
  for(const l of L){
    const a=N[l.s], b=N[l.t];
    const on = lit ? (lit.has(l.s)&&lit.has(l.t)) : false;
    ctx.strokeStyle = on ? C.coral : C.line;
    ctx.globalAlpha = lit ? (on?0.95:0.10) : (l.w!=null? 0.35+0.45*l.wN : 0.6);
    ctx.beginPath(); ctx.moveTo(a.x,a.y); ctx.lineTo(b.x,b.y); ctx.stroke();
  }
  ctx.globalAlpha=1;
  for(const n of N){
    const r=R(n);
    const isHub = n.fullDeg >= maxDeg*0.55;
    let fill = isHub ? C.amber : C.teal;
    if(n.seed) fill = C.coral;
    ctx.globalAlpha = lit ? (lit.has(n.id)?1:0.15) : 1;
    ctx.beginPath(); ctx.arc(n.x,n.y,r,0,6.2832);
    ctx.fillStyle=fill; ctx.fill();
    if(n.id===focus||n.id===hover){
      ctx.lineWidth=2/view.k; ctx.strokeStyle='#eaf1f8'; ctx.stroke();
    }
  }
  ctx.globalAlpha=1;

  // labels: lit neighbourhood, else the top few by degree; overlaps culled
  let cand = lit ? N.filter(n=>lit.has(n.id)) : byDeg.slice(0, 14);
  if(hover!=null && !cand.some(n=>n.id===hover)) cand = [N[hover], ...cand];
  ctx.font = `${12/view.k}px ui-monospace, Menlo, monospace`;
  ctx.textAlign='center'; ctx.lineJoin='round';
  const boxes=[];
  for(const n of cand.slice(0,40)){
    const t = n.label.length>30 ? n.label.slice(0,28)+'\u2026' : n.label;
    const w = ctx.measureText(t).width;
    const x = n.x, y = n.y - R(n) - 6/view.k;
    const sx = x*view.k+view.x, sy = y*view.k+view.y, sw = w*view.k;
    const box=[sx-sw/2-3, sy-13, sx+sw/2+3, sy+3];
    if(boxes.some(b=>!(box[2]<b[0]||box[0]>b[2]||box[3]<b[1]||box[1]>b[3]))) continue;
    boxes.push(box);
    ctx.lineWidth = 3.5/view.k; ctx.strokeStyle = 'rgba(14,20,27,0.92)';
    ctx.strokeText(t, x, y);
    ctx.fillStyle = (n.id===hover||n.id===focus) ? '#eaf1f8' : C.text;
    ctx.fillText(t, x, y);
  }
  ctx.restore();
}

const wmax = Math.max(...L.map(l=>l.w??0), 1), wmin = Math.min(...L.map(l=>l.w??0), 0);
L.forEach(l => l.wN = l.w==null ? 0.5 : (wmax===wmin?0.5:(l.w-wmin)/(wmax-wmin)));

function tick(){
  step(); draw();
  if(fitted<2 && alpha<(fitted?0.02:0.35)){ fitted++; fitView(); }
  requestAnimationFrame(tick);
}
tick();

/* ---------- interaction ---------- */
function toWorld(e){
  const r=cv.getBoundingClientRect();
  return {x:(e.clientX-r.left-view.x)/view.k, y:(e.clientY-r.top-view.y)/view.k};
}
function at(p){
  let best=null,bd=1e9;
  for(const n of N){
    const d=Math.hypot(n.x-p.x,n.y-p.y);
    if(d<Math.max(R(n)+4,9) && d<bd){bd=d;best=n;}
  }
  return best;
}
let panning=null;
cv.addEventListener('mousedown',e=>{
  const p=toWorld(e), n=at(p);
  if(n){ dragNode=n; n.fx=n.x; n.fy=n.y; focus=n.id; alpha=Math.max(alpha,0.3); }
  else { panning={x:e.clientX-view.x, y:e.clientY-view.y}; focus=null; }
  cv.classList.add('drag');
});
addEventListener('mousemove',e=>{
  if(dragNode){ const p=toWorld(e); dragNode.fx=p.x; dragNode.fy=p.y; alpha=Math.max(alpha,0.15); return; }
  if(panning){ view.x=e.clientX-panning.x; view.y=e.clientY-panning.y; return; }
  const r=cv.getBoundingClientRect();
  if(e.clientX<r.left||e.clientX>r.right||e.clientY<r.top||e.clientY>r.bottom){
    hover=null; tip.style.opacity=0; return;
  }
  const n=at(toWorld(e));
  hover = n? n.id : null;
  if(n){
    tip.innerHTML = `<b>${esc(n.label)}</b><span>degree ${n.deg} here &middot; ${n.fullDeg} in full graph`
                  + (n.type&&n.type!=='None'?` &middot; ${esc(n.type)}`:'') + `</span>`;
    tip.style.opacity=1;
    tip.style.left=Math.min(e.clientX-r.left+14, r.width-290)+'px';
    tip.style.top=(e.clientY-r.top+14)+'px';
  } else tip.style.opacity=0;
});
addEventListener('mouseup',()=>{ dragNode=null; panning=null; cv.classList.remove('drag'); });
cv.addEventListener('dblclick',e=>{ const n=at(toWorld(e)); if(n){n.fx=null;n.fy=null;alpha=Math.max(alpha,0.3);} });
cv.addEventListener('wheel',e=>{
  e.preventDefault();
  const r=cv.getBoundingClientRect(), mx=e.clientX-r.left, my=e.clientY-r.top;
  const k=Math.exp(-e.deltaY*0.0015), nk=Math.max(0.15,Math.min(6,view.k*k));
  view.x = mx-(mx-view.x)*(nk/view.k); view.y = my-(my-view.y)*(nk/view.k); view.k=nk;
},{passive:false});

document.getElementById('find').addEventListener('input',e=>{
  const q=e.target.value.trim().toLowerCase();
  if(!q){ focus=null; return; }
  const hit=N.find(n=>n.label.toLowerCase().includes(q));
  if(hit){
    focus=hit.id;
    view.k=Math.max(view.k,1.2); view.x=W/2-hit.x*view.k; view.y=H/2-hit.y*view.k;
  }
});
document.getElementById('freeze').addEventListener('click',e=>{
  running=!running; e.target.textContent = running?'pause layout':'resume layout';
  if(running) alpha=Math.max(alpha,0.2);
});
document.getElementById('reset').addEventListener('click',()=>{
  document.getElementById('find').value=''; focus=null; fitted=2; fitView();
});
</script>
</body>
</html>
"""


def build_html(payload, meta, name, stamp, title):
    return (HTML
            .replace("__DATA__", json.dumps(payload))
            .replace("__META__", json.dumps(meta))
            .replace("__NAME__", name)
            .replace("__STAMP__", stamp)
            .replace("__TITLE__", title))


def build_meta(p, mode_desc):
    shape = [
        ["nodes", f"{p['n_nodes']:,}"],
        ["edges", f"{p['n_edges']:,}"],
        ["density", f"{p['density']:.5f}"],
        ["components", f"{len(p['components'])}"],
        ["largest", f"{p['largest_component_pct']:.1f}%"],
        ["isolated", f"{p['n_isolated']:,}"],
        ["median degree", f"{pct_of(p['degrees'], .5):.0f}"],
        ["max degree", f"{p['degrees'][-1] if p['degrees'] else 0}"],
    ]
    if p["weight_key"]:
        ws = sorted(p["weights"])
        if ws:
            shape.append([f"weight p50", f"{pct_of(ws, .5):.3f}"])
            shape.append([f"weight range", f"{min(ws):.2f}–{max(ws):.2f}"])
    else:
        shape.append(["edge weight", "none"])

    notes = []
    if not p["weight_key"]:
        notes.append("No numeric edge weight found — every edge costs 1, so "
                     "weighted scoring collapses to hop counting.")
    else:
        ws = p["weights"]
        if ws and min(ws) >= 0 and max(ws) <= 1:
            notes.append("Weights sit in [0,1]. Dijkstra minimises weight — confirm "
                         "smaller means <em>more</em> related, or traversal walks the "
                         "weakest edges.")
    if len(p["components"]) > 1:
        notes.append(f"{len(p['components'])} components. Anything outside the largest "
                     f"is permanently unreachable, not a bridgeable gap.")
    if p["n_isolated"]:
        notes.append(f"{p['n_isolated']} isolated node(s) can never be matched or bridged.")
    if p["word_count_mean"] > 3:
        notes.append(f"Canonical names average {p['word_count_mean']:.1f} words. A "
                     "similarity threshold tuned on short skill strings will not "
                     "transfer — calibrate it against this graph.")
    if p["n_duplicate_groups"]:
        notes.append(f"{p['n_duplicate_groups']} name group(s) differ only by case or "
                     "punctuation — dedupe before any dynamic MERGE.")
    if p["divergence"] and p["divergence"]["pct"] > 5:
        notes.append(f"On {p['divergence']['pct']:.0f}% of sampled pairs the cheapest "
                     "path is longer in hops than the shortest one. Hops and weighted "
                     "distance need separate traversals.")

    hop_warn = ""
    h2 = p["hops"].get(2, 0)
    if h2 > 60:
        hop_warn = ("At 2 hops most of the graph is reachable, so &ldquo;bridgeable&rdquo; "
                    "stops discriminating. Gate on path cost, not hop count alone.")
    elif h2 and h2 < 5:
        hop_warn = "Two-hop bridging will fire rarely on a graph this sparse."

    return {
        "shape": shape,
        "hops": [[h, v] for h, v in p["hops"].items()],
        "hopWarn": hop_warn,
        "components": p["components"][:20],
        "hubs": [[str(n), d] for n, d in p["hubs"][:10]],
        "notes": notes,
        "mode": mode_desc,
    }


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--graph", default="data/skill_graph.pkl", type=Path)
    ap.add_argument("--out", default="reports/graph_report.html", type=Path)
    ap.add_argument("--mode", choices=["hubs", "around", "random"], default="hubs")
    ap.add_argument("--around", default=None, help="seed node for --mode around")
    ap.add_argument("--hops", type=int, default=2)
    ap.add_argument("--max-nodes", type=int, default=220)
    ap.add_argument("--sample", type=int, default=60,
                    help="source nodes sampled for reachability stats")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--no-html", action="store_true")
    args = ap.parse_args()

    if not args.graph.exists():
        sys.exit(f"Graph not found: {args.graph}")

    print(f"\nloading {args.graph} ...")
    G = load_graph(args.graph)

    p = profile(G, sample=args.sample, seed=args.seed)
    report(p)

    if args.no_html:
        return

    SG, seed_node = pick_subgraph(G, args.mode, args.around, args.hops,
                                  args.max_nodes, args.seed, p["label_key"])
    payload = serialise(SG, G, p["label_key"], p["weight_key"], p["type_key"], seed_node)

    if args.mode == "around":
        mode_desc = f"{args.hops}-hop neighbourhood of {str(seed_node)[:40]}"
    elif args.mode == "random":
        mode_desc = f"random sample (seed {args.seed})"
    else:
        mode_desc = f"top {len(payload['nodes'])} by degree"

    meta = build_meta(p, mode_desc)
    html = build_html(payload, meta, args.graph.stem,
                      f"{p['n_nodes']:,} nodes / {p['n_edges']:,} edges",
                      f"{args.graph.stem} — graph audit")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html, encoding="utf-8")
    print(f"  wrote {args.out}  ({len(payload['nodes'])} nodes rendered, {mode_desc})")
    print(f"  open it:  file://{args.out.resolve()}\n")


if __name__ == "__main__":
    main()