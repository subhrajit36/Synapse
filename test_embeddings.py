from synapse.graph.build_graph import build_skill_graph, add_semantic_edges, add_seed_edges

G = add_seed_edges(add_semantic_edges(build_skill_graph()))
skills = [n for n, d in G.nodes(data=True) if d['node_type'] == 'skill']
print(f'Total skills: {len(skills)}')
has_emb = sum(1 for s in skills if 'embedding' in G.nodes[s])
print(f'Skills with embeddings: {has_emb}')
# Check a few
for s in ['Docker', 'Kubernetes', 'PyTorch', 'FastAPI']:
    if s in G.nodes and 'embedding' in G.nodes[s]:
        emb = G.nodes[s]['embedding']
        shape = emb.shape if hasattr(emb, 'shape') else len(emb)
        print(f'  {s}: embedding shape {shape}')