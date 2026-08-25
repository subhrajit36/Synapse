from synapse.graph.build_graph import build_skill_graph, add_semantic_edges, add_seed_edges
from synapse.matching.entity_linker import EntityLinker

G = add_seed_edges(add_semantic_edges(build_skill_graph()))
skills = [n for n, d in G.nodes(data=True) if d['node_type'] == 'skill']
node_texts = {n: f'{n} ({G.nodes[n].get("category", "")})' for n in skills}
linker = EntityLinker(skills, node_texts=node_texts, model_name='BAAI/bge-small-en-v1.5')

result = linker.link_many(['Docker', 'Kubernetes', 'aws', 'PyTorch', 'react'], source_id='test')
print(f'Resolved: {result.nodes}')
print(f'Unresolved: {[r.surface for r in result.unresolved]}')
print(f'Resolution rate: {result.resolution_rate:.2%}')