import json

cache = {}
with open('data/eval/typed_edge_cache.jsonl', 'r') as f:
    for line in f:
        data = json.loads(line)
        key = tuple(sorted([data['a'], data['b']]))
        cache[key] = data

probes = [
    ('TensorFlow', 'Keras'),
    ('Docker', 'Kubernetes'),
    ('Apache Spark', 'Apache Hadoop'),
]

for a, b in probes:
    key = tuple(sorted([a, b]))
    if key in cache:
        d = cache[key]
        print(f'{a} <-> {b}:')
        print(f'  edge_type: {d["edge_type"]}')
        print(f'  direction: {d["direction"]}')
        print(f'  confidence: {d["confidence"]:.2f}')
        print(f'  rationale: {d["rationale"]}')
        print()