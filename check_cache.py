import json

cache = {}
with open('data/eval/typed_edge_cache.jsonl', 'r') as f:
    for line in f:
        data = json.loads(line)
        key = tuple(sorted([data['a'], data['b']]))
        cache[key] = data

probes = [
    ('Docker', 'Kubernetes'),
    ('PyTorch', 'TensorFlow'),
    ('TensorFlow', 'Keras'),
    ('XGBoost', 'LightGBM'),
    ('GitHub', 'GitLab'),
    ('MySQL', 'PostgreSQL'),
    ('Apache Spark', 'Apache Hadoop'),
]

for a, b in probes:
    key = tuple(sorted([a, b]))
    if key in cache:
        d = cache[key]
        print(f'{a} <-> {b}: {d["edge_type"]} (conf={d["confidence"]:.2f}) dir={d["direction"]}')
    else:
        print(f'{a} <-> {b}: NOT IN CACHE')