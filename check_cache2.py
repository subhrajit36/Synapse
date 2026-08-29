import json

cache = {}
with open('data/eval/typed_edge_cache.jsonl', 'r') as f:
    for line in f:
        data = json.loads(line)
        key = tuple(sorted([data['a'], data['b']]))
        cache[key] = data

print(f"Total cached pairs: {len(cache)}")

# Check all substitution group pairs
from synapse.eval.dataset import SUBSTITUTION_GROUPS

for group in SUBSTITUTION_GROUPS:
    for i in range(len(group)):
        for j in range(i + 1, len(group)):
            a, b = sorted([group[i], group[j]])
            key = (a, b)
            if key in cache:
                d = cache[key]
                print(f'{a} <-> {b}: {d["edge_type"]} (conf={d["confidence"]:.2f}) dir={d["direction"]}')
            else:
                print(f'{a} <-> {b}: NOT IN CACHE')