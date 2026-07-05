from sentence_transformers import SentenceTransformer, util

# A small, fast, battle-tested embedding model. Downloads once (~90MB), then cached.
model = SentenceTransformer("all-MiniLM-L6-v2")

skills = ["Docker", "Kubernetes", "PyTorch", "TensorFlow", "React", "Adobe Photoshop"]

# encode() turns each skill string into a vector (list of numbers) capturing meaning.
vectors = model.encode(skills)
print("Each skill is now a vector of length:", len(vectors[0]))

# cosine similarity: 1.0 = same direction/meaning, ~0 = unrelated.
print("\nPairwise similarity (higher = more related):")
for i in range(len(skills)):
    for j in range(i + 1, len(skills)):
        sim = util.cos_sim(vectors[i], vectors[j]).item()
        print(f"  {skills[i]:16s} <-> {skills[j]:16s} : {sim:.2f}")
