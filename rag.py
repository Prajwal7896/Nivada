import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

# ✅ LOAD STORED DATA
with open("complaints_store.pkl", "rb") as f:
    data = pickle.load(f)

texts = data["texts"]
solutions = data["solutions"]

# ✅ CREATE EMBEDDINGS (only once)
text_embeddings = model.encode(texts, convert_to_tensor=True)


def generate_rag_response(query):
    # 🔍 Encode query
    query_emb = model.encode([query], convert_to_tensor=True)

    # 📊 Compute similarity
    similarities = (text_embeddings @ query_emb.T).cpu().numpy().flatten()

    # 🔝 Top 3 similar complaints
    top_indices = similarities.argsort()[-3:][::-1]

    similar_cases = []
    for idx in top_indices:
        similar_cases.append({
            "complaint": texts[idx],
            "solution": solutions[idx]
        })

    # 🎯 Best solution
    final_solution = similar_cases[0]["solution"]

    return {
        "final_solution": final_solution,
        "similar_cases": similar_cases
    }