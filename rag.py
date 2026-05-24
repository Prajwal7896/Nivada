import pickle
import numpy as np
import torch
from collections import defaultdict

from sentence_transformers import SentenceTransformer, util, CrossEncoder
from rank_bm25 import BM25Okapi


# =========================
# LOAD DATA
# =========================
with open("complaints_store.pkl", "rb") as f:
    data = pickle.load(f)

texts = data["texts"]
solutions = data["solutions"]
categories = data["categories"]


# =========================
# MODELS
# =========================
embedder = SentenceTransformer("all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


# =========================
# MEMORY SYSTEM
# =========================
memory = defaultdict(lambda: {"good": 0, "bad": 0})


# =========================
# INDEXES
# =========================
text_embeddings = embedder.encode(
    texts,
    convert_to_tensor=True,
    normalize_embeddings=True
)

bm25 = BM25Okapi([t.lower().split() for t in texts])


# =========================
# 🧠 PLANNER AGENT
# =========================
def planner(query):
    q = query.lower()

    plan = {
        "use_bm25": False,
        "use_semantic": True,
        "need_rerank": True,
        "depth": "normal"
    }

    if len(q.split()) < 4:
        plan["depth"] = "deep"

    if any(w in q for w in ["exact", "code", "error", "log"]):
        plan["use_bm25"] = True

    if "urgent" in q:
        plan["depth"] = "critical"

    return plan


# =========================
# 🚦 ROUTER AGENT
# =========================
def router(plan):
    if plan["depth"] == "critical":
        return 30
    if plan["depth"] == "deep":
        return 20
    return 10


# =========================
# 🔍 RETRIEVER AGENT
# =========================
def retrieve(query, top_k=10, use_bm25=True):

    candidates = set()

    if use_bm25:
        tokens = query.lower().split()
        bm25_scores = bm25.get_scores(tokens)
        bm25_top = np.argsort(bm25_scores)[-top_k:]
        candidates.update(bm25_top.tolist())

    emb = embedder.encode(query, convert_to_tensor=True, normalize_embeddings=True)
    sim = util.cos_sim(emb, text_embeddings)[0]

    sem_top = torch.topk(sim, k=top_k * 2).indices.cpu().numpy()
    candidates.update(sem_top.tolist())

    return list(candidates)


# =========================
# 🎯 RERANKER AGENT
# =========================
def rerank(query, candidates):
    pairs = [(query, texts[i]) for i in candidates]
    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(candidates, scores),
        key=lambda x: x[1],
        reverse=True
    )

    return ranked


# =========================
# 🧪 EVIDENCE VALIDATOR
# =========================
def validate(score):
    if score > 0.75:
        return "strong"
    if score > 0.45:
        return "medium"
    return "weak"


# =========================
# 🪞 REFLECTION AGENT
# =========================
def reflect(results):
    if not results:
        return False

    best_score = results[0]["score"]

    # self-check logic
    if best_score < 0.55:
        return False

    # diversity check
    categories = set(r["category"] for r in results)
    if len(categories) == 1 and best_score < 0.7:
        return False

    return True


# =========================
# 💾 MEMORY UPDATER
# =========================
def update_memory(query, idx, good=True):
    if good:
        memory[idx]["good"] += 1
    else:
        memory[idx]["bad"] += 1


# =========================
# 🧠 MAIN AUTONOMOUS AGENT PIPELINE
# =========================
def generate_rag_response(query):

    plan = planner(query)
    top_k = router(plan)

    candidates = retrieve(
        query,
        top_k=top_k,
        use_bm25=plan["use_bm25"]
    )

    ranked = rerank(query, candidates)

    results = []

    for idx, score in ranked[:5]:

        boost = memory[idx]["good"] * 0.03
        penalty = memory[idx]["bad"] * 0.03

        final_score = float(score) + boost - penalty

        results.append({
            "complaint": texts[idx],
            "solution": solutions[idx],
            "category": categories[idx],
            "score": round(final_score, 4),
            "confidence": validate(final_score)
        })

    # 🪞 reflection step
    if not reflect(results):
        return {
            "final_solution": "I need more information to provide a reliable resolution.",
            "confidence": 0.0,
            "similar_cases": results
        }

    best = results[0]

    return {
        "final_solution": best["solution"],
        "confidence": best["score"],
        "confidence_level": best["confidence"],
        "similar_cases": results
    }


# =========================
# OPTIONAL FEEDBACK API
# =========================
def feedback(idx, is_helpful: bool):
    update_memory(None, idx, is_helpful)