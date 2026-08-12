import time
import json
import chromadb
from rank_bm25 import BM25Okapi
from config import DOCUMENTS, TEST_QUERY, RRF_K


def bm25_sparse_search(documents, query, top_k=3):
    tokenized_docs = [doc.lower().split() for doc in documents]
    bm25 = BM25Okapi(tokenized_docs)
    
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return [{"doc_idx": idx, "score": float(scores[idx]), "rank": r + 1} for r, idx in enumerate(ranked_indices[:top_k])]


def dense_vector_search(documents, query, top_k=3):
    client = chromadb.Client()
    collection = client.create_collection(name=f"hybrid_demo_{int(time.time()*1000)}")
    ids = [f"doc_{i}" for i in range(len(documents))]
    
    collection.add(ids=ids, documents=documents)
    results = collection.query(query_texts=[query], n_results=top_k)
    
    retrieved_ids = results["ids"][0] if results["ids"] else []
    ranked_results = []
    for r, doc_id in enumerate(retrieved_ids):
        idx = int(doc_id.split("_")[1])
        ranked_results.append({"doc_idx": idx, "rank": r + 1})
        
    return ranked_results


def reciprocal_rank_fusion(dense_ranks, sparse_ranks, k=RRF_K):
    rrf_scores = {}

    for item in dense_ranks:
        idx = item["doc_idx"]
        rank = item["rank"]
        rrf_scores[idx] = rrf_scores.get(idx, 0.0) + (1.0 / (k + rank))

    for item in sparse_ranks:
        idx = item["doc_idx"]
        rank = item["rank"]
        rrf_scores[idx] = rrf_scores.get(idx, 0.0) + (1.0 / (k + rank))

    sorted_fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return [{"doc_idx": idx, "rrf_score": round(score, 6), "fused_rank": r + 1} for r, (idx, score) in enumerate(sorted_fused)]


def main():
    print("🔀 Running Hybrid Search Benchmark (BM25 + ChromaDB Dense)...")

    t0_bm25 = time.time()
    sparse_res = bm25_sparse_search(DOCUMENTS, TEST_QUERY)
    t_bm25 = time.time() - t0_bm25

    t0_dense = time.time()
    dense_res = dense_vector_search(DOCUMENTS, TEST_QUERY)
    t_dense = time.time() - t0_dense

    t0_hybrid = time.time()
    hybrid_res = reciprocal_rank_fusion(dense_res, sparse_res)
    t_hybrid = time.time() - t0_hybrid

    payload = {
        "query": TEST_QUERY,
        "documents": DOCUMENTS,
        "sparse_bm25_top": [
            {"doc": DOCUMENTS[item["doc_idx"]], "rank": item["rank"]} for item in sparse_res
        ],
        "dense_chroma_top": [
            {"doc": DOCUMENTS[item["doc_idx"]], "rank": item["rank"]} for item in dense_res
        ],
        "hybrid_rrf_top": [
            {"doc": DOCUMENTS[item["doc_idx"]], "rrf_score": item["rrf_score"], "rank": item["fused_rank"]} for item in hybrid_res
        ],
        "latencies": {
            "bm25_sec": round(t_bm25, 5),
            "dense_sec": round(t_dense, 5),
            "hybrid_fusion_sec": round(t_hybrid, 5)
        }
    }

    with open("hybrid_results.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print("✅ Hybrid Search Benchmark complete. Output saved to hybrid_results.json")


if __name__ == "__main__":
    main()