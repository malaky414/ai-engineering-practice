import time
import json
import chromadb
from sentence_transformers import CrossEncoder

from config import DOCUMENTS, TEST_QUERY, RERANKER_MODEL_NAME


def stage1_dense_retrieval(documents, query, top_k=4):
    client = chromadb.Client()
    collection = client.create_collection(name=f"rerank_demo_{int(time.time()*1000)}")
    ids = [f"doc_{i}" for i in range(len(documents))]
    
    collection.add(ids=ids, documents=documents)
    results = collection.query(query_texts=[query], n_results=top_k)
    
    retrieved_ids = results["ids"][0] if results["ids"] else []
    retrieved_docs = results["documents"][0] if results["documents"] else []
    
    stage1_candidates = []
    for r, (doc_id, doc_text) in enumerate(zip(retrieved_ids, retrieved_docs)):
        idx = int(doc_id.split("_")[1])
        stage1_candidates.append({
            "doc_idx": idx,
            "doc_text": doc_text,
            "stage1_rank": r + 1
        })
        
    return stage1_candidates


def stage2_cross_encoder_rerank(candidates, query, model_name=RERANKER_MODEL_NAME):
    reranker = CrossEncoder(model_name)
    
    pairs = [[query, item["doc_text"]] for item in candidates]
    
    t0 = time.time()
    scores = reranker.predict(pairs)
    rerank_time = time.time() - t0
    
    for item, score in zip(candidates, scores):
        item["rerank_score"] = float(score)
        
    reranked_results = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
    
    for r, item in enumerate(reranked_results):
        item["stage2_rank"] = r + 1
        
    return reranked_results, rerank_time


def main():
    print("🎯 Running Two-Stage Retrieval Experiment (ChromaDB + Cross-Encoder Reranker)...")

    t0_stage1 = time.time()
    stage1_candidates = stage1_dense_retrieval(DOCUMENTS, TEST_QUERY, top_k=4)
    t_stage1 = time.time() - t0_stage1

    reranked_candidates, t_stage2 = stage2_cross_encoder_rerank(stage1_candidates, TEST_QUERY)

    payload = {
        "query": TEST_QUERY,
        "stage1_vector_only": [
            {"doc": item["doc_text"], "rank": item["stage1_rank"]} for item in stage1_candidates
        ],
        "stage2_reranked": [
            {
                "doc": item["doc_text"], 
                "stage1_rank": item["stage1_rank"],
                "stage2_rank": item["stage2_rank"], 
                "score": round(item["rerank_score"], 4)
            } for item in reranked_candidates
        ],
        "latencies": {
            "stage1_retrieval_sec": round(t_stage1, 5),
            "stage2_rerank_sec": round(t_stage2, 5)
        }
    }

    with open("reranker_results.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print("✅ Reranker Benchmark complete. Output saved to reranker_results.json")


if __name__ == "__main__":
    main()