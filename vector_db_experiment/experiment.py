import time
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import chromadb

from config import NUM_DOCUMENTS, EMBEDDING_DIM, NUM_QUERIES, TOP_K, COLLECTION_NAME


def generate_mock_embeddings(num_items, dim):
    np.random.seed(42)
    vectors = np.random.randn(num_items, dim).astype(np.float32)
    # L2 Normalization so dot product equals cosine similarity
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / norms


def benchmark_in_memory(docs_vecs, query_vecs):
    print("⏳ Running In-Memory (Cosine Similarity) Benchmark...")
    start_index = time.time()
    stored_vectors = np.copy(docs_vecs)
    indexing_time = time.time() - start_index
    start_query = time.time()
    for q_vec in query_vecs:
        sim_scores = cosine_similarity(q_vec.reshape(1, -1), stored_vectors)[0]
        top_k_indices = np.argsort(sim_scores)[::-1][:TOP_K]
    
    total_query_time = time.time() - start_query

    return indexing_time, total_query_time


def benchmark_chroma_db(docs_vecs, query_vecs):
    """
    Benchmarks ChromaDB Vector Indexing and Nearest Neighbor Querying.
    """
    print("⚡ Running ChromaDB Benchmark...")
    
    # Initialize Ephemeral (In-Memory) Chroma Client
    client = chromadb.Client()
    collection = client.create_collection(name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})

    # Indexing Phase
    doc_ids = [f"doc_{i}" for i in range(len(docs_vecs))]
    embeddings_list = docs_vecs.tolist()

    start_index = time.time()
    collection.add(
        ids=doc_ids,
        embeddings=embeddings_list
    )
    indexing_time = time.time() - start_index

    # Query Phase
    start_query = time.time()
    for q_vec in query_vecs:
        collection.query(
            query_embeddings=[q_vec.tolist()],
            n_results=TOP_K
        )
    total_query_time = time.time() - start_query

    return indexing_time, total_query_time


def main():
    print(f"📦 Generating {NUM_DOCUMENTS} mock embeddings of dimension {EMBEDDING_DIM}...")
    docs_embeddings = generate_mock_embeddings(NUM_DOCUMENTS, EMBEDDING_DIM)
    query_embeddings = generate_mock_embeddings(NUM_QUERIES, EMBEDDING_DIM)

    # 1. Benchmark Flat In-Memory Search
    mem_index_t, mem_query_t = benchmark_in_memory(docs_embeddings, query_embeddings)

    # 2. Benchmark ChromaDB Search
    chroma_index_t, chroma_query_t = benchmark_chroma_db(docs_embeddings, query_embeddings)
    output_payload = {
        "num_documents": NUM_DOCUMENTS,
        "embedding_dim": EMBEDDING_DIM,
        "num_queries": NUM_QUERIES,
        "top_k": TOP_K,
        "in_memory": {
            "indexing_time_seconds": round(mem_index_t, 5),
            "query_time_seconds": round(mem_query_t, 5),
            "avg_query_ms": round((mem_query_t / NUM_QUERIES) * 1000, 3)
        },
        "chromadb": {
            "indexing_time_seconds": round(chroma_index_t, 5),
            "query_time_seconds": round(chroma_query_t, 5),
            "avg_query_ms": round((chroma_query_t / NUM_QUERIES) * 1000, 3)
        }
    }

    with open("vector_db_results.json", "w", encoding="utf-8") as f:
        json.dump(output_payload, f, ensure_ascii=False, indent=2)

    print("\n✅ Vector DB Benchmark completed successfully. Saved to vector_db_results.json")


if __name__ == "__main__":
    main()