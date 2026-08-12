import json


def analyze_hybrid():
    try:
        with open("hybrid_results.json", "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print("❌ Error: hybrid_results.json not found. Run experiment.py first.")
        return

    print("\n" + "="*85)
    print("📊 Day 8: Hybrid Search Evaluation (BM25 vs Dense vs RRF Hybrid)")
    print("="*85)
    print(f"🎯 Query: '{data['query']}'")
    print("-" * 85)

    print("\n🔍 1. Sparse Search (BM25 Keywords - Top Document):")
    print(f"   Rank 1: {data['sparse_bm25_top'][0]['doc']}")

    print("\n🔮 2. Dense Search (ChromaDB Embeddings - Top Document):")
    print(f"   Rank 1: {data['dense_chroma_top'][0]['doc']}")

    print("\n🏆 3. Hybrid Search (RRF Combined Rank - Top Document):")
    print(f"   Rank 1: {data['hybrid_rrf_top'][0]['doc']} (RRF Score: {data['hybrid_rrf_top'][0]['rrf_score']})")

    print("\n" + "="*85)
    print("⏱️ Latencies Breakdown:")
    print(f"   BM25: {data['latencies']['bm25_sec']:.5f}s | Dense Vector: {data['latencies']['dense_sec']:.5f}s | RRF Fusion: {data['latencies']['hybrid_fusion_sec']:.5f}s")
    print("="*85 + "\n")


if __name__ == "__main__":
    analyze_hybrid()