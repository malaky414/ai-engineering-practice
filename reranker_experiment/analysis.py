import json


def analyze_reranker():
    try:
        with open("reranker_results.json", "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print("❌ Error: reranker_results.json not found. Run experiment.py first.")
        return

    print("\n" + "="*85)
    print("📊 Day 9: Two-Stage Retrieval & Reranking Evaluation")
    print("="*85)
    print(f"🎯 Query: '{data['query']}'")
    print("-" * 85)

    print("\n🔮 Stage 1: Top Document from Dense Vector Search (ChromaDB):")
    print(f"   Rank 1: {data['stage1_vector_only'][0]['doc']}")

    print("\n🏆 Stage 2: Top Document after Cross-Encoder Reranking:")
    top_stage2 = data['stage2_reranked'][0]
    print(f"   Rank 1: {top_stage2['doc']}")
    print(f"   (Cross-Encoder Score: {top_stage2['score']} | Was Rank {top_stage2['stage1_rank']} in Stage 1)")

    print("\n" + "="*85)
    print("📋 Complete Reranked Candidate List:")
    print("-" * 85)
    for item in data['stage2_reranked']:
        print(f"   [Rank {item['stage2_rank']}] (Score: {item['score']:>7.4f}) (Stage 1 Rank was: {item['stage1_rank']}) -> {item['doc'][:70]}...")

    print("\n" + "="*85)
    print("⏱️ Latencies Breakdown:")
    print(f"   Stage 1 (Vector Search): {data['latencies']['stage1_retrieval_sec']:.5f}s | Stage 2 (Cross-Encoder): {data['latencies']['stage2_rerank_sec']:.5f}s")
    print("="*85 + "\n")


if __name__ == "__main__":
    analyze_reranker()