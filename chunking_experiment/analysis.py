import json


def analyze_chunking():
    try:
        with open("chunking_results.json", "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print("❌ Error: chunking_results.json not found. Run experiment.py first.")
        return

    results = data["results"]

    print("\n" + "="*85)
    print("📊 Day 6: Chunking Strategies & Retrieval Benchmark Summary")
    print(f"🎯 Test Query: '{data['test_query']}'")
    print("="*85)
    print(f"{'Strategy':<22} | {'Chunks':<7} | {'Avg Len':<9} | {'Index Time':<12} | {'Top Similarity':<15}")
    print("-" * 85)

    for strat, m in results.items():
        print(f"{strat:<22} | {m['num_chunks']:<7} | {m['avg_chunk_length']:<9} | {m['indexing_time_seconds']:.4f}s{'':<5} | {m['top_similarity_score']:.4f}")

    print("="*85 + "\n")


if __name__ == "__main__":
    analyze_chunking()