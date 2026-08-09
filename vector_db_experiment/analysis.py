import json


def analyze_results():
    """
    Loads benchmark timing metrics and formats comparative evaluation table.
    """
    try:
        with open("vector_db_results.json", "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print("❌ Error: vector_db_results.json not found. Run experiment.py first.")
        return

    mem = data["in_memory"]
    chroma = data["chromadb"]

    print("\n" + "="*75)
    print("📊 Day 5: Vector Search Benchmark (In-Memory vs ChromaDB)")
    print("="*75)
    print(f"{'Engine':<20} | {'Indexing Time':<18} | {'Total Query Time':<18} | {'Avg/Query (ms)':<12}")
    print("-" * 75)
    print(f"{'Flat (NumPy/Cosine)':<20} | {mem['indexing_time_seconds']:.4f}s{'':<11} | {mem['query_time_seconds']:.4f}s{'':<11} | {mem['avg_query_ms']:.2f}ms")
    print(f"{'ChromaDB (HNSW)':<20} | {chroma['indexing_time_seconds']:.4f}s{'':<11} | {chroma['query_time_seconds']:.4f}s{'':<11} | {chroma['avg_query_ms']:.2f}ms")
    print("="*75)
    
    # Calculate difference
    if chroma['query_time_seconds'] > 0:
        ratio = mem['query_time_seconds'] / chroma['query_time_seconds']
        print(f"💡 Query Efficiency Ratio: ChromaDB is {ratio:.2f}x relative to Flat Exact Search")
    print("="*75 + "\n")


if __name__ == "__main__":
    analyze_results()