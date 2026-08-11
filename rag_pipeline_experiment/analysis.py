import json


def analyze_rag():
    try:
        with open("rag_results.json", "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print("❌ Error: rag_results.json not found. Run experiment.py first.")
        return

    m = data["metrics"]

    print("\n" + "="*80)
    print("📊 Day 7: End-to-End RAG Pipeline Evaluation Summary")
    print("="*80)
    print(f"🎯 Query: '{data['query']}'")
    print("-" * 80)
    print(f"⏱️ Retrieval Latency:  {m['retrieval_latency_sec']:.4f}s")
    print(f"⏱️ Direct Generation:  {m['direct_gen_latency_sec']:.4f}s")
    print(f"⏱️ RAG Generation:     {m['rag_gen_latency_sec']:.4f}s")
    print("="*80)
    print("\n🤖 [Direct Answer (No RAG / Hallucination Risk)]:")
    print(f"   {data['direct_answer'].strip()}")
    print("\n🎯 [RAG Answer (Context-Grounded & Accurate)]:")
    print(f"   {data['rag_answer'].strip()}")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    analyze_rag()