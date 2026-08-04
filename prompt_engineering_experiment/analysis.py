import json

def analyze_results():
    """
    Loads benchmark logs and computes key evaluation metrics per prompting style.
    """
    try:
        with open("results.json", "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print("❌ Error: results.json not found. Run experiment.py first.")
        return

    styles = ["Zero-Shot", "Few-Shot", "Chain-of-Thought"]
    
    # Initialize metrics structure
    stats = {
        style: {
            "exact_matches": 0, 
            "cat_matches": 0, 
            "urg_matches": 0, 
            "valid_json": 0, 
            "total_time": 0.0
        } for style in styles
    }
    total_cases = len(data)

    for item in data:
        for style in styles:
            res = item["styles"][style]
            if res["exact_match"]:
                stats[style]["exact_matches"] += 1
            if res["category_correct"]:
                stats[style]["cat_matches"] += 1
            if res["urgency_correct"]:
                stats[style]["urg_matches"] += 1
            if res["is_valid_json"]:
                stats[style]["valid_json"] += 1
            stats[style]["total_time"] += res["latency_seconds"]

    print("\n" + "="*75)
    print("📊 Prompt Engineering Strategy Benchmark Summary (Day 2)")
    print("="*75)
    print(f"{'Metric / Style':<25} | {'Zero-Shot':<12} | {'Few-Shot':<12} | {'Chain-of-Thought':<15}")
    print("-" * 75)

    # Format and display metric output rows
    for style in styles:
        s = stats[style]
        exact_acc = (s["exact_matches"] / total_cases) * 100
        cat_acc = (s["cat_matches"] / total_cases) * 100
        urg_acc = (s["urg_matches"] / total_cases) * 100
        json_valid = (s["valid_json"] / total_cases) * 100
        avg_time = s["total_time"] / total_cases

        s["exact_acc_str"] = f"{exact_acc:.0f}%"
        s["cat_acc_str"] = f"{cat_acc:.0f}%"
        s["urg_acc_str"] = f"{urg_acc:.0f}%"
        s["json_str"] = f"{json_valid:.0f}%"
        s["avg_time_str"] = f"{avg_time:.2f}s"

    print(f"{'Overall Exact Accuracy':<25} | {stats['Zero-Shot']['exact_acc_str']:<12} | {stats['Few-Shot']['exact_acc_str']:<12} | {stats['Chain-of-Thought']['exact_acc_str']:<15}")
    print(f"{'Category Accuracy':<25} | {stats['Zero-Shot']['cat_acc_str']:<12} | {stats['Few-Shot']['cat_acc_str']:<12} | {stats['Chain-of-Thought']['cat_acc_str']:<15}")
    print(f"{'Urgency Accuracy':<25} | {stats['Zero-Shot']['urg_acc_str']:<12} | {stats['Few-Shot']['urg_acc_str']:<12} | {stats['Chain-of-Thought']['urg_acc_str']:<15}")
    print(f"{'JSON Formatting Success':<25} | {stats['Zero-Shot']['json_str']:<12} | {stats['Few-Shot']['json_str']:<12} | {stats['Chain-of-Thought']['json_str']:<15}")
    print(f"{'Avg Latency (Seconds)':<25} | {stats['Zero-Shot']['avg_time_str']:<12} | {stats['Few-Shot']['avg_time_str']:<12} | {stats['Chain-of-Thought']['avg_time_str']:<15}")
    print("="*75 + "\n")

if __name__ == "__main__":
    analyze_results()