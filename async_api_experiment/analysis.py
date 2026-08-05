import json


def analyze_performance():
    try:
        with open("async_results.json", "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print("❌ Error: async_results.json not found. Run experiment.py first.")
        return

    sync_time = data["async_duration_seconds"] if "async_duration_seconds" in data else 0
    sync_time = data["sync_duration_seconds"]
    async_time = data["async_duration_seconds"]
    total_reqs = data["total_requests"]

    # Calculate Speedup Factor and Time Saved
    speedup = sync_time / async_time if async_time > 0 else 0
    time_saved = sync_time - async_time
    percentage_improvement = ((sync_time - async_time) / sync_time) * 100 if sync_time > 0 else 0

    print("\n" + "="*70)
    print("📊 Day 3: Async vs Sequential HTTP Performance Summary")
    print("="*70)
    print(f"{'Execution Mode':<25} | {'Total Time':<15} | {'Avg Time per Req':<20}")
    print("-" * 70)
    print(f"{'Sequential (Sync)':<25} | {sync_time:.3f}s{'':<9} | {(sync_time/total_reqs):.3f}s{'':<14}")
    print(f"{'Parallel (Async)':<25} | {async_time:.3f}s{'':<9} | {(async_time/total_reqs):.3f}s{'':<14}")
    print("="*70)
    print(f"🚀 Speedup Factor: {speedup:.2f}x Faster")
    print(f"⏱️ Time Saved: {time_saved:.3f} seconds ({percentage_improvement:.1f}% reduction)")
    print("="*70 + "\n")


if __name__ == "__main__":
    analyze_performance()