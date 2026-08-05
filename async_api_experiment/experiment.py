# experiment.py
import time
import json
import asyncio
import httpx
from config import BASE_URL, CITIES, DEFAULT_PARAMS


def fetch_sync(client, city):
    """
    Executes a single synchronous HTTP GET request.
    """
    params = {**DEFAULT_PARAMS, "latitude": city["lat"], "longitude": city["lon"]}
    response = client.get(BASE_URL, params=params)
    return response.json()


def run_sequential_benchmark():
    """
    Fires 10 HTTP requests sequentially (one after another).
    """
    print("⏳ Running Sequential (Sync) Requests...")
    start_time = time.time()
    results = []
    
    with httpx.Client(timeout=10.0) as client:
        for city in CITIES:
            data = fetch_sync(client, city)
            results.append({"city": city["name"], "temp": data.get("current_weather", {}).get("temperature")})
            
    total_time = time.time() - start_time
    return results, total_time


async def fetch_async(client, city):
    """
    Executes a single asynchronous HTTP GET request.
    """
    params = {**DEFAULT_PARAMS, "latitude": city["lat"], "longitude": city["lon"]}
    response = await client.get(BASE_URL, params=params)
    return response.json()


async def run_parallel_benchmark():
    """
    Fires 10 HTTP requests concurrently in parallel using asyncio.gather.
    """
    print("⚡ Running Parallel (Async) Requests...")
    start_time = time.time()
    
    async with httpx.AsyncClient(timeout=10.0) as client:
        tasks = [fetch_async(client, city) for city in CITIES]
        responses = await asyncio.gather(*tasks)
        
    results = [
        {"city": CITIES[i]["name"], "temp": res.get("current_weather", {}).get("temperature")}
        for i, res in enumerate(responses)
    ]
    
    total_time = time.time() - start_time
    return results, total_time


def main():
    # 1. Sequential Execution
    sync_results, sync_duration = run_sequential_benchmark()

    # 2. Parallel Async Execution
    async_results, async_duration = asyncio.run(run_parallel_benchmark())

    # Save benchmark payload to disk
    output_payload = {
        "sync_duration_seconds": round(sync_duration, 4),
        "async_duration_seconds": round(async_duration, 4),
        "total_requests": len(CITIES),
        "sync_sample_results": sync_results,
        "async_sample_results": async_results
    }

    with open("async_results.json", "w", encoding="utf-8") as f:
        json.dump(output_payload, f, ensure_ascii=False, indent=2)

    print("\n✅ Async Benchmark completed successfully. Saved to async_results.json")


if __name__ == "__main__":
    main()