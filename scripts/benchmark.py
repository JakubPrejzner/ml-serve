"""
Quick async benchmark for ml-serve /v1/predict endpoint.

Usage:
    python scripts/benchmark.py
    python scripts/benchmark.py --url http://localhost:8000 --n 200 --concurrency 20
"""

import argparse
import asyncio
import statistics
import time

import httpx

PAYLOAD = {"text": "This product exceeded all my expectations, absolutely love it!"}


async def send_request(
    client: httpx.AsyncClient, url: str, results: list[float], errors: list[int]
) -> None:
    start = time.perf_counter()
    try:
        resp = await client.post(url, json=PAYLOAD)
        elapsed_ms = (time.perf_counter() - start) * 1000
        if resp.status_code == 200:
            results.append(elapsed_ms)
        else:
            errors.append(resp.status_code)
    except httpx.RequestError:
        errors.append(0)


async def run_benchmark(base_url: str, n: int, concurrency: int) -> None:
    url = f"{base_url}/v1/predict"
    results: list[float] = []
    errors: list[int] = []
    semaphore = asyncio.Semaphore(concurrency)

    async def limited_request(client: httpx.AsyncClient) -> None:
        async with semaphore:
            await send_request(client, url, results, errors)

    print(f"Benchmarking {url}")
    print(f"  requests: {n}  |  concurrency: {concurrency}")
    print("-" * 55)

    wall_start = time.perf_counter()

    async with httpx.AsyncClient(timeout=30.0) as client:
        tasks = [limited_request(client) for _ in range(n)]
        await asyncio.gather(*tasks)

    wall_elapsed = time.perf_counter() - wall_start

    if not results:
        print("All requests failed.")
        return

    results.sort()
    total_ok = len(results)
    total_err = len(errors)
    throughput = total_ok / wall_elapsed

    p50 = results[int(total_ok * 0.50)]
    p95 = results[int(min(total_ok * 0.95, total_ok - 1))]
    p99 = results[int(min(total_ok * 0.99, total_ok - 1))]
    avg = statistics.mean(results)

    print()
    print(f"  {'Metric':<22} {'Value':>12}")
    print(f"  {'─' * 22} {'─' * 12}")
    print(f"  {'Total requests':<22} {n:>12}")
    print(f"  {'Successful':<22} {total_ok:>12}")
    print(f"  {'Failed':<22} {total_err:>12}")
    print(f"  {'Avg latency (ms)':<22} {avg:>12.2f}")
    print(f"  {'p50 (ms)':<22} {p50:>12.2f}")
    print(f"  {'p95 (ms)':<22} {p95:>12.2f}")
    print(f"  {'p99 (ms)':<22} {p99:>12.2f}")
    print(f"  {'Throughput (req/s)':<22} {throughput:>12.2f}")
    print(f"  {'Wall time (s)':<22} {wall_elapsed:>12.2f}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="ml-serve benchmark")
    parser.add_argument("--url", default="http://localhost:8000", help="Base URL")
    parser.add_argument("--n", type=int, default=100, help="Number of requests")
    parser.add_argument("--concurrency", type=int, default=10, help="Concurrent requests")
    args = parser.parse_args()

    asyncio.run(run_benchmark(args.url, args.n, args.concurrency))


if __name__ == "__main__":
    main()
