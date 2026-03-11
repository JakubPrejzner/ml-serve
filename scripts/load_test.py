"""
Stepped load test for ml-serve — ramps concurrency and reports per-level stats.

Usage:
    python scripts/load_test.py
    python scripts/load_test.py --url http://localhost:8000 --duration 10
"""

import argparse
import asyncio
import statistics
import time

import httpx

PAYLOAD = {"text": "The quality is surprisingly good for the price."}
CONCURRENCY_LEVELS = [1, 5, 10, 20]


async def worker(
    client: httpx.AsyncClient,
    url: str,
    stop_event: asyncio.Event,
    results: list[float],
    errors: list[int],
) -> None:
    while not stop_event.is_set():
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


async def run_level(
    base_url: str, concurrency: int, duration: int
) -> dict[str, float | int]:
    url = f"{base_url}/v1/predict"
    results: list[float] = []
    errors: list[int] = []
    stop = asyncio.Event()

    async with httpx.AsyncClient(timeout=30.0) as client:
        workers = [
            asyncio.create_task(worker(client, url, stop, results, errors))
            for _ in range(concurrency)
        ]

        await asyncio.sleep(duration)
        stop.set()

        # give workers a moment to wrap up
        await asyncio.gather(*workers, return_exceptions=True)

    total_ok = len(results)
    total_err = len(errors)
    total = total_ok + total_err
    throughput = total_ok / duration if duration else 0

    if results:
        results.sort()
        avg = statistics.mean(results)
        p99 = results[int(min(len(results) * 0.99, len(results) - 1))]
    else:
        avg = 0.0
        p99 = 0.0

    error_rate = (total_err / total * 100) if total else 0.0

    return {
        "concurrency": concurrency,
        "requests": total,
        "throughput": round(throughput, 2),
        "avg_ms": round(avg, 2),
        "p99_ms": round(p99, 2),
        "error_pct": round(error_rate, 2),
    }


async def run_load_test(base_url: str, duration: int) -> None:
    print(f"Load test: {base_url}/v1/predict")
    print(f"  Duration per level: {duration}s")
    print(f"  Concurrency levels: {CONCURRENCY_LEVELS}")
    print()

    summaries: list[dict[str, float | int]] = []

    for level in CONCURRENCY_LEVELS:
        print(f"  Running with {level} concurrent users ...", end=" ", flush=True)
        result = await run_level(base_url, level, duration)
        summaries.append(result)
        print(f"{result['requests']} reqs, {result['throughput']} req/s")

    # summary table
    print()
    header = f"  {'Concurrency':>12} {'Requests':>10} {'Throughput':>12} {'Avg (ms)':>10} {'p99 (ms)':>10} {'Errors':>8}"
    print(header)
    print(f"  {'─' * 12} {'─' * 10} {'─' * 12} {'─' * 10} {'─' * 10} {'─' * 8}")

    for s in summaries:
        print(
            f"  {s['concurrency']:>12} {s['requests']:>10} "
            f"{s['throughput']:>10} /s {s['avg_ms']:>10} {s['p99_ms']:>10} "
            f"{s['error_pct']:>7}%"
        )

    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="ml-serve load test")
    parser.add_argument("--url", default="http://localhost:8000", help="Base URL")
    parser.add_argument("--duration", type=int, default=10, help="Seconds per concurrency level")
    args = parser.parse_args()

    asyncio.run(run_load_test(args.url, args.duration))


if __name__ == "__main__":
    main()
