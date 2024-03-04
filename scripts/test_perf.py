import asyncio
import httpx
import statistics as stats
import argparse
from datetime import datetime, timedelta
import os
import json

# Environment variable for Replicate API token
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")

# Global counters and lists
times = []
failures = 0
sstps = []  # Single Stream Tokens Per Second
start_end_times = []  # Store start and end times for each request
client = httpx.AsyncClient()
returned_requests = []
n_requests = 0
n_cog_already_running_prediction = 0


def format_request_data(n_input_tokens, n_output_tokens, target):
    prompt = " a" * n_input_tokens

    if target not in ["cog-triton", "triton"]:
        return {
            "version": target,
            "input": {
                "prompt": prompt,
                "min_length": n_output_tokens,
                "max_new_tokens": n_output_tokens,
            },
        }

    elif target == "cog-triton":
        return {
            "input": {
                "prompt": prompt,
                "max_new_tokens": n_output_tokens,
                "min_length": n_output_tokens,
            }
        }
    elif target == "triton":
        return {
            "text_input": prompt,
            "max_tokens": n_output_tokens,
            "min_length": n_output_tokens,
            "bad_words": [],
            "stop_words": [],
        }

    else:
        raise ValueError("Invalid target.")


async def poll_replicate_request(response, headers):
    prediction = response.json()
    prediction_id = prediction["id"]

    # Poll for the prediction completion
    status = ""
    while status not in ["succeeded", "failed"]:
        await asyncio.sleep(1)  # Poll every 0.25 seconds
        response = await client.get(
            f"https://api.replicate.com/v1/predictions/{prediction_id}", headers=headers
        )
        response = response.json()
        status = response["status"]
        if status == "succeeded":
            return response
        elif status == "failed":
            print(f"Prediction {prediction_id} failed")
            return response


def parse_cog_time(x):
    x = x[:26] + "Z"
    return datetime.fromisoformat(x.rstrip("Z"))


async def make_request(url, headers, data, target):
    global failures, sstps, start_end_times, returned_requests, n_requests, n_cog_already_running_prediction
    start_time = datetime.now()
    # print(f"[REQUEST STARTED]: {start_time}")  # Log start time

    try:
        # For local cog-triton or triton requests
        n_requests += 1
        response = await client.post(url, headers=headers, json=data)

        if target not in ["cog-triton", "triton"]:
            response = await poll_replicate_request(response, headers)
            completed_at = response["completed_at"]
            end_time = parse_cog_time(completed_at)
            delta = (end_time - start_time).total_seconds()

        else:
            end_time = datetime.now()
            delta = (end_time - start_time).total_seconds()

        times.append(delta)
        start_end_times.append((start_time, end_time))

        if not isinstance(response, dict) and not response.status_code == 200:
            if response.status_code == 409:
                n_cog_already_running_prediction += 1
            else:
                failures += 1

        if not isinstance(response, dict):
            prefix = "data: "
            if response.text.startswith(prefix):
                json_str = response.text[len(prefix) :]
            else:
                json_str = response.text
            response = json.loads(json_str)

        returned_requests.append(response)

        if (
            isinstance(response, dict)
            and "status" in response
            and response["status"] == "failed"
        ):
            failures += 1
            print(f"Request failed: {response['error']}")
            return

        if "text_output" in response and response["text_output"]:
            sstps.append(data["max_tokens"] / delta)
        elif "output" in response and response["output"]:
            sstps.append(data["input"]["max_new_tokens"] / delta)
        else:
            failures += 1

        # if target != "triton":
        #     completed_at = response["completed_at"]
        #     response_id = response["id"]
        #     print(
        #         f"{response_id}: start time {start_time}, end time {end_time}, delta {delta}, completed_at {completed_at}"
        #     )

    except Exception as e:
        failures += 1
        print(f"Request failed: {e}")


async def perform_requests(rate, duration, url, headers, data, mode, target):
    start_time = datetime.now()
    tasks = []
    while (datetime.now() - start_time).total_seconds() < duration:
        if mode == "batch":
            tasks = [
                asyncio.create_task(make_request(url, headers, data, target))
                for _ in range(rate)
            ]
            await asyncio.gather(*tasks)
            await asyncio.sleep(1)  # Rate limit batches per second
        elif mode == "rps":
            await asyncio.sleep(1 / rate)  # Sleep to maintain the rate
            tasks.append(asyncio.create_task(make_request(url, headers, data, target)))

    if tasks:
        await asyncio.gather(*tasks)


def calculate_concurrency(start_end_times, duration, start_time):
    """
    Calculate concurrency levels for each 5ms interval of the test duration.
    """
    # Calculate the total number of intervals in the given duration
    # Each interval is 5ms, so multiply duration by 200 to get the number of intervals in one second
    total_intervals = duration * 200
    concurrency_levels = []

    # Iterate over each 5ms interval
    for interval in range(total_intervals):
        interval_start = start_time + timedelta(milliseconds=interval * 5)
        interval_end = interval_start + timedelta(milliseconds=5)

        # Count requests that were active during this interval
        concurrency = sum(
            1
            for start, end in start_end_times
            if start < interval_end and end > interval_start
        )

        concurrency_levels.append(concurrency)

    return concurrency_levels


async def main():
    parser = argparse.ArgumentParser(
        description="Benchmark script for Triton or Cog server."
    )
    parser.add_argument(
        "--target",
        required=True,
        help="Target server for the benchmark.",
    )
    parser.add_argument(
        "--rate",
        type=float,
        required=True,
        help="Number of requests per second (for rps) or total concurrent requests (for batch).",
    )
    parser.add_argument(
        "--unit",
        type=str,
        choices=["rps", "batch"],
        required=True,
        help="Mode of operation: rps for requests per second, batch for concurrent requests.",
    )
    parser.add_argument(
        "--duration", type=int, required=True, help="Duration of test in seconds."
    )
    parser.add_argument(
        "--n_input_tokens", type=int, required=True, help="Number of input tokens."
    )
    parser.add_argument(
        "--n_output_tokens", type=int, required=True, help="Number of output tokens."
    )
    parser.add_argument(
        "--output_file",
        default="returned_requests.txt",
        type=str,
        required=False,
        help="Output file for failed responses.",
    )

    args = parser.parse_args()

    url = (
        "http://localhost:8000/v2/models/tensorrt_llm_bls/generate_stream"
        if args.target == "triton"
        else "http://localhost:5000/predictions"
    )

    if args.target not in ["cog-triton", "triton"]:
        headers = {
            "Authorization": f"Token {REPLICATE_API_TOKEN}",
            "Content-Type": "application/json",
        }
        url = "https://api.replicate.com/v1/predictions"

    else:
        headers = {"Content-Type": "application/json"}
        url = (
            "http://localhost:8000/v2/models/tensorrt_llm_bls/generate_stream"
            if args.target == "triton"
            else "http://localhost:5000/predictions"
        )

    data = format_request_data(args.n_input_tokens, args.n_output_tokens, args.target)
    start_time = datetime.now()

    rate = int(args.rate) if args.unit == "batch" else args.rate

    await perform_requests(
        rate, args.duration, url, headers, data, args.unit, args.target
    )

    total_requests = (
        args.rate * args.duration
        if args.unit == "rps"
        else args.rate * (args.duration // 1)
    )
    elapsed = datetime.now() - start_time

    # Ensure at least one request was made for statistics calculations
    if not times:
        print("No requests completed.")
        return
    print("---" * 10)
    print("Test Configuration:")
    print("---" * 10)
    print(f"Target: {args.target}")
    print(f"Mode: {args.unit}")
    print(f"Rate: {args.rate} {args.unit}")
    print(f"Duration: {args.duration} seconds")
    print(f"Input tokens: {args.n_input_tokens}")
    print(f"Output tokens: {args.n_output_tokens}")
    print("---" * 10)
    if args.unit == "rps":
        print("Concurrency levels:")
        concurrency_levels = calculate_concurrency(
            start_end_times, args.duration, start_time
        )
        # mean, median, mode, max, min concurrency
        print(f"Mode concurrency: {stats.mode(concurrency_levels)}")
        print(f"Mean concurrency: {stats.mean(concurrency_levels)}")
        print(f"Median concurrency: {stats.median(concurrency_levels)}")
        print(f"Max concurrency: {max(concurrency_levels)}")
        print(f"Min concurrency: {min(concurrency_levels)}")
        print("---" * 10)
    print("Statistics for completed predictions:")
    print("---" * 10)
    print("Response times (seconds):")
    print("---" * 10)
    if sstps:
        print(f"Average single-stream TPS: {stats.mean(sstps):.3f}")
    print("Median response latency:", round(stats.median(times), 3), "seconds")
    print("Mean response latency:", round(stats.mean(times), 3), "seconds")
    print("Max response latency:", round(max(times), 3), "seconds")
    print("Min response latency:", round(min(times), 3), "seconds")
    failure_rate = failures / total_requests if total_requests > 0 else 0
    print("---" * 10)
    print(f"Total requests made: {n_requests}")
    if "cog" in args.target:
        print(
            f"Cog already running prediction errors: {n_cog_already_running_prediction}"
        )
    print(f"Failure rate: {failure_rate:.3f}, Total failures: {failures}")
    print(
        f"Empty output count: {failures}, Non-empty output count: {total_requests - failures}"
    )
    print(f"E2E throughput: {total_requests / elapsed.total_seconds():.3f} rps")

    with open(args.output_file, "w") as f:
        for request in returned_requests:
            f.write(json.dumps(request))
            f.write("\n")

    # print(
    #     f"Throughput: {stats.mean(sstps) * args.n_output_tokens:.3f} tokens per second"
    # )


if __name__ == "__main__":
    asyncio.run(main())
