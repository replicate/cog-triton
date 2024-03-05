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
start_times = []
client = httpx.AsyncClient()
returned_requests = []
n_requests_made = 0
n_requests_started = 0
n_requests_completed = 0
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
    global failures, sstps, start_end_times, returned_requests, n_requests_made, n_cog_already_running_prediction, start_times, n_requests_completed, n_requests_started
    start_time = datetime.now()
    start_times.append(start_time)
    # print(f"[REQUEST STARTED]: {start_time}")  # Log start time

    try:
        # Make the request
        response = await client.post(url, headers=headers, json=data)
        n_requests_made += 1

        if target in ["cog-triton", "triton"]:
            end_time = datetime.now()
            returned_requests.append(response.text)

        else:
            response = await poll_replicate_request(response, headers)
            completed_at = response["completed_at"]
            end_time = parse_cog_time(completed_at)
            returned_requests.append(response)

        # Handle "Already running a prediction" from cog-triton
        if target == "cog-triton":
            if (
                "Already running a prediction" in response.text
                and response.status_code == 409
            ):
                n_cog_already_running_prediction += 1
                return

        # Handle "Already running a prediction" from production
        elif target != "triton":
            if (
                "detail" in response
                and "Already running a prediction" in response["detail"]
            ):
                n_cog_already_running_prediction += 1
                return

        # if we get here, the request was at least picked up by cog, if we're using cog
        n_requests_started += 1

        request_completed = False
        if target == "triton":
            if response.status_code != 200:
                failures += 1
            prefix = "data: "
            json_str = response.text[len(prefix) :]
            response = json.loads(json_str)

            # If output is empty, count as failure
            if not response["text_output"]:
                failures += 1
            else:
                n_requests_completed += 1
                request_completed = True

        else:
            if target == "cog-triton":
                response = response.json()
            # print(response)
            if response["status"] == "failed":
                failures += 1
                # print(f"Request failed: {response['error']}")
            elif response["status"] == "succeeded":
                n_requests_completed += 1
                request_completed = True

        if request_completed:
            delta = (end_time - start_time).total_seconds()
            times.append(delta)
            start_end_times.append((start_time, end_time))
            if target != "triton":
                max_tokens = data["input"]["max_new_tokens"]
            else:
                max_tokens = data["max_tokens"]
            sstps.append(max_tokens / delta)

        # if target != "triton":
        #     completed_at = response["completed_at"]
        #     response_id = response["id"]
        #     print(
        #         f"{response_id}: start time {start_time}, end time {end_time}, delta {delta}, completed_at {completed_at}"
        #     )

    except Exception as e:
        failures += 1
        print(response)
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


def estimate_rps(start_times):
    # Ensure start_times is sorted
    # Calculate intervals between consecutive requests in seconds
    intervals = [
        (start_times[i] - start_times[i - 1]).total_seconds()
        for i in range(1, len(start_times))
    ]

    # Convert intervals to rates (requests per second)
    # Avoid division by zero by filtering out zero intervals
    rates = [1 / interval for interval in intervals if interval > 0]

    return rates


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
    failure_rate = failures / n_requests_started if n_requests_started > 0 else 0
    print("---" * 10)
    print(f"Total requests made: {n_requests_made}")
    print(f"Total requests started: {n_requests_started}")
    print(f"Total requests completed: {n_requests_completed}")
    # Calculate mean and median of the rates
    if args.unit == "rps":
        rates = estimate_rps(start_times)
        mean_rps = stats.mean(rates) if rates else 0
        median_rps = stats.median(rates) if rates else 0
        print(f"Observed RPS: {mean_rps:.3f} (mean), {median_rps:.3f} (median)")
    print(f"Failure rate: {failure_rate:.3f}, Total failures: {failures}")
    if "cog" in args.target:
        print(
            f"Cog already running prediction errors: {n_cog_already_running_prediction}"
        )
    print(f"E2E throughput: {n_requests_completed / elapsed.total_seconds():.3f} rps")

    with open(args.output_file, "w") as f:
        for request in returned_requests:
            f.write(json.dumps(request))
            f.write("\n")

    # print(
    #     f"Throughput: {stats.mean(sstps) * args.n_output_tokens:.3f} tokens per second"
    # )


if __name__ == "__main__":
    asyncio.run(main())
