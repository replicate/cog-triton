import asyncio
import httpx
import statistics as stats
import argparse
from datetime import datetime, timedelta
import os
import json
import random
import cProfile
# Environment variable for Replicate API token
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")

# Global counters and lists
times = []
failures = 0
sstps = []  # Single Stream Tokens Per Second
start_end_times = []  # Store start and end times for each request
start_times = []
client = httpx.AsyncClient(timeout=300)
client.headers["Replicate-Debug"] = "full-trace"

returned_requests = []
n_requests_made = 0
n_requests_started = 0
n_requests_completed = 0
n_cog_already_running_prediction = 0

server_side_expected_tps = []
server_side_actual_tps = []
server_side_expected_execution_time = []
server_side_actual_execution_time = []
server_side_expected_time_to_first_token = []
server_side_actual_time_to_first_token = []

def parse_request_logs(logs_string):
    global server_side_expected_tps, server_side_actual_tps
    global server_side_expected_execution_time, server_side_actual_execution_time
    global server_side_expected_time_to_first_token, server_side_actual_time_to_first_token
        # Split the logs string into lines
    log_lines = logs_string.split("\n")

    # Iterate over each log line and extract the relevant metrics
    for line in log_lines:
        if ":" in line:
            key, value = line.split(":", 1)
            key = key.strip().lower().replace(" ", "_")
            value = value.strip()

            # Convert numeric values to float or int and append to the corresponding global list
            if key == "expected_tokens_per_second":
                server_side_expected_tps.append(float(value))
            elif key == "serverside_tokens_per_second":
                server_side_actual_tps.append(float(value))
            elif key == "expected_execution_time":
                server_side_expected_execution_time.append(float(value.split(" ")[0]))
            elif key == "serverside_execution_time":
                server_side_actual_execution_time.append(float(value.split(" ")[0]))
            elif key == "expected_time_to_first_token":
                server_side_expected_time_to_first_token.append(float(value.split(" ")[0]))
            elif key == "serverside_time_to_first_token":
                server_side_actual_time_to_first_token.append(float(value.split(" ")[0]))


def parse_cog_time(x):
    x = x[:26] + "Z"
    return datetime.fromisoformat(x.rstrip("Z"))


async def make_request(url, headers, data):
    global failures, sstps, start_end_times, returned_requests, n_requests_made
    global n_cog_already_running_prediction, start_times, n_requests_completed, n_requests_started

    start_time = datetime.now()
    start_times.append(start_time)
    # print(f"[REQUEST STARTED]: {start_time}")  # Log start time

    try:
        # Make the request
        response = await client.post(url, headers=headers, json=data)
        n_requests_made += 1

        end_time = datetime.now()
        returned_requests.append(response.text)


        # Handle "Already running a prediction" from cog-triton
        if (
            "Already running a prediction" in response.text
            and response.status_code == 409
        ):
            n_cog_already_running_prediction += 1
            return


        # if we get here, the request was at least picked up by cog, if we're using cog
        n_requests_started += 1

        request_completed = False

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
            max_tokens = data["input"]["n_output_tokens"]

            sstps.append(max_tokens / delta)

            #parse server-side logs to get internal metrics
            parse_request_logs(response["logs"])


    except Exception as e:
        failures += 1
        print(response)
        print(f"Request failed: {e}")


async def perform_requests(rate, duration, url, headers, data, mode):
    start_time = datetime.now()
    tasks = []
    while (datetime.now() - start_time).total_seconds() < duration:
        if mode == "batch":
            tasks = [
                asyncio.create_task(make_request(url, headers, data))
                for _ in range(rate)
            ]
            start_batch_time = datetime.now()
            await asyncio.gather(*tasks)
            end_batch_time = datetime.now()
            await asyncio.sleep(1)  # Rate limit batches per second
        elif mode == "rps":
            await asyncio.sleep(1 / rate)  # Sleep to maintain the rate
            tasks.append(asyncio.create_task(make_request(url, headers, data)))

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
        "--n_output_tokens", type=int, required=True, help="Number of output tokens."
    )  
    parser.add_argument(
        "--tps", type=int, required=True, help="Number of tokens per second."
    )
    parser.add_argument(
        "--output_method", type=str, required=True, help="Output method.", choices=["yield", "wait", "print", "buffer"]
    )
    parser.add_argument(
        "--buffer_size", type=int, required=False, help="Buffer size."
    )
    parser.add_argument(
        "--output_file",
        default="returned_requests.txt",
        type=str,
        required=False,
        help="Output file for failed responses.",
    )

    args = parser.parse_args()


    headers = {"Content-Type": "application/json"}
    url = "http://localhost:5000/predictions"
    

    data = {
            "input": {
                "tps": args.tps,
                "n_output_tokens": args.n_output_tokens,
                "output_method": args.output_method,
                "buffer_size": args.buffer_size,
            }
        }


    rate = int(args.rate) if args.unit == "batch" else args.rate
    
    start_time = datetime.now()

    await perform_requests(
        rate, args.duration, url, headers, data, args.unit,
    )
    elapsed = datetime.now() - start_time


    total_requests = (
        args.rate * args.duration
        if args.unit == "rps"
        else args.rate * (args.duration // 1)
    )

    # Ensure at least one request was made for statistics calculations
    if not times:
        print("No requests completed.")
        return
    print("---" * 10)
    print("Test Configuration:")
    print("---" * 10)
    print(f"Output Method: {args.output_method}")
    print(f"Mode: {args.unit}")
    print(f"Rate: {args.rate} {args.unit}")
    print(f"Duration: {args.duration} seconds")
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
    if sstps:
        print("Single-stream TPS:")
        if len(sstps) > 1:
            print(f"SSTPS - Std: {stats.stdev(sstps):.3f}")
        print(f"SSTPS - Median: {stats.median(sstps):.3f}")
        print(f"SSTPS - Mean: {stats.mean(sstps):.3f}")
        print(f"SSTPS - Max: {max(sstps):.3f}")
        print(f"SSTPS - Min: {min(sstps):.3f}")
    print("---" * 10)
    if len(times) > 1:
        print(f"Latency - Std: {stats.stdev(times):.3f} seconds")
    print("Median response latency:", round(stats.median(times), 3), "seconds")
    print("Mean response latency:", round(stats.mean(times), 3), "seconds")
    print("Max response latency:", round(max(times), 3), "seconds")
    print("Min response latency:", round(min(times), 3), "seconds")
 
    print("---" * 10)
    print("Server-side metrics:")
    print("---" * 10)
    
    if len(server_side_expected_tps) > 1:
        print("Server-side TPS")
        print(f"--Expected mean: {stats.mean(server_side_expected_tps):.3f}, Actual mean: {stats.mean(server_side_actual_tps):.3f}")
        print(f"--Expected std: {stats.stdev(server_side_expected_tps):.3f}, Actual std: {stats.stdev(server_side_actual_tps):.3f}")
        print(f"--Expected median: {stats.median(server_side_expected_tps):.3f}, Actual median: {stats.median(server_side_actual_tps):.3f}")
        print(f"--Expected min: {min(server_side_expected_tps):.3f}, Actual min: {min(server_side_actual_tps):.3f}")
        print(f"--Expected max: {max(server_side_expected_tps):.3f}, Actual max: {max(server_side_actual_tps):.3f}")
    else:
        print("Server-side TPS")
        print(f"--Expected mean: {stats.mean(server_side_expected_tps):.3f}, Actual mean: {stats.mean(server_side_actual_tps):.3f}")
        print(f"--Expected std: N/A, Actual std: N/A")
        print(f"--Expected median: {stats.median(server_side_expected_tps):.3f}, Actual median: {stats.median(server_side_actual_tps):.3f}")
        print(f"--Expected min: {min(server_side_expected_tps):.3f}, Actual min: {min(server_side_actual_tps):.3f}")
        print(f"--Expected max: {max(server_side_expected_tps):.3f}, Actual max: {max(server_side_actual_tps):.3f}")

    if len(server_side_expected_execution_time) > 1:
        print("Response Latency")
        print(f"--Expected mean: {stats.mean(server_side_expected_execution_time):.3f}, Actual mean: {stats.mean(server_side_actual_execution_time):.3f}")
        print(f"--Expected std: {stats.stdev(server_side_expected_execution_time):.3f}, Actual std: {stats.stdev(server_side_actual_execution_time):.3f}")
        print(f"--Expected median: {stats.median(server_side_expected_execution_time):.3f}, Actual median: {stats.median(server_side_actual_execution_time):.3f}")
        print(f"--Expected min: {min(server_side_expected_execution_time):.3f}, Actual min: {min(server_side_actual_execution_time):.3f}")
        print(f"--Expected max: {max(server_side_expected_execution_time):.3f}, Actual max: {max(server_side_actual_execution_time):.3f}")
    else:
        print("Response Latency")
        print(f"--Expected mean: {stats.mean(server_side_expected_execution_time):.3f}, Actual mean: {stats.mean(server_side_actual_execution_time):.3f}")
        print("--Expected std: N/A, Actual std: N/A")
        print(f"--Expected median: {stats.median(server_side_expected_execution_time):.3f}, Actual median: {stats.median(server_side_actual_execution_time):.3f}")
        print(f"--Expected min: {min(server_side_expected_execution_time):.3f}, Actual min: {min(server_side_actual_execution_time):.3f}")
        print(f"--Expected max: {max(server_side_expected_execution_time):.3f}, Actual max: {max(server_side_actual_execution_time):.3f}")

    if len(server_side_expected_time_to_first_token) > 1:
        print("Time to First Token")
        print(f"--Expected mean: {stats.mean(server_side_expected_time_to_first_token):.3f}, Actual mean: {stats.mean(server_side_actual_time_to_first_token):.3f}")
        print(f"--Expected std: {stats.stdev(server_side_expected_time_to_first_token):.3f}, Actual std: {stats.stdev(server_side_actual_time_to_first_token):.3f}")
        print(f"--Expected median: {stats.median(server_side_expected_time_to_first_token):.3f}, Actual median: {stats.median(server_side_actual_time_to_first_token):.3f}")
        print(f"--Expected min: {min(server_side_expected_time_to_first_token):.3f}, Actual min: {min(server_side_actual_time_to_first_token):.3f}")
        print(f"--Expected max: {max(server_side_expected_time_to_first_token):.3f}, Actual max: {max(server_side_actual_time_to_first_token):.3f}")
    else:
        print("Time to First Token")
        print(f"--Expected mean: {stats.mean(server_side_expected_time_to_first_token):.3f}, Actual mean: {stats.mean(server_side_actual_time_to_first_token):.3f}")
        print("--Expected std: N/A, Actual std: N/A")
        print(f"--Expected median: {stats.median(server_side_expected_time_to_first_token):.3f}, Actual median: {stats.median(server_side_actual_time_to_first_token):.3f}")
        print(f"--Expected min: {min(server_side_expected_time_to_first_token):.3f}, Actual min: {min(server_side_actual_time_to_first_token):.3f}")
        print(f"--Expected max: {max(server_side_expected_time_to_first_token):.3f}, Actual max: {max(server_side_actual_time_to_first_token):.3f}")

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
    print(f"Cog already running prediction: {n_cog_already_running_prediction}")
    print(f"E2E throughput: {n_requests_completed / elapsed.total_seconds():.3f} rps")


    with open(args.output_file, "w") as f:
        for request in returned_requests:
            f.write(json.dumps(request))
            f.write("\n")



if __name__ == "__main__":
    asyncio.run(main())
