import asyncio
import httpx
import statistics as stats
import argparse
from datetime import datetime, timedelta
import os
import json
import matplotlib.pyplot as plt
import os
from datetime import datetime
import sys
# Environment variable for Replicate API token
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")

from transformers import AutoTokenizer

# Global counters and lists
times = []
failures = 0
sstps = []  # Single Stream Tokens Per Second
start_end_times = []  # Store start and end times for each request
start_times = []
client = httpx.AsyncClient(timeout=300)
returned_requests = []
n_requests_made = 0
n_requests_started = 0
n_requests_completed = 0
n_cog_already_running_prediction = 0

tokenizer = AutoTokenizer.from_pretrained("triton_model_repo/tensorrt_llm/1/")



server_side_expected_tps = []
server_side_actual_tps = []
server_side_expected_execution_time = []
server_side_actual_execution_time = []
server_side_expected_time_to_first_token = []
server_side_actual_time_to_first_token = []

def parse_request_logs(logs_string):
    global server_side_actual_tps
    global server_side_actual_execution_time
    global server_side_actual_time_to_first_token
        # Split the logs string into lines
    log_lines = logs_string.split("\n")

    # Iterate over each log line and extract the relevant metrics
    for line in log_lines:
        if ":" in line:
            key, value = line.split(":", 1)
            key = key.strip().lower().replace(" ", "_")
            value = value.strip()

            # Convert numeric values to float or int and append to the corresponding global list

            if key == "serverside_tokens_per_second":
                server_side_actual_tps.append(float(value))
            elif key == "serverside_execution_time":
                server_side_actual_execution_time.append(float(value.split(" ")[0]))
            elif key == "serverside_time_to_first_token":
                server_side_actual_time_to_first_token.append(float(value.split(" ")[0]))

def count_output_tokens(text):
    return len(tokenizer.tokenize(text))

def format_request_data(n_input_tokens, n_output_tokens, target):
    prompt = " a" *(n_input_tokens - 1)
    prompt_ids = count_output_tokens(prompt)
    assert n_input_tokens == prompt_ids, f"incorrect number of input tokens: {prompt_ids}. Expected: {n_input_tokens}"


    if target not in ["cog-triton", "triton"]:
        return {
            "version": target,
            "input": {
                "prompt": prompt,
                "min_new_tokens": n_output_tokens,
                "max_new_tokens": n_output_tokens,
            },
        }

    elif target == "cog-triton":
        return {
            "input": {
                "prompt": prompt,
                "max_new_tokens": n_output_tokens,
                "min_new_tokens": n_output_tokens,
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

    # try:
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
        if not response["output_ids"]:
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

        
        if target == "cog-triton":
            output = ''.join(response["output"])
            n_input_tokens = data["input"]["max_new_tokens"]
            max_tokens = data["input"]["max_new_tokens"]
        elif target == "triton":
            # Trito
            input_text = data["text_input"]
            output = response["output_ids"][len(input_text):]
            n_input_tokens = data["min_length"]
            max_tokens = data["max_tokens"]

        else:
            output = response["output"]
            n_input_tokens = data["input"]["max_new_tokens"]
            max_tokens = data["input"]["max_new_tokens"]
        
        # assert n_output_tokens == max_tokens, f"incorrect number of output tokens: {n_output_tokens}. Expected: {max_tokens}"
        
        delta = (end_time - start_time).total_seconds()
        times.append(delta)
        start_end_times.append((start_time, end_time))
        if target != "triton":
            max_tokens = data["input"]["max_new_tokens"]
        else:
            max_tokens = data["max_tokens"]
        sstps.append(max_tokens / delta)

        if "logs" in response:
            parse_request_logs(response["logs"])


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
            await asyncio.sleep(.5)  # Rate limit batches per second
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

def plot_metrics_with_lines(x, y, x_label, y_label, title, file_name):
    plt.figure(figsize=(10, 6))
    
    # Sort the x and y values based on x to connect them correctly
    sorted_indices = sorted(range(len(x)), key=lambda i: x[i])
    sorted_x = [x[i] for i in sorted_indices]
    sorted_y = [y[i] for i in sorted_indices]
    
    # Plot points and connect them with a line
    plt.scatter(sorted_x, sorted_y, alpha=0.5)  # Plot points
    plt.plot(sorted_x, sorted_y, '-o', label='Data', color='blue')  # Connect points with line
    
    # Add a dotted line for the median
    median_value = stats.median(y)
    plt.axhline(y=median_value, color='r', linestyle='--', label=f'Median: {median_value:.3f}')
    
    # Add titles and labels
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()  # Show legend
    
    # Save and close
    plt.savefig(file_name)
    plt.close()

def create_run_dir(args):
    base_dir = "perf-results"
    os.makedirs(base_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    unique_dir_name = f"{timestamp}-{args.target}-{args.unit}-{int(args.rate)}-{args.duration}-{args.n_input_tokens}-{args.n_output_tokens}"
    run_dir = os.path.join(base_dir, unique_dir_name)
    os.makedirs(run_dir)

    return run_dir


class DualOutput:
    def __init__(self, filePath):
        self.terminal = sys.stdout
        self.log = open(filePath, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):  # needed for Python 3 compatibility
        pass

async def main(run_dir, args):

    sys.stdout = DualOutput(os.path.join(run_dir, "output.txt"))

    if args.target not in ["cog-triton", "triton"]:
        headers = {
            "Authorization": f"Token {REPLICATE_API_TOKEN}",
            "Content-Type": "application/json",
        }
        url = "https://api.replicate.com/v1/predictions"

    else:
        headers = {"Content-Type": "application/json"}
        url = (
            "http://localhost:8000/v2/models/ensemble/generate_stream"
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
        print("Single-stream TPS:")
        if len(sstps) > 1:
            print(f"SSTPS - Std: {stats.stdev(sstps):.3f}")
        print(f"SSTPS - Median: {stats.median(sstps):.3f}")
        print(f"SSTPS - Mean: {stats.mean(sstps):.3f}")
        print(f"SSTPS - Max: {max(sstps):.3f}")
        print(f"SSTPS - Min: {min(sstps):.3f}")
        print("---" * 10)

    print("Median response latency:", round(stats.median(times), 3), "seconds")
    print("Mean response latency:", round(stats.mean(times), 3), "seconds")
    if len(times) > 1:
        print(f"Response Latency - Std: {stats.stdev(times):.3f} seconds")
    print("Max response latency:", round(max(times), 3), "seconds")
    print("Min response latency:", round(min(times), 3), "seconds")

    print("---" * 10)
    print("Server-side metrics:")
    print("---" * 10)
    if len(server_side_actual_tps) > 0:
        if len(server_side_actual_tps) > 1:
            print("Server-side TPS")
            print(f"--Actual mean: {stats.mean(server_side_actual_tps):.3f}")
            print(f"--Actual std: {stats.stdev(server_side_actual_tps):.3f}")
            print(f"--Actual median: {stats.median(server_side_actual_tps):.3f}")
            print(f"--Actual min: {min(server_side_actual_tps):.3f}")
            print(f"--Actual max: {max(server_side_actual_tps):.3f}")
        else:
            print("Server-side TPS")
            print(f"--Actual mean: {stats.mean(server_side_actual_tps):.3f}")
            print("--Actual std: N/A")
            print(f"--Actual median: {stats.median(server_side_actual_tps):.3f}")
            print(f"--Actual min: {min(server_side_actual_tps):.3f}")
            print(f"--Actual max: {max(server_side_actual_tps):.3f}")

        if len(server_side_actual_execution_time) > 1:
            print("Response Latency")
            print(f"--Actual mean: {stats.mean(server_side_actual_execution_time):.3f}")
            print(f"--Actual std: {stats.stdev(server_side_actual_execution_time):.3f}")
            print(f"--Actual median: {stats.median(server_side_actual_execution_time):.3f}")
            print(f"--Actual min: {min(server_side_actual_execution_time):.3f}")
            print(f"--Actual max: {max(server_side_actual_execution_time):.3f}")
        else:
            print("Response Latency")
            print(f"--Actual mean: {stats.mean(server_side_actual_execution_time):.3f}")
            print("--Actual std: N/A")
            print(f"--Actual median: {stats.median(server_side_actual_execution_time):.3f}")
            print(f"--Actual min: {min(server_side_actual_execution_time):.3f}")
            print(f"--Actual max: {max(server_side_actual_execution_time):.3f}")

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
        print(f"Cog already running prediction: {n_cog_already_running_prediction}")
    print(f"E2E throughput: {n_requests_completed / elapsed.total_seconds():.3f} rps")
    
    
    output_file_path = os.path.join(run_dir, "returned_requests.json")

    with open(output_file_path, "w") as f:
        for request in returned_requests:
            f.write(json.dumps(request))
            f.write("\n")

    plot_file_path_tps = os.path.join(run_dir, 'tps_per_response_with_lines.png')
    plot_metrics_with_lines(
        range(len(sstps)), sstps,
        'Response Number', 'Single-Stream TPS',
        f'{args.target} Single-stream TPS per Response -- {args.unit}={args.rate}', plot_file_path_tps
    )
    
    plot_file_path_latency = os.path.join(run_dir, 'latency_per_response_with_lines.png')
    plot_metrics_with_lines(
        range(len(times)), times,
        'Response Number', 'Latency (seconds)',
        f'{args.target} Latency per Response -- {args.unit}={args.rate}', plot_file_path_latency
    )

    # print(
    #     f"Throughput: {stats.mean(sstps) * args.n_output_tokens:.3f} tokens per second"
    # )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark script for Triton or Cog server.")
    parser.add_argument("--target", required=True, help="Target server for the benchmark.")
    parser.add_argument("--rate", type=float, required=True, help="Number of requests per second (for rps) or total concurrent requests (for batch).")
    parser.add_argument("--unit", type=str, choices=["rps", "batch"], required=True, help="Mode of operation: rps for requests per second, batch for concurrent requests.")
    parser.add_argument("--duration", type=int, required=True, help="Duration of test in seconds.")
    parser.add_argument("--n_input_tokens", type=int, required=True, help="Number of input tokens.")
    parser.add_argument("--n_output_tokens", type=int, required=True, help="Number of output tokens.")
    args = parser.parse_args()

    base_dir = "perf-results"
    os.makedirs(base_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    unique_dir_name = f"{timestamp}-{args.target}-{args.unit}-{int(args.rate)}-{args.duration}-{args.n_input_tokens}-{args.n_output_tokens}"
    run_dir = os.path.join(base_dir, unique_dir_name)
    os.makedirs(run_dir, exist_ok=True)
    sys.stdout = DualOutput(os.path.join(run_dir, "output.txt"))
    asyncio.run(main(run_dir, args))


    # plot_file_path_tps = os.path.join(run_dir, 'tps_per_response_with_lines.png')
    # plot_metrics_with_lines(
    #     range(len(sstps)), sstps,
    #     x_label='Response Number',
    #     y_label='Single-Stream TPS',
    #     title='TPS per Response',
    #     file_path=plot_file_path_tps
    # )

    # # Similarly, adjust other file output operations within `main`
    # with open(os.path.join(run_dir, "output.txt"), "w") as f:
    #     f.write("Your output here")

    # # And for saving the args for reproducibility
    # with open(os.path.join(run_dir, "args.json"), "w") as args_file:
    #     json.dump(vars(args), args_file, indent=4)