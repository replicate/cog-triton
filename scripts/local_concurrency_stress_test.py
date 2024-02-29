import asyncio
import json
import statistics as stats
import time
import subprocess
import sys
from datetime import datetime as dt
import argparse

times = []
failures = 0
concurrent_requests_levels = []
max_concurrency_errors = 0
empty_output_count = 0
non_empty_output_count = 0


async def curl_predict(response_file):
    global max_concurrency_errors, empty_output_count, non_empty_output_count, failures  # Add 'failures' to global declaration
    start_time = dt.now()

    # curl -X POST localhost:8000/v2/models/ensemble/generate -d '{"text_input": "What is machine learning?", "max_tokens": 20, "bad_words": "", "stop_words": ""}'
    process = await asyncio.create_subprocess_shell(
        'curl -s -X POST -H "Content-Type: application/json" '
        '-d \'{"input": {"prompt": "Water + Fire = Steam\\nEarth + Water = Plant\\nHuman + Robe = Judge\\nCow + Fire = Steak\\nKing + Ocean = Poseidon\\nComputer + Spy =", '
        '"max_new_tokens":20, "stop_words": "\\n"}}\' http://localhost:5000/predictions',
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    # process = await asyncio.create_subprocess_shell(
    #     'curl -s -X POST -H "Content-Type: application/json" '
    #     '-d \'{"text_input": "Water + Fire = Steam\\nEarth + Water = Plant\\nHuman + Robe = Judge\\nCow + Fire = Steak\\nKing + Ocean = Poseidon\\nComputer + Spy =", '
    #     '"max_tokens":20, "bad_words": [], "stop_words": ["\\n"], "stream":false}\' '
    #     "http://localhost:8000/v2/models/tensorrt_llm_bls/generate_stream",
    #     stdout=asyncio.subprocess.PIPE,
    #     stderr=asyncio.subprocess.PIPE,
    # )

    stdout, stderr = await process.communicate()
    end_time = dt.now()
    delta = (end_time - start_time).total_seconds()
    times.append(delta)

    if process.returncode == 0 and stdout:
        try:
            # Decode the stdout bytes object to a string
            decoded_stdout = stdout.decode("utf-8").strip()

            # Use an alternative to 'removeprefix' for Python versions before 3.9
            prefix = "data: "
            if decoded_stdout.startswith(prefix):
                json_str = decoded_stdout[len(prefix) :]
            else:
                json_str = decoded_stdout

            response = json.loads(json_str)
            print(json.dumps(response), file=response_file)

            if "text_output" in response and response["text_output"]:
                non_empty_output_count += 1
                times.append(delta)
            else:
                failures += 1
        except json.JSONDecodeError:
            failures += 1
            print("Failed to decode JSON from response.", file=response_file)
            print(
                f"Raw response: {decoded_stdout}", file=response_file
            )  # Log raw response for debugging
    else:
        failures += 1


async def main():
    parser = argparse.ArgumentParser(
        description="Script to test server with specified load."
    )
    parser.add_argument("--rps", type=float, help="Requests per second.")
    parser.add_argument("--cr", type=int, help="Concurrent requests per burst.")
    parser.add_argument(
        "--duration", type=int, help="Duration for RPS mode in seconds."
    )
    parser.add_argument(
        "--bursts", type=int, help="Number of bursts for concurrent mode."
    )
    parser.add_argument("output_file", type=str, help="Output file for responses.")

    args = parser.parse_args()
    output_file = args.output_file
    total_requests_made = 0

    if args.rps and args.duration:
        mode = "rps"
        rps = args.rps
        duration = args.duration
        total_requests = int(
            rps * duration
        )  # Define total_requests based on calculation
        tasks = []  # Collect tasks to ensure they are all completed
        start = time.time()
        end_time = start + duration
        print(f"Mode: {mode}, {rps} requests per second for {duration} seconds")

        with open(output_file, "w") as response_file:
            while time.time() < end_time:
                if (
                    len(tasks) >= rps * duration
                ):  # Prevent creating more tasks than planned
                    break
                task = asyncio.create_task(curl_predict(response_file))
                tasks.append(task)
                await asyncio.sleep(1 / rps)
            await asyncio.gather(*tasks)  # Wait for all tasks to complete

    elif args.cr and args.bursts:
        mode = "cr"
        cr = args.cr
        bursts = args.bursts
        total_requests_made = cr * bursts
        print(
            f"Mode: {mode}, {cr} concurrent requests per burst, {bursts} total bursts"
        )

        with open(output_file, "w") as response_file:
            for _ in range(bursts):
                tasks = [
                    asyncio.create_task(curl_predict(response_file)) for _ in range(cr)
                ]
                await asyncio.gather(*tasks)

    elapsed = time.time() - start
    print("Statistics for completed predictions:")
    if times:
        print("Median:", round(stats.median(times), 3), "seconds")
        print("Mean:", round(stats.mean(times), 3), "seconds")
        print("Max:", round(max(times), 3), "seconds")
        print("Min:", round(min(times), 3), "seconds")
    print(f"Failure rate: {failures / total_requests:.3f}, Total failures: {failures}")
    print(f"Cog at max concurrency errors: {max_concurrency_errors}")
    print(
        f"Empty output count: {empty_output_count}, Non-empty output count: {non_empty_output_count}"
    )
    print(f"E2E throughput: {total_requests/elapsed:.3f}rps")


asyncio.run(main())
