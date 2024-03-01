import requests, time
import concurrent.futures
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-hf")


def send_request(prompt, output_tokens):
    url = "http://localhost:8000/v2/models/ensemble/generate"
    data = {
        "text_input": prompt,
        "parameters": {
            "max_tokens": output_tokens,
            "min_length": output_tokens,
            "bad_words": [""],
            "stop_words": [""],
            "temperature": 1.0,
        },
    }
    response = requests.post(url, json=data)
    return response


def calculate_margins(
    total_tokens_per_second, instance_cost_per_sec, token_price_range_1k
):
    import numpy as np

    start, end, step = token_price_range_1k
    margins = {}

    for price_per_1k_tokens in np.arange(start, end, step):
        cost_to_generate_1k_tokens = (
            1000 / total_tokens_per_second * instance_cost_per_sec
        )
        margin = (
            price_per_1k_tokens - cost_to_generate_1k_tokens
        ) / price_per_1k_tokens
        margins[price_per_1k_tokens] = margin

    return margins


def concurrent_test(
    n_threads,
    prompt,
    output_tokens,
    instance_cost_per_second,
    token_price_range_1k,
    gpu_type,
    n_gpus,
):
    n_tokens_in_prompt = len(tokenizer.encode(prompt))
    out = []
    start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(send_request, prompt, output_tokens)
            for _ in range(n_threads)
        ]
        for future in concurrent.futures.as_completed(futures):
            response = future.result()
            out.append(response.json()["text_output"])
    request_time = time.perf_counter() - start
    # Triton prepends input tokens to output tokens, so we subtract the number of input tokens from the total number of tokens to get the number of tokens generated.
    toks = sum([len(tokenizer.encode(o)) for o in out]) - n_tokens_in_prompt * n_threads

    total_tokens_per_second = toks / request_time
    single_stream_tps = total_tokens_per_second / n_threads

    margins = calculate_margins(
        total_tokens_per_second, instance_cost_per_second, token_price_range_1k
    )
    cost_to_generate_1k_tokens = (
        1000 / total_tokens_per_second * instance_cost_per_second
    )

    results = {
        "batch_size": n_threads,
        "total_tokens": toks,
        "total_time": request_time,
        "total_tokens_per_second": total_tokens_per_second,
        "single_stream_tps": single_stream_tps,
        "cost_to_generate_1k_tokens": cost_to_generate_1k_tokens,
        "instance_cost_per_second": instance_cost_per_second,
        "gpu_type": gpu_type,
        "n_gpus": n_gpus,
    }

    # Add each margin as a separate key
    for price, margin in margins.items():
        results[f"margin_at_price_{price}"] = margin

    return results


def graph_results(title, filename):
    import matplotlib.pyplot as plt
    import pandas as pd

    # Assuming data is in a Pandas DataFrame 'df'

    df = pd.read_csv(filename + ".csv")

    fig, axes = plt.subplots(
        nrows=3, ncols=2, figsize=(15, 15), constrained_layout=True
    )

    # Pane 1: Total TPS vs. Batch Size
    axes[0, 0].plot(df["batch_size"], df["total_tokens_per_second"], marker="o")
    axes[0, 0].set_title("Total TPS vs. Batch Size")
    axes[0, 0].set_xlabel("Batch Size")
    axes[0, 0].set_ylabel("Total Tokens per Second")
    axes[0, 0].set_xticks(df["batch_size"])
    axes[0, 0].set_xticklabels(df["batch_size"])
    for x, y in zip(df["batch_size"], df["total_tokens_per_second"]):
        axes[0, 0].annotate(
            f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, 10), ha="center"
        )

    # Pane 2: Total Time vs. Batch Size
    axes[0, 1].plot(df["batch_size"], df["total_time"], marker="o")
    axes[0, 1].set_title("Latency vs. Batch Size")
    axes[0, 1].set_xlabel("Batch Size")
    axes[0, 1].set_ylabel("Latency (s)")
    axes[0, 1].set_xticks(df["batch_size"])
    axes[0, 1].set_xticklabels(df["batch_size"])
    for x, y in zip(df["batch_size"], df["total_time"]):
        axes[0, 1].annotate(
            f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, 10), ha="center"
        )

    # Pane 3: Single Stream TPS vs. Batch Size
    axes[1, 0].plot(df["batch_size"], df["single_stream_tps"], marker="o")
    axes[1, 0].set_title("Single Stream TPS vs. Batch Size")
    axes[1, 0].set_xlabel("Batch Size")
    axes[1, 0].set_ylabel("Single Stream Tokens per Second")
    axes[1, 0].set_xticks(df["batch_size"])
    axes[1, 0].set_xticklabels(df["batch_size"])
    for x, y in zip(df["batch_size"], df["single_stream_tps"]):
        axes[1, 0].annotate(
            f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, 10), ha="center"
        )

    # Pane 4: Percent Change Bar Chart

    numeric_cols_for_pct_change = [
        "total_tokens_per_second",
        "total_time",
        "single_stream_tps",
    ]
    percent_changes = df[numeric_cols_for_pct_change].pct_change().dropna()

    # Using a range for x-axis
    x_axis = range(len(percent_changes))

    axes[1, 1].bar(x_axis, percent_changes["total_tokens_per_second"], color="g")
    axes[1, 1].set_title("Percent Change in Total TPS vs. Batch Size")
    axes[1, 1].set_xlabel("Batch Size Transition")
    axes[1, 1].set_ylabel("Percent Change")

    # Set x-ticks to represent transitions between batch sizes
    batch_size_transitions = [
        f'{int(df.iloc[i-1]["batch_size"])} to {int(df.iloc[i]["batch_size"])}'
        for i in range(1, len(df))
    ]
    axes[1, 1].set_xticks(x_axis)
    axes[1, 1].set_xticklabels(batch_size_transitions, rotation=45)

    # Annotate bars with percent change values
    for x, y in zip(x_axis, percent_changes["total_tokens_per_second"]):
        axes[1, 1].annotate(
            f"{y:.2%}", (x, y), textcoords="offset points", xytext=(0, 10), ha="center"
        )

    # New Pane for Margin Analysis at position 5 (bottom right)

    # Pane for Margin Analysis
    margin_columns = [col for col in df.columns if col.startswith("margin_at_price_")]
    batch_sizes = df["batch_size"]

    for col in margin_columns:
        axes[2, 0].plot(
            batch_sizes,
            df[col],
            label=f"${round(float(col.split('_')[-1]),4)}/1k tokens",
        )  # Use the price as the label

    # Set x-ticks to observed batch sizes only
    axes[2, 0].set_xticks(batch_sizes)
    axes[2, 0].set_xticklabels(batch_sizes)

    # Add more detail to the y-axis
    axes[2, 0].yaxis.set_major_locator(
        plt.MaxNLocator(10)
    )  # Example: 10 ticks on the y-axis
    axes[2, 0].axhline(y=0, color="gray", linestyle="--")

    axes[2, 0].set_title("Margin vs. Batch Size")
    axes[2, 0].set_xlabel("Batch Size")
    axes[2, 0].set_ylabel("Margin")
    axes[2, 0].legend()

    fig.suptitle(title)

    plt.tight_layout()
    plt.savefig(filename + ".png")
    print(f"Results graphed to {filename}.png")


if __name__ == "__main__":
    import sys
    import json
    import csv
    import argparse

    parser = argparse.ArgumentParser(description="Hamels Load Test")
    parser.add_argument(
        "--input_tokens",
        type=int,
        help="Number of input tokens",
        default=128,
        required=False,
    )
    parser.add_argument(
        "--output_tokens",
        type=int,
        help="Number of output tokens",
        default=128,
        required=False,
    )
    parser.add_argument(
        "--n_requests",
        type=int,
        nargs="+",
        help="Number of requests to test",
        default=[1, 2],
        required=True,
    )
    parser.add_argument(
        "--model",
        type=str,
        help="String describing the model you are benchmarking.",
        required=True,
    )
    parser.add_argument(
        "--instance_cost_per_second",
        type=float,
        help="Hourly instance cost",
        default=0.0,
        required=True,
    )
    parser.add_argument(
        "--token_price_range_1k",
        type=float,
        nargs=3,
        help="Range of token prices to test",
        default=[0.0018, 0.002, 0.0001],
        required=True,
    )
    parser.add_argument("--gpu_type", type=str, help="GPU type", required=True)
    parser.add_argument("--n_gpus", type=int, help="Number of GPUs", required=True)

    args = parser.parse_args()

    input_tokens = args.input_tokens
    output_tokens = args.output_tokens
    n_requests_to_test = args.n_requests
    instance_cost_per_second = args.instance_cost_per_second
    token_price_range_1k = args.token_price_range_1k
    gpu_type = args.gpu_type
    n_gpus = args.n_gpus
    model = args.model
    title = f"{n_gpus}x{gpu_type} {model} {input_tokens}/{output_tokens}"
    filename = f"loadtest_{title.replace(' ', '_').replace('/', '_').lower()}"
    results = []
    prompt = " a" * input_tokens

    # print a summary of the testing configuration
    print("Testing configuration:")
    print(f"Input tokens: {input_tokens}")
    print(f"Output tokens: {output_tokens}")
    print(f"Concurrency to test: {n_requests_to_test}")
    print(f"Instance cost per second: {instance_cost_per_second}")
    print(f"Token price range: {token_price_range_1k}")

    for n_requests in n_requests_to_test:
        print(f"Testing {n_requests} requests")
        results.append(
            concurrent_test(
                n_requests,
                prompt,
                output_tokens,
                instance_cost_per_second,
                token_price_range_1k,
                gpu_type,
                n_gpus,
            )
        )

    with open(filename + ".csv", "w", newline="") as csvfile:
        # Dynamically determine fieldnames from the keys of the first result
        fieldnames = list(results[0].keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result)

        print(f"Results written to {filename}.csv")

    graph_results(title, filename)
