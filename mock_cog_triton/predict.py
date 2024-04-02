# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md
import os
import subprocess
import time
import httpx
from cog import BasePredictor, ConcatenateIterator, Input
import time


import asyncio
from cog import ConcatenateIterator


class Predictor(BasePredictor):
    async def setup(self, weights: str = None) -> None:
        pass

    async def accurate_sleep(self, target_duration):
        await asyncio.sleep(target_duration)

    async def predict(self, tps: int = 44, n_output_tokens: int = 128, output_method: str = "yield", buffer_size: int = None) -> ConcatenateIterator:
        sleep_offset = 0.00001

        if output_method == "buffer" and buffer_size is None:
            raise ValueError("Buffer size must be set when using buffer output method.")

        num_tokens_processed = 0
        sleep_time = 1 / tps - sleep_offset
        start_time = time.perf_counter()
        output = ""
        buffer = 0
        while num_tokens_processed < n_output_tokens:
            token_start = time.perf_counter()
            await self.accurate_sleep(sleep_time)
            token_end = time.perf_counter()

            if num_tokens_processed == 0:
                first_token_time = token_end

            if output_method == "yield":
                yield str(token_end - token_start) + " "
            elif output_method == "wait":
                output += str(token_end - token_start) + " "
            elif output_method == "buffer":
                if buffer == buffer_size:
                    yield output

                    output = ""
                    buffer = 0
                output += str(token_end - token_start) + " "
                buffer += 1

            num_tokens_processed += 1

        end_time = time.perf_counter()

        if output_method == "wait":
            yield output

        latency = end_time - start_time
        actual_tps = num_tokens_processed / latency
        time_to_first_token = first_token_time - start_time

        self.log(f"Tokens processed: {num_tokens_processed}\n")

        self.log(f"Expected tokens per second: {tps}\n")
        self.log(f"Serverside tokens per second: {round(actual_tps, 2)}\n")

        self.log(f"Expected execution time: {round(n_output_tokens / tps, 2)}\n")
        self.log(f"Serverside execution time: {round(latency, 2)} seconds\n")

        self.log(f"Expected time to first token: {round(1 / tps, 2)} seconds\n")
        self.log(f"Serverside time to first token: {round(time_to_first_token, 2)} seconds\n")
