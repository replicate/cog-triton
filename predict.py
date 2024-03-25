# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md
import os
import subprocess
import time
import httpx
from cog import BasePredictor, ConcatenateIterator, Input
import torch
from sse import receive_sse
from triton_config_generator import generate_configs, load_yaml_config

import numpy as np

from utils import (
    maybe_download_tarball_with_pget,
    StreamingTokenStopSequenceHandler,
)

import json
import numpy as np
from typing import AsyncGenerator, List, Any
from pathlib import Path
from tensorrt_llm.executor import ParallelGenerationExecutor, GenerationExecutor#, GenerationRequest
from tensorrt_llm.hlapi.tokenizer import TransformersTokenizer
from transformers import AutoTokenizer
import tensorrt_llm.bindings as tllm


class GenerationRequest:

    def __init__(self,
                 req_id: int,
                 ids: torch.Tensor,
                 end_id: int,
                 pad_id: int,
                 streaming: bool = True,
                 **kwargs):
        self.prompt = None
        self.ids = ids
        self.streaming = streaming
        self.kwargs = kwargs
        self.end_id = end_id
        self.pad_id = pad_id
        self._id = req_id

    def get_inference_request(self) -> tllm.InferenceRequest:
        ir = tllm.InferenceRequest(self._id)
        ir.input_ids = self.ids.to(dtype=torch.int32)
        ir.is_streaming = self.streaming

        def set_property(name: str,
                         dtype: torch.dtype = torch.int32,
                         default: Any = None):
            if name in self.kwargs or default is not None:
                value = self.kwargs.get(name, default)
                setattr(ir, name, torch.tensor([value], dtype=dtype))

        set_property("max_new_tokens", default=[8])

        set_property("end_id", default=self.end_id)
        set_property("pad_id", default=self.pad_id)

        set_property("min_length")
        set_property("temperature", torch.float32)
        set_property("runtime_top_k", torch.int32)
        set_property("runtime_top_p", torch.float32)
        set_property("random_seed", torch.int64)

        if 'stop_words_list' in self.kwargs:
            stop_words_tensor = torch.tensor(self.kwargs['stop_words_list'], dtype=torch.int32)
            setattr(ir, 'stop_words_list', stop_words_tensor)

        return ir


class Predictor(BasePredictor):
    async def setup(self, weights: str = "") -> None:

        engine_dir = os.environ.get(
            "ENGINE_DIR", "/src/triton_model_repo/tensorrt_llm/1/"
        )
        
        if weights:
            self.log(f"Downloading model files from {weights}...")
            maybe_download_tarball_with_pget(
                url=weights,
                dest=engine_dir,
        )


        if not os.listdir(engine_dir):
            print("Engine directory is empty. Exiting.")
            self.model_exists = False
            return
        self.model_exists = True


        self.system_prompt_exists = os.getenv("SYSTEM_PROMPT", None)
        self.end_id = os.getenv("END_ID", 2)
        self.pad_id = os.getenv("PAD_ID", 2)

        if weights:
            self.log(f"Downloading model files from {weights}...")
            maybe_download_tarball_with_pget(
                url=weights,
                dest=engine_dir,
            )

        
        tokenizer_dir = engine_dir
        tokenizer = TransformersTokenizer.from_pretrained(
            tokenizer_dir,
            legacy=False,
            padding_side='left',
            truncation_side='left',
            trust_remote_code=True,
            use_fast=True, 
        )

        world_size=int(os.getenv("WORLD_SIZE", 1))
        if world_size > 1:
            self.executor = ParallelGenerationExecutor(
                tp_size=world_size,
                engine_dir=engine_dir, 
                tokenizer=tokenizer,
                max_beam_width=1,
                kvcache_free_gpu_memory_fraction = 0.95
            )

        else:
            executor_config = tllm.TrtGptModelOptionalParams()
            executor_config.kv_cache_config.free_gpu_memory_fraction = 0.95
            self.executor = GenerationExecutor(
                engine_dir=engine_dir, 
                tokenizer=tokenizer,
                executor_config = executor_config,
                max_beam_width=1,
            )

        print("Setup Complete.")

    async def predict(
        self,
        prompt: str = Input(description="Prompt to send to the model."),
        system_prompt: str = Input(
            description="System prompt to send to the model. This is prepended to the prompt and helps guide system behavior.",
            default=os.getenv("SYSTEM_PROMPT", ""),
        ),
        max_new_tokens: int = Input(
            description="Maximum number of tokens to generate. A word is generally 2-3 tokens",
            ge=1,
            default=128,
        ),
        min_new_tokens: int = Input(
            description="Minimum number of tokens to generate. To disable, set to -1. A word is generally 2-3 tokens.",
            ge=-1,
            default=None,
        ),
        temperature: float = Input(
            description="Adjusts randomness of outputs, greater than 1 is random and 0 is deterministic, 0.75 is a good starting value.",
            ge=0.0,
            le=5,
            default=0.7,
        ),
        top_p: float = Input(
            description="When decoding text, samples from the top p percentage of most likely tokens; lower to ignore less likely tokens",
            ge=0.0,
            le=1.0,
            default=0.95,
        ),
        top_k: int = Input(
            description="When decoding text, samples from the top k most likely tokens; lower to ignore less likely tokens",
            ge=-1,
            default=0,
        ),
        stop_sequences: str = Input(
            description="A comma-separated list of sequences to stop generation at. For example, '<end>,<stop>' will stop generation at the first instance of 'end' or '<stop>'.",
            default=None,
        ),
        length_penalty: float = Input(
            description="A parameter that controls how long the outputs are. If < 1, the model will tend to generate shorter outputs, and > 1 will tend to generate longer outputs.",
            ge=0.0,
            le=5.0,
            default=1.0,
        ),
        presence_penalty: float = Input(
            description="A parameter that penalizes repeated tokens regardless of the number of appearances. As the value increases, the model will be less likely to repeat tokens in the output.",
            default=0.0,
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed",
            default=None,
        ),
        prompt_template: str = Input(
            description="Template for formatting the prompt. Can be an arbitrary string, but must contain the substring `{prompt}`.",
            default=os.getenv("PROMPT_TEMPLATE", "{prompt}"),
        ),
    ) -> ConcatenateIterator:
        if not self.model_exists:
            self.log(
                "Your model directory is empty, so there's nothing to do. Remember, you can't run this like a normal model. You need to YOLO!"
            )
            return

        formatted_prompt = self._format_prompt(
            prompt=prompt, system_prompt=system_prompt, prompt_template=prompt_template
        )
        args = self._process_args(
            prompt=formatted_prompt,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            length_penalty=length_penalty,
            presence_penalty=presence_penalty,
            stop_words=stop_sequences,
            seed=seed,
        )
        
        # Args seem to be specified here: https://github.com/NVIDIA/TensorRT-LLM/blob/4bb65f216f2f1dbddb022f4fdd8925c2856baa58/docs/source/inference_request.md?plain=1#L16
        # But, current official implementation of GenerationRequest doesn't support them all
        # Re-implemented at the top of this module
        request = GenerationRequest(
            req_id=self.executor.get_next_request_id(),
            ids=args["input_ids"],
            streaming=True,
            max_new_tokens=args["max_tokens"],
            temperature=args["temperature"],
            runtime_top_p=args["top_p"],
            runtime_top_k=int(args["top_k"]),
            stop_words_list=args["stop_words"],
            min_length=args["min_length"],
            pad_id=args["pad_id"],
            end_id=args["end_id"],
            random_seed=args["random_seed"]
        )
        
        promise = self.executor.submit(request)

        token_cache = args["input_ids"].cpu().numpy().tolist()
        prompt_length = len(token_cache)
        output_tokens = []
        prev_decoded_text = ""

        async for output in promise:
            token_cache.extend(output.token_ids)
            token = token_cache[prompt_length:]
            decoded_text = self.executor.tokenizer.decode(token_cache[prompt_length:], add_special_tokens=False)
            
            # Replace partial emojis with an empty string
            decoded_text = decoded_text.replace("\N{Replacement Character}", "")
            
            # Remove the tokens that were already yielded
            current_output = decoded_text[len(prev_decoded_text):]

            if current_output:
                yield current_output
                prev_decoded_text = decoded_text



        self.log(f"Random seed used: `{args['random_seed']}`\n")
        self.log(
            "Note: Random seed will not impact output if greedy decoding is used.\n"
        )
        self.log(f"Formatted prompt: `{formatted_prompt}`")
    

    def _process_args(
        self,
        prompt: str,
        max_new_tokens: int = 250,
        min_new_tokens: int = None,
        top_k: int = 0,
        top_p: float = 0.0,
        temperature: float = 1.0,
        length_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        stop_words: str = None,
        seed: int = None,
        stream: bool = True,
    ):


        min_new_tokens = 0 if min_new_tokens is None else min_new_tokens

        pad_id = 2
        end_id = 2

        if top_k < 0:
            top_k = 0
        if min_new_tokens < 0:
            min_new_tokens = 0

        if not seed:
            seed = int(np.random.randint(0, 100000))

        input_ids = self.executor.tokenizer.encode(prompt, return_tensors="pt")
        max_new_tokens = [max_new_tokens]
        
        stop_words_list = stop_words.split(",") if stop_words else []

        stop_words_list = self._to_word_list_format([stop_words_list])


        args = {
            "input_ids": input_ids,
            "max_tokens": max_new_tokens,
            "min_length": min_new_tokens,
            "top_k": top_k,
            "temperature": temperature,
            "top_p": top_p,
            "length_penalty": length_penalty,
            "presence_penalty": presence_penalty,
            "stop_words": stop_words_list,
            "stream": stream,
            "random_seed": seed,
            "pad_id": pad_id,
            "end_id": end_id,
        }

        return args

    def _format_prompt(
        self, prompt: str, prompt_template: str, system_prompt: str
    ) -> str:
        if not prompt_template:
            return prompt
        if "system_prompt" in prompt_template:
            system_prompt = system_prompt if system_prompt else ""
            formatted_prompt = prompt_template.format(
                system_prompt=system_prompt, prompt=prompt
            )
            return formatted_prompt
        formatted_prompt = prompt_template.format(prompt=prompt)
        return formatted_prompt

    def _to_word_list_format(self, word_lists: List[List[str | bytes]]):
        """
        word_lists format:
            len(word_lists) == batch_size
            word_lists[i] means the words associated to batch item i. A "word" may actually be any string. Like "lorem" or "lorem ipsum".
        """
        # assert self.tokenizer != None, "need to set tokenizer"
        if word_lists is None:
            # Return an empty array of shape (1,2,0)
            return np.empty([1, 2, 0], dtype="int32")

        flat_ids = []
        offsets = []
        arbitrary_start_sequence_token = "!"
        arbitrary_start_sequence_id = self.executor.tokenizer.encode(
            "!", add_special_tokens=False
        )[0]

        for word_list in word_lists:
            item_flat_ids = []
            item_offsets = []

            for word in word_list:
                if isinstance(word, bytes):
                    word = word.decode()

                word = arbitrary_start_sequence_token + word
                ids = self.executor.tokenizer.encode(word, add_special_tokens=False)
                if ids[0] != arbitrary_start_sequence_id:
                    raise ValueError(
                        f"To standardize tokenizer behavior, we prepend '{arbitrary_start_sequence_token}' to the string representation of each stop sequence."
                        "We then strip the corresponding first token from the stop sequence IDs."
                        "However, the first token of the stop sequence IDs was not '{arbitrary_start_sequence_id}', which suggestions there is a problem with the tokenizer that you are using."
                    )
                else:
                    ids = ids[1:]
                if len(ids) == 0:
                    continue

                item_flat_ids += ids
                item_offsets.append(len(ids))

            flat_ids.append(np.array(item_flat_ids))
            offsets.append(np.cumsum(np.array(item_offsets)))

        pad_to = max(1, max(len(ids) for ids in flat_ids))

        for i, (ids, offs) in enumerate(zip(flat_ids, offsets)):
            flat_ids[i] = np.pad(ids, (0, pad_to - len(ids)), constant_values=0)
            offsets[i] = np.pad(offs, (0, pad_to - len(offs)), constant_values=-1)

        return np.array([flat_ids, offsets], dtype="int32").transpose((1, 0, 2))
