import argparse
import csv
import os
import queue
import sys
import time
from functools import partial
import uuid

import numpy as np
import tritonclient.grpc as grpcclient
from transformers import AutoTokenizer, LlamaTokenizer, T5Tokenizer
from tritonclient.utils import InferenceServerException, np_to_triton_dtype


# Lines 18-168 were copied verbatim from ./tensorrtllm_backend/inflight_batcher_llm/client/inflight_batcher_llm_client.py


np_bfloat16 = np.dtype('V2', metadata={"dtype": "bfloat16"})

_str_to_np_dict = dict(
    float16=np.float16,
    float32=np.float32,
    int32=np.int32,
    bfloat16=np_bfloat16,
)


def curate_log_output(token_sequence,
                      identifier="Input",
                      log_max_sequence_len=256):
    if len(token_sequence) > log_max_sequence_len:
        print(f"{identifier} sequence starts with: ",
              token_sequence[:log_max_sequence_len])
    else:
        print(f"{identifier} sequence: ", token_sequence)

def str_dtype_to_np(dtype):
    ret = _str_to_np_dict.get(dtype)
    assert ret is not None, f'Unsupported dtype: {dtype}'
    return ret

def check_output_names(expected_outputs, infer_result):
    if expected_outputs:
        output_names = set([o.name for o in infer_result._result.outputs])
        if set(expected_outputs) != output_names:
            raise Exception(
                f"expected outputs do not match actual outputs {expected_outputs} != {output_names}"
            )

class UserData:

    def __init__(self):
        self._completed_requests = queue.Queue()
        self._decoded_tokens = queue.Queue() 

def prepare_tensor(name, input):
    t = grpcclient.InferInput(name, input.shape,
                              np_to_triton_dtype(input.dtype))
    t.set_data_from_numpy(input)
    return t

def prepare_outputs(output_names):

    outputs = []
    for output_name in output_names:
        outputs.append(grpcclient.InferRequestedOutput(output_name))
    return outputs

def prepare_inputs(input_ids_data, input_lengths_data, request_output_len_data,
                   beam_width_data, temperature_data, repetition_penalty_data,
                   presence_penalty_data, frequency_penalty_data,
                   streaming_data, end_id, pad_id, prompt_embedding_table_data,
                   prompt_vocab_size_data, lora_weights_data, lora_config_data,
                   return_log_probs_data, top_k_data, top_p_data,
                   draft_ids_data, return_context_logits_data,
                   return_generation_logits_data):
    inputs = [
        prepare_tensor("input_ids", input_ids_data),
        prepare_tensor("input_lengths", input_lengths_data),
        prepare_tensor("request_output_len", request_output_len_data),
        prepare_tensor("beam_width", beam_width_data),
        prepare_tensor("temperature", temperature_data),
        prepare_tensor("streaming", streaming_data),
        prepare_tensor("end_id", end_id),
        prepare_tensor("pad_id", pad_id),
        prepare_tensor("return_log_probs", return_log_probs_data),
        prepare_tensor("runtime_top_k", top_k_data),
        prepare_tensor("runtime_top_p", top_p_data),
    ]
    if prompt_embedding_table_data is not None:
        inputs += [
            prepare_tensor("prompt_embedding_table",
                           prompt_embedding_table_data),
            prepare_tensor("prompt_vocab_size", prompt_vocab_size_data)
        ]
    if lora_weights_data is not None:
        inputs += [
            prepare_tensor("lora_weights", lora_weights_data),
            prepare_tensor("lora_config", lora_config_data),
        ]
    if repetition_penalty_data is not None:
        inputs += [
            prepare_tensor("repetition_penalty", repetition_penalty_data),
        ]
    if presence_penalty_data is not None:
        inputs += [
            prepare_tensor("presence_penalty", presence_penalty_data),
        ]
    if frequency_penalty_data is not None:
        inputs += [
            prepare_tensor("frequency_penalty", frequency_penalty_data),
        ]
    if draft_ids_data is not None:
        inputs += [
            prepare_tensor("draft_input_ids", draft_ids_data),
        ]
    if return_context_logits_data is not None:
        inputs += [
            prepare_tensor("return_context_logits",
                           return_context_logits_data),
        ]
    if return_generation_logits_data is not None:
        inputs += [
            prepare_tensor("return_generation_logits",
                           return_generation_logits_data),
        ]
    return inputs

def prepare_stop_signals():

    inputs = [
        grpcclient.InferInput('input_ids', [1, 1], "INT32"),
        grpcclient.InferInput('input_lengths', [1, 1], "INT32"),
        grpcclient.InferInput('request_output_len', [1, 1], "INT32"),
        grpcclient.InferInput('stop', [1, 1], "BOOL"),
    ]

    inputs[0].set_data_from_numpy(np.empty([1, 1], dtype=np.int32))
    inputs[1].set_data_from_numpy(np.zeros([1, 1], dtype=np.int32))
    inputs[2].set_data_from_numpy(np.array([[0]], dtype=np.int32))
    inputs[3].set_data_from_numpy(np.array([[True]], dtype='bool'))

    return inputs


# Define the callback function. Note the last two parameters should be
# result and error. InferenceServerClient would povide the results of an
# inference as grpcclient.InferResult in result. For successful
# inference, error will be None, otherwise it will be an object of
# tritonclientutils.InferenceServerException holding the error details
def callback(user_data, tokenizer, result, error):
    if error:
        user_data._completed_requests.put(error)
    else:
        user_data._completed_requests.put(result)
        if result.get_output('output_ids') is not None:
            output_ids = result.as_numpy('output_ids')
            seq_lens = result.as_numpy('sequence_length')
            if seq_lens == None or seq_lens[0][0] > 0:
                tokens = list(output_ids[0][0])
                decoded_tokens = tokenizer.decode(tokens)
                user_data._decoded_tokens.put(decoded_tokens)
                print(decoded_tokens, flush=True)

class TRTLLMClient:

    def __init__(
            self, 
            tokenizer, 
            url="localhost:8001",
            verbose=False, 
            ssl=False,
            root_certificates=None, 
            private_key=None, 
            certificate_chain=None,
            stream_timeout=None,
            
        ):
        
        self.tokenizer = tokenizer
        self.url = url
        self.verbose = verbose
        self.ssl = ssl
        self.root_certificates = root_certificates
        self.private_key = private_key
        self.certificate_chain = certificate_chain
        self.stream_timeout = stream_timeout

        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.pad_id = tokenizer.encode(tokenizer.pad_token,
                                  add_special_tokens=False)[0]
        self.end_id = tokenizer.encode(tokenizer.eos_token,
                                  add_special_tokens=False)[0]
        
        self.end_id_data = np.array([[self.end_id]], dtype=np.int32)
        self.pad_id_data = np.array([[self.pad_id]], dtype=np.int32)


        self.prompt_embedding_table_data = None
        self.prompt_vocab_size_data = None
        
        # This are additional settings that we might want to use
        self.lora_weights_data = None
        self.lora_config_data = None
        self.return_context_logits_data = None


    def _preprocess_request_parameters(
            self,
            # supported user inputs
            text,
            output_len=26,
            top_k=1,
            top_p=0.0,
            temperature=1.0,
            beam_width=1,
            repetition_penalty=None,
            presence_penalty=None,
            frequency_penalty=None,
            # currently unsupported inputs
            prompt_embedding_table_data=None,
            prompt_vocab_size_data=None,
            lora_weights_data=None,
            lora_config_data=None,
            return_log_probs=None,
            return_context_logits=None,
            return_generation_logits=None,
            draft_ids=None,
        ): 

        streaming_data = np.array([[True]], dtype=bool)

        end_id_data = self.end_id_data
        pad_id_data = self.pad_id_data

        input_ids = [self.tokenizer.encode(text)]
        # This isn't necessary, but it's the original client 
        # implementation for logging sequences.
        # Keeping it around for now. Should refactor at some point.
        curate_log_output(input_ids[0], "Input")
        input_ids_data = np.array(input_ids, dtype=np.int32)
        input_lengths = [[len(ii)] for ii in input_ids]
        input_lengths_data = np.array(input_lengths, dtype=np.int32)
        
        request_output_len_data = np.array([[output_len]], dtype=np.int32)
        beam_width_data = np.array([[beam_width]], dtype=np.int32)
        top_k_data = np.array([[top_k]], dtype=np.int32)
        top_p_data = np.array([[top_p]], dtype=np.float32)
        temperature_data = np.array([[temperature]], dtype=np.float32)

        repetition_penalty_data = np.array(repetition_penalty, dtype=np.float32) if repetition_penalty else None
        presence_penalty_data = np.array(presence_penalty, dtype=np.float32) if presence_penalty else None
        frequency_penalty_data = np.array(frequency_penalty, dtype=np.float32) if frequency_penalty else None

        return_log_probs_data = np.array([[return_log_probs]], dtype=bool) if return_log_probs else None
        return_context_logits_data = np.array([[return_context_logits]], dtype=bool) if return_context_logits else None
        return_generation_logits_data = np.array([[return_generation_logits]], dtype=bool) if return_generation_logits else None
        

        draft_ids_data = np.array(draft_ids, dtype=np.int32) if draft_ids else None

        return_log_probs_data = np.array([[return_log_probs]], dtype=bool)

        inputs = prepare_inputs(
            input_ids_data, input_lengths_data, request_output_len_data,
            beam_width_data, temperature_data, repetition_penalty_data,
            presence_penalty_data, frequency_penalty_data, streaming_data,
            end_id_data, pad_id_data, prompt_embedding_table_data,
            prompt_vocab_size_data, lora_weights_data, lora_config_data,
            return_log_probs_data, top_k_data, top_p_data, draft_ids_data,
            return_context_logits_data, return_generation_logits_data
        )

        return inputs
    
    def run(
            self,
            text,
            output_len=26,
            top_k=1,
            top_p=0.0,
            temperature=1.0,
            beam_width=1,
            repetition_penalty=None,
            presence_penalty=None,
            frequency_penalty=None,
            # currently unsupported inputs
            prompt_embedding_table_data=None,
            prompt_vocab_size_data=None,
            lora_weights_data=None,
            lora_config_data=None,
            return_log_probs=None,
            return_context_logits=None,
            return_generation_logits=None,
            draft_ids=None,
        ):


        inputs = self._preprocess_request_parameters(
            text,
            output_len=output_len,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            beam_width=beam_width,
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            prompt_embedding_table_data=prompt_embedding_table_data,
            prompt_vocab_size_data=prompt_vocab_size_data,
            lora_weights_data=lora_weights_data,
            lora_config_data=lora_config_data,
            return_log_probs=return_log_probs,
            return_context_logits=return_context_logits,
            return_generation_logits=return_generation_logits,
            draft_ids=draft_ids,
        )

        stop_inputs = None
        request_id = self.generate_request_id()


        sequence_lengths = []
        cum_log_probs = None
        output_log_probs = None
        context_logits = None
        generation_logits = None

        user_data = UserData() 
        
        actual_output_ids = [[]]
        with grpcclient.InferenceServerClient(
                url=self.url,
                verbose=self.verbose,
                ssl=self.ssl,
                root_certificates=self.root_certificates,
                private_key=self.private_key,
                certificate_chain=self.certificate_chain,
        ) as triton_client:
            
            try:
                # Establish stream
                triton_client.start_stream(
                    callback=partial(callback, user_data, self.tokenizer),
                    stream_timeout=self.stream_timeout,
                )
                # Send request
                # code for this method is here: https://github.com/triton-inference-server/client/blob/24c1ff7969e4f8f9e31a5c98237b6c1b5972bfca/src/python/library/tritonclient/grpc/_client.py#L1815
                triton_client.async_stream_infer(
                    'tensorrt_llm',
                    inputs,
                    request_id=request_id,
                )

                # Send stop signal
                # triton_client.stop_stream(cancel_requests=cancel_requests)

                # Parse the responses
                while True:

                    try:
                        result = user_data._completed_requests.get(block=False)
                        decoded_token = user_data._decoded_tokens.get(block=False)
                        yield decoded_token

                    except queue.Empty:
                        break
                    
                    if type(result) == InferenceServerException:
                        if result.status() == "StatusCode.CANCELLED":
                            print("Request is cancelled")
                        else:
                            print("Received an error from server:")
                            print(result)
                            raise result
            except Exception as e:
                err = "Encountered error: " + str(e)
                print(err)
                sys.exit(err)


    @staticmethod
    def generate_request_id():
        return str(uuid.uuid4())


        


        
if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(
        "/src/tensorrtllm_backend/tensorrt_llm/examples/gpt/gpt2",
        padding_side='left',
        trust_remote_code=True
    )

    triton_trt_llm_client = TRTLLMClient(
        tokenizer=tokenizer,
    )

    triton_trt_llm_client.run(
        text="This is fucking dumb wtf"
    )    



