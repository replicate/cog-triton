# Copyright 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import json
import traceback
import time
import numpy as np
import triton_python_backend_utils as pb_utils
from transformers import LlamaTokenizerFast, GPT2TokenizerFast

class TritonPythonModel:

    def initialize(self, args):

        self.output_dtype = pb_utils.triton_string_to_numpy("TYPE_STRING")

        # Parse model configs
        model_config = json.loads(args['model_config'])

        params = model_config['parameters']

        accumulate_tokens_str = ''
        if 'accumulate_tokens' in params:
            accumulate_tokens_str = params['accumulate_tokens']['string_value']

        self.accumulate_tokens = accumulate_tokens_str.lower() in [
            'true', 'yes', '1', 't'
        ]

        self.decoupled = pb_utils.using_decoupled_model_transaction_policy(
            model_config)

        self.logger = pb_utils.Logger

        self.bls_input_tensor_names = [
            "text_input", "max_tokens", "bad_words", "stop_words", "end_id",
            "pad_id", "top_k", "top_p", "temperature", "length_penalty",
            "repetition_penalty", "min_length", "presence_penalty",
            "frequency_penalty", "random_seed", "return_log_probs",
            "return_context_logits", "return_generation_logits", "beam_width",
            "stream", "prompt_embedding_table", "prompt_vocab_size",
            "embedding_bias_words", "embedding_bias_weights"
        ]

        self.preproc_input_to_bls_input_map = {
            "QUERY": "text_input",
            "REQUEST_OUTPUT_LEN": "max_tokens",
            "BAD_WORDS_DICT": "bad_words",
            "STOP_WORDS_DICT": "stop_words",
            "EMBEDDING_BIAS_WORDS": "embedding_bias_words",
            "EMBEDDING_BIAS_WEIGHTS": "embedding_bias_weights",
            "END_ID": "end_id",
            "PAD_ID": "pad_id"
        }

        self.preproc_output_to_trtllm_input_map = {
            "INPUT_ID": "input_ids",
            "REQUEST_INPUT_LEN": "input_lengths",
            "REQUEST_OUTPUT_LEN": "request_output_len",
            "BAD_WORDS_IDS": "bad_words_list",
            "STOP_WORDS_IDS": "stop_words_list",
            "EMBEDDING_BIAS": "embedding_bias",
            "OUT_END_ID": "end_id",
            "OUT_PAD_ID": "pad_id",
        }

        self.trtllm_input_to_bls_input_map = {
            "beam_width": "beam_width",
            "runtime_top_k": "top_k",
            "runtime_top_p": "top_p",
            "len_penalty": "length_penalty",
            "repetition_penalty": "repetition_penalty",
            "min_length": "min_length",
            "presence_penalty": "presence_penalty",
            "frequency_penalty": "frequency_penalty",
            "random_seed": "random_seed",
            "return_log_probs": "return_log_probs",
            "return_context_logits": "return_context_logits",
            "return_generation_logits": "return_generation_logits",
            "streaming": "stream",
            "prompt_embedding_table": "prompt_embedding_table",
            "prompt_vocab_size": "prompt_vocab_size",
        }

        self.trtllm_output_to_postproc_input_map = {
            "output_ids": "TOKENS_BATCH",
            "sequence_length": "SEQUENCE_LENGTH",
            "cum_log_probs": "CUM_LOG_PROBS",
            "output_log_probs": "OUTPUT_LOG_PROBS",
            "context_logits": "CONTEXT_LOGITS",
            "generation_logits": "GENERATION_LOGITS"
        }

        self.postproc_output_to_bls_output_map = {
            "OUTPUT": "text_output",
            "OUT_CUM_LOG_PROBS": "cum_log_probs",
            "OUT_OUTPUT_LOG_PROBS": "output_log_probs",
            "OUT_CONTEXT_LOGITS": "context_logits",
            "OUT_GENERATION_LOGITS": "generation_logits"
        }

    def _get_bls_input_tensors_map(self, request):

        bls_input_tensors_map = {}
        for input_tensor_name in self.bls_input_tensor_names:
            tensor = pb_utils.get_input_tensor_by_name(request,
                                                       input_tensor_name)
            if tensor != None:
                bls_input_tensors_map[input_tensor_name] = tensor

        return bls_input_tensors_map

    def _get_preproc_input_tensors(self, bls_input_tensors_map):

        preproc_input_tensors = []

        for preproc_name, bls_name in self.preproc_input_to_bls_input_map.items(
        ):

            if bls_name in bls_input_tensors_map:
                tensor = bls_input_tensors_map[bls_name]
                # Change the name to what the preprocessor expects
                preproc_input_tensors.append(
                    pb_utils.Tensor(preproc_name, tensor.as_numpy()))

        return preproc_input_tensors

    def _get_trtllm_input_tensors(self, bls_input_tensors_map,
                                  preproc_output_tensors):

        trtllm_input_tensors = []

        # Set input tensors from preprocessor outputs
        for preproc_output_tensor in preproc_output_tensors:

            trtllm_tensor_name = self.preproc_output_to_trtllm_input_map[
                preproc_output_tensor.name()]
            trtllm_input_tensors.append(
                pb_utils.Tensor(trtllm_tensor_name,
                                preproc_output_tensor.as_numpy()))

        # Set input tensors from bls inputs
        for trtllm_name, bls_name in self.trtllm_input_to_bls_input_map.items(
        ):

            if bls_name in bls_input_tensors_map:
                tensor = bls_input_tensors_map[bls_name]
                # Change the name to what the preprocessor expects
                trtllm_input_tensors.append(
                    pb_utils.Tensor(trtllm_name, tensor.as_numpy()))

        return trtllm_input_tensors

    def _get_postproc_input_tensors(self, tokens, trtllm_output_tensors):

        postproc_input_tensors = []

        for trtllm_output_tensor in trtllm_output_tensors:

            # If in decoupled mode, option to append new tokens to existing tokens before calling postprocessor
            # This might be needed for some tokenizers
            # Note that in that case, the client must overwrite previously received output text
            if (self.accumulate_tokens and self.decoupled
                    and trtllm_output_tensor.name() == "output_ids"):

                new_tokens = trtllm_output_tensor.as_numpy()
                if new_tokens.ndim != 3:
                    raise pb_utils.TritonModelException(
                        "Expected output_ids tensor to have 3 dims.")
                if new_tokens.shape[0] != 1:
                    raise pb_utils.TritonModelException(
                        "Expected output_ids tensor to have batch size of 1")
                if new_tokens.shape[1] != 1:
                    raise pb_utils.TritonModelException(
                        "Accumulation of tokens is only implemented for beam width = 1"
                    )

                tokens = new_tokens if (tokens is None) else np.concatenate(
                    (tokens, new_tokens), axis=2)

                # output ids
                postproc_output_ids_name = self.trtllm_output_to_postproc_input_map[
                    "output_ids"]
                postproc_input_tensors.append(
                    pb_utils.Tensor(postproc_output_ids_name, tokens))

                # sequence length
                np_seq_len_tensor = np.array([[tokens.shape[2]]],
                                             dtype=np.int32)
                postproc_seq_len_name = self.trtllm_output_to_postproc_input_map[
                    "sequence_length"]
                postproc_input_tensors.append(
                    pb_utils.Tensor(postproc_seq_len_name, np_seq_len_tensor))

        # Set input tensors from trtllm outputs
        for trtllm_output_tensor in trtllm_output_tensors:

            # output_ids and sequence_length were handled earlier
            if (self.accumulate_tokens and self.decoupled
                    and (trtllm_output_tensor.name() == "output_ids"
                         or trtllm_output_tensor.name() == "sequence_length")):
                continue

            postproc_tensor_name = self.trtllm_output_to_postproc_input_map[
                trtllm_output_tensor.name()]

            postproc_input_tensors.append(
                pb_utils.Tensor(postproc_tensor_name,
                                trtllm_output_tensor.as_numpy()))

        return tokens, postproc_input_tensors

    def _get_bls_output_tensors(self, postproc_output_tensors):

        bls_output_tensors = []

        # Set input tensors from trtllm outputs
        for postproc_output_tensor in postproc_output_tensors:

            bls_tensor_name = self.postproc_output_to_bls_output_map[
                postproc_output_tensor.name()]
            bls_output_tensors.append(
                pb_utils.Tensor(bls_tensor_name,
                                postproc_output_tensor.as_numpy()))

        return bls_output_tensors
    
    def _postprocessing(self, tokens_batch, sequence_lengths):
        start = time.time()
        outputs = []
        for batch_idx, beam_tokens in enumerate(tokens_batch):
            for beam_idx, tokens in enumerate(beam_tokens):
                inner_loop_time = time.time()
                seq_len = sequence_lengths[batch_idx][beam_idx]
                tokens_to_decode = tokens[:seq_len]
                tokenizer_start_time = time.time()
                output = self.tokenizer.decode(
                    tokens_to_decode,
                    skip_special_tokens=True)
                tokenizer_output_time = time.time()
                outputs.append(output.encode('utf8'))
                end_inner_loop = time.time()
        
        end = time.time()
        # print(f"Total time: {end - start}")
        # print(f"Tokenizer time: {tokenizer_output_time - tokenizer_start_time}")
        # print(f"Inner loop time: {end_inner_loop - inner_loop_time}")
        # print("n tokens to decode", len(tokens[:seq_len]))
        return outputs

    def execute(self, requests):
        bls_start_time = time.time()
        responses = []
        bls_response_sender = None

        for request in requests:

            #Get the response sender for the BLS
            if self.decoupled:
                bls_response_sender = request.get_response_sender()

            try:
                # Get the bls input tensors
                bls_input_tensors_map = self._get_bls_input_tensors_map(
                    request)

                #Check the batch dimension
                for name, tensor in bls_input_tensors_map.items():
                    batch_dim = tensor.as_numpy().shape[0]

                    if batch_dim != 1:

                        err_str = "Inflight batching backend expects requests with batch size of 1."
                        self.logger.log_error(err_str)
                        raise pb_utils.TritonModelException(err_str)

                # Create the preprocessor input tensors
                preproc_input_tensors = self._get_preproc_input_tensors(
                    bls_input_tensors_map)

                preproc_request = pb_utils.InferenceRequest(
                    model_name="preprocessing",
                    inputs=preproc_input_tensors,
                    requested_output_names=list(
                        self.preproc_output_to_trtllm_input_map.keys()))

                #Execute preprocessor
                bls_preproc_start_time = time.time()
                preproc_response = preproc_request.exec()
                print(f"Preprocessing Execution Time: {time.time() - bls_preproc_start_time}")

                if preproc_response.has_error():
                    raise pb_utils.TritonModelException(
                        preproc_response.error().message())

                # Create the trtllm input tensors
                trtllm_input_tensors = self._get_trtllm_input_tensors(
                    bls_input_tensors_map, preproc_response.output_tensors())

                trtllm_request = pb_utils.InferenceRequest(
                    model_name="tensorrt_llm",
                    inputs=trtllm_input_tensors,
                    requested_output_names=list(
                        self.trtllm_output_to_postproc_input_map.keys()))

                #Execute trtllm
                trtllm_responses = trtllm_request.exec(
                    decoupled=self.decoupled)

                if not self.decoupled:
                    trtllm_responses = [trtllm_responses]

                tokens = None
                

                #Loop over the trtllm responses
                for trtllm_response in trtllm_responses:

                    if trtllm_response.has_error():
                        raise pb_utils.TritonModelException(
                            trtllm_response.error().message())

                    trtllm_output_tensors = trtllm_response.output_tensors()

                    tokens, postproc_input_tensors = self._get_postproc_input_tensors(
                        tokens, trtllm_output_tensors)
                    
                    sequence_lengths = np.array([[tokens.shape[2]]],
                                             dtype=np.int32)
                    
                    
                    # sequence_lengths = pb_utils.get_input_tensor_by_name(
                    #     request, 'SEQUENCE_LENGTH').as_numpy()
                    
                    # sequence_lengths = postproc_input_tensors[1].as_numpy()
                    outputs = self._postprocessing(tokens, sequence_lengths)

                    # print(f"Tokens type: {type(tokens)}")

                    # postproc_request = pb_utils.InferenceRequest(
                    #     model_name="postprocessing",
                    #     inputs=postproc_input_tensors,
                    #     requested_output_names=list(
                    #         self.postproc_output_to_bls_output_map.keys()))

                    # #Execute postprocessor
                    # start_time = time.time()
                    # postproc_response = postproc_request.exec()
                    # print(f"BLS Execution Time: {time.time() - start_time}")
                    # if postproc_response.has_error():
                    #     raise pb_utils.TritonModelException(
                    #         postproc_response.error().message())

                    # Create the BLS response
                    # bls_output_tensors = self._get_bls_output_tensors(
                    #     postproc_response.output_tensors())

                    # bls_response = pb_utils.InferenceResponse(
                    #     output_tensors=bls_output_tensors)

                    responses = []

                    outputs = np.array(outputs).astype(self.output_dtype)

                    output_tensor = pb_utils.Tensor(
                        'text_output',
                        outputs)
                    
                    
                    bls_response = pb_utils.InferenceResponse(output_tensors=[
                        output_tensor, #out_cum_log_probs, out_output_log_probs,
                        #out_context_logits, out_generation_logits
                    ])
                    responses.append(bls_response)

                    if self.decoupled:
                        bls_response_sender.send(bls_response)
                    else:
                        responses.append(bls_response)

                # All responses have been sent, set final flag
                if self.decoupled:
                    bls_response_sender.send(
                        flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL)
                
                print(f"BLS Execution Time: {time.time() - bls_start_time}")

            except Exception:

                self.logger.log_error(traceback.format_exc())
                # If encountering an error, send a response with err msg
                error_response = pb_utils.InferenceResponse(
                    output_tensors=[],
                    error=pb_utils.TritonError(traceback.format_exc()))

                if self.decoupled:
                    bls_response_sender.send(error_response)
                    bls_response_sender.send(
                        flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL)
                else:
                    responses.append(error_response)

        if self.decoupled:
            return None
        else:
            assert len(responses) == len(requests)
            return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is optional. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print('Cleaning up...')
