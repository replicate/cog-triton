# if ./triton_model_repo doesn't exist, create it and copy triton_templates to it
if [ ! -d "./triton_model_repo" ]; then
    mkdir ./triton_model_repo
    mkdir -p ./triton_templates
    # Copy current templates from tensorrtllm_backend, we don't want ours to get stale
    cp -r tensorrtllm_backend/all_models/inflight_batcher_llm ./triton_templates 
    # copy model components to our target model directory
    cp -r ./triton_templates/ensemble triton_model_repo/
    cp -r ./triton_templates/preprocessing triton_model_repo/
    cp -r ./triton_templates//postprocessing triton_model_repo/
    cp -r ./triton_templates/tensorrt_llm triton_model_repo/
    cp -r ./triton_templates/tensorrt_llm_bls triton_model_repo/
fi

# Use the triton_fill_template.py script to generate the config.pbtxt files, overwriting the defaults
# we copied.

# get tokenizer_dir from TOKENIZER_DIR env variable, set default to /src/triton_model_repo/tensorrt_llm/1
TOKENIZER_DIR=${TOKENIZER_DIR:-/src/triton_model_repo/tensorrt_llm/1/}
MAX_BATCH_SIZE=${MAX_BATCH_SIZE:-64}

# Generate preprocessing config
python3 scripts/triton_fill_template.py triton_templates/preprocessing/config.pbtxt \
     "triton_max_batch_size:${MAX_BATCH_SIZE},tokenizer_dir:${TOKENIZER_DIR},tokenizer_type:auto,preprocessing_instance_count:${MAX_BATCH_SIZE:-64}" \
     > ./triton_model_repo/preprocessing/config.pbtxt

# Generate tensorrt_llm config
python3 scripts/triton_fill_template.py triton_templates/tensorrt_llm/config.pbtxt \
    "enable_kv_cache_reuse:true,triton_max_batch_size:${MAX_BATCH_SIZE},decoupled_mode:True,engine_dir:/src/triton_model_repo/tensorrt_llm/1,batching_strategy:inflight_fused_batching,batch_scheduler_policy:max_utilization,max_queue_delay_microseconds:100" \
     > ./triton_model_repo/tensorrt_llm/config.pbtxt

# Generate postprocessing config
python3 scripts/triton_fill_template.py triton_templates/postprocessing/config.pbtxt \
    "triton_max_batch_size:${MAX_BATCH_SIZE},tokenizer_dir:${TOKENIZER_DIR},tokenizer_type:auto,postprocessing_instance_count:${MAX_BATCH_SIZE:-64}" \
     > ./triton_model_repo/postprocessing/config.pbtxt

# Generate ensemble config
python3 scripts/triton_fill_template.py triton_templates/ensemble/config.pbtxt \
    "triton_max_batch_size:${MAX_BATCH_SIZE}" \
     > ./triton_model_repo/ensemble/config.pbtxt

# Generate tensorrt_llm_bls config
python3 scripts/triton_fill_template.py triton_templates/ensemble/config.pbtxt \
    "triton_max_batch_size:${MAX_BATCH_SIZE},decoupled_mode:True,bls_instance_count:${MAX_BATCH_SIZE},accumulate_tokens:true" \
     > ./triton_model_repo/ensemble/config.pbtxt