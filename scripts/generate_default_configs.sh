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
fi

# Use the triton_fill_template.py script to generate the config.pbtxt files, overwriting the defaults
# we copied.

# Generate preprocessing config
python3 scripts/triton_fill_template.py triton_templates/preprocessing/config.pbtxt \
     "triton_max_batch_size:4,tokenizer_dir:/src/gpt2/,tokenizer_type:auto,preprocessing_instance_count:1" \
     > ./triton_model_repo/preprocessing/config.pbtxt

# Generate tensorrt_llm config
python3 scripts/triton_fill_template.py triton_templates/tensorrt_llm/config.pbtxt \
    "triton_max_batch_size:4,decoupled_mode:True,engine_dir:/src/triton_model_repo/tensorrt_llm/1,batching_strategy:inflight_fused_batching,batch_scheduler_policy:max_utilization,max_queue_delay_microseconds:100" \
     > ./triton_model_repo/tensorrt_llm/config.pbtxt

# Generate postprocessing config
python3 scripts/triton_fill_template.py triton_templates/postprocessing/config.pbtxt \
    "triton_max_batch_size:4,tokenizer_dir:/src/gpt2/,tokenizer_type:auto,postprocessing_instance_count:1" \
     > ./triton_model_repo/postprocessing/config.pbtxt

# Generate ensemble config
python3 scripts/triton_fill_template.py triton_templates/ensemble/config.pbtxt \
    "triton_max_batch_size:4" \
     > ./triton_model_repo/ensemble/config.pbtxt