import tempfile
import sys
sys.path.append(".")

from triton_config_generator import generate_configs, load_yaml_config
 
def test_load_yaml_config():
    config = load_yaml_config("./tests/assets/default_triton_config.yaml")
    assert config == {'preprocessing': {'args': {'tokenizer_dir': '/src/triton_model_repo/tensorrt_llm/1/', 'tokenizer_type': 'auto', 'triton_max_batch_size': 64, 'preprocessing_instance_count': 64}}, 'tensorrt_llm': {'args': {'engine_dir': '/src/triton_model_repo/tensorrt_llm/1/', 'triton_max_batch_size': 64, 'decoupled_mode': True, 'batching_strategy': 'inflight_fused_batching', 'batch_scheduler_policy': 'max_utilization', 'max_queue_delay_microseconds': 100}}, 'postprocessing': {'args': {'tokenizer_dir': '/src/triton_model_repo/tensorrt_llm/1/', 'tokenizer_type': 'llama', 'triton_max_batch_size': 64, 'postprocessing_instance_count': 64}}, 'ensemble': {'args': {'triton_max_batch_size': 64}}, 'tensorrt_llm_bls': {'args': {'triton_max_batch_size': 64, 'decoupled_mode': True, 'bls_instance_count': 64, 'accumulate_tokens': 'true'}}}

def test_generate_configs():
    with tempfile.TemporaryDirectory() as temp_dir:
        config = load_yaml_config("./tests/assets/default_triton_config.yaml")
        for key in config:
            config[key]['template'] = f'./tests/assets/triton_templates/{key}/config.pbtxt'
            config[key]['output'] = f'{temp_dir}/triton_model_repo/{key}/config.pbtxt'
    
        generate_configs(config)

        # check if outputs match ./tests/assets/triton_model_repo/{key}/config.pbtxt
        for key in config:
            with open(f'./tests/assets/triton_model_repo/{key}/config.pbtxt', 'r') as f:
                expected_output = f.read()
            with open(f'{temp_dir}/triton_model_repo/{key}/config.pbtxt', 'r') as f:
                output = f.read()
            assert output == expected_output

    assert True