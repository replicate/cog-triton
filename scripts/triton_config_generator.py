#!/usr/bin/env python3
"""
Fill Triton config template based on YAML configuration.
"""
import argparse
import yaml
from string import Template
from pathlib import Path


def get_server_config(config):
    """
    Retrieves the server configuration:
       - If the 'instantiate' key is present, it will look for the server configuration in the 'server' key of the 'instantiate' dictionary.
         If the 'server' key is not present, it will return None and default configs will be used.
       - If the 'instantiate' key is not present, assumes that the config is just a triton server config.

    Args:
        config (dict): The configuration dictionary.

    Returns:
        dict or None: The server configuration if found, None otherwise.
    """
    if 'instantiate' in config:
        if 'server' in config['instantiate']:
            return config['instantiate']['server']
        else:
            return None
    else:
        return config

def load_yaml_config(yaml_file):
    with open(yaml_file, 'r') as f:
        config = yaml.safe_load(f)
        return get_server_config(config)

def fill_template(template_file, substitutions):
    with open(template_file, 'r') as f:
        template = Template(f.read())
    return template.safe_substitute(substitutions)


def write_config(config, output_file):
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        f.write(config)


def get_config_paths(model_config, model):
    template_file = model_config.get('template', f'triton_templates/{model}/config.pbtxt')
    output_file = model_config.get('output', f'./triton_model_repo/{model}/config.pbtxt')
    return template_file, output_file


def get_default_args(model):

    ensemble_args = {}

    preprocessing_args = {
        'tokenizer_dir': '/src/triton_model_repo/tensorrt_llm/1/',
        'tokenizer_type': 'auto',
    }
    postprocessing_args = {
        'tokenizer_dir': '/src/triton_model_repo/tensorrt_llm/1/',
        'tokenizer_type': 'auto',
    }
    tensorrt_llm_args = {
        'engine_dir': '/src/triton_model_repo/tensorrt_llm/1/',
        'decoupled_mode': 'True',
        'batching_strategy': 'inflight_fused_batching',
        'max_queue_delay_microseconds': 100,
        'batch_scheduler_policy': 'max_utilization'
    }
    tensorrt_llm_bls_args = {
        'decoupled_mode': 'True',
        'accumulate_tokens': 'true',
    }

    default_args = {
        'preprocessing': preprocessing_args,
        'postprocessing': postprocessing_args,
        'tensorrt_llm': tensorrt_llm_args,
        'tensorrt_llm_bls': tensorrt_llm_bls_args,
        'ensemble': ensemble_args
    }


    if model in default_args:
        return default_args[model]
    else:
        raise ValueError(f"Model {model} not found in default args. Must be one of {list(default_args.keys())}")

def populate_max_batch_size_and_instance_count(config, max_batch_size, default_models):
    for model in default_models:
        if model not in config:
            config[model] = {'args': {}}

        config[model]['args']['triton_max_batch_size'] = max_batch_size
        
        if model in ["ensemble", "tensorrt_llm"]:
            continue
        elif model == 'tensorrt_llm_bls':
            instance_count_key = "bls_instance_count"
        else:
            instance_count_key = f"{model}_instance_count"

        config[model]['args'][instance_count_key] = max_batch_size

    return config

def main(yaml_file):
    config = load_yaml_config(yaml_file)
    max_batch_size = config.get('max_batch_size')
    default_models = ['preprocessing', 'tensorrt_llm', 'postprocessing', 'ensemble', 'tensorrt_llm_bls']

     
    if max_batch_size:
        config = populate_max_batch_size_and_instance_count(config, max_batch_size, default_models)


    for model in default_models:
        if model not in config:
            config[model] = {'args': {}}

        model_config = config[model]

        template_file, output_file = get_config_paths(model_config, model)

        default_args = get_default_args(model)
        substitutions = {**default_args, **model_config.get('args', {})}

        filled_config = fill_template(template_file, substitutions)
        write_config(filled_config, output_file)
    
    print("Serving with config:")
    print(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fill Triton config templates based on YAML configuration.')
    parser.add_argument('yaml_file', help='Path to the YAML configuration file.')
    args = parser.parse_args()

    main(args.yaml_file)