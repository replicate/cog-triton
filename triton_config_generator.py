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
        return yaml.safe_load(f)
        # return get_server_config(config)

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


def generate_configs(config):
    models = ['preprocessing', 'tensorrt_llm', 'postprocessing', 'ensemble', 'tensorrt_llm_bls']

    for model in models:
        if model not in config:
            continue

        model_config = config[model]

        template_file, output_file = get_config_paths(model_config, model)

        substitutions = model_config.get('args', {})

        filled_config = fill_template(template_file, substitutions)
        write_config(filled_config, output_file)

def main(yaml_file):
    config = load_yaml_config(yaml_file)
    generate_configs(config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fill Triton config templates based on YAML configuration.')
    parser.add_argument('yaml_file', help='Path to the YAML configuration file.')
    args = parser.parse_args()

    main(args.yaml_file)