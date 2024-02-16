#!/usr/bin/env python3

"""
Generate Triton config files from an updated YAML configuration format.
"""

import os
import yaml
from argparse import ArgumentParser
from string import Template

DEFAULTS = {
    "tokenizer_dir": "/src/triton_model_repo/tensorrt_llm/1",
    "engine_dir": "/src/triton_model_repo/tensorrt_llm/1",
}


def apply_template(template_path, substitutions):
    with open(template_path) as f:
        template = Template(f.read())
    # Apply default values if not overridden in substitutions
    for key, default_value in DEFAULTS.items():
        if key not in substitutions:
            substitutions[key] = default_value
    return template.safe_substitute(substitutions)


def process_server_config(server_config, templates_dir, output_dir):
    for model, params in server_config.items():
        template_file = os.path.join(templates_dir, f"{model}/config.pbtxt")
        output_file = os.path.join(output_dir, f"{model}/config.pbtxt")

        if not os.path.exists(os.path.dirname(output_file)):
            os.makedirs(os.path.dirname(output_file))

        # Apply template substitutions
        pbtxt = apply_template(template_file, params)

        # Write the result
        with open(output_file, "w") as f:
            f.write(pbtxt)
        print(f"Generated {output_file}")


def main(yaml_config, templates_dir, output_dir):
    with open(yaml_config) as f:
        config = yaml.safe_load(f)

    # Determine if 'server' is a top-level key or nested under 'instantiate'
    server_config = config.get(
        "server", config.get("instantiate", {}).get("server", {})
    )

    if not server_config:
        raise ValueError("No 'server' configuration found in YAML file")

    process_server_config(server_config, templates_dir, output_dir)


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Generate Triton config files from YAML configuration."
    )
    parser.add_argument(
        "--yaml_config",
        default="config.yaml",
        help="Path to the YAML configuration file.",
        required=False,  # This makes the argument optional.
    )
    parser.add_argument(
        "--templates_dir",
        default="./triton_templates/",
        help="Directory containing the .pbtxt template files.",
        required=False,  # This makes the argument optional.
    )
    parser.add_argument(
        "--output_dir",
        default="./triton_model_repo",
        help="Target directory for generated .pbtxt files.",
        required=False,  # This makes the argument optional.
    )
    args = parser.parse_args()

    main(args.yaml_config, args.templates_dir, args.output_dir)
