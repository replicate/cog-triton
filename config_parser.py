import yaml
from omegaconf import OmegaConf
from pygments import highlight
from pygments.lexers import YamlLexer
from pygments.formatters import TerminalFormatter


# read yaml with omegaconf
class ConfigParser:
    def load_config(self, config_path: str) -> dict:
        """Load a config file from a path.

        Args:
            config_path (str): Path to the config file.

        Returns:
            dict: Dictionary containing the config.
        """
        config = {}
        if config_path:
            with open(config_path, "r") as f:
                config = OmegaConf.load(f)

        return config

    def update_config(self, config: dict, **kwargs) -> dict:
        """Update a config with kwargs.

        Args:
            config (dict): Dictionary containing the config.
            **kwargs: Key-value pairs to update the config with.

        Returns:
            dict: Dictionary containing the updated config.
        """
        drop_keys = []
        for k, v in kwargs.items():
            if k in config and v is not None:
                raise ValueError(
                    f"You specified '{k}' as an input argument and in your config, this is not allowed."
                )
            elif v is None:
                drop_keys.append(k)

        for k in drop_keys:
            kwargs.pop(k)

        config.update(kwargs, merge=True)
        return config

    def print_config(self, config) -> None:
        # Convert the configuration to a YAML formatted string
        yaml_str = OmegaConf.to_yaml(config)

        # Highlight the YAML string
        highlighted_str = highlight(yaml_str, YamlLexer(), TerminalFormatter())

        # Print the highlighted string
        print("=========================================")
        print("Using the following config:\n\n")

        print(highlighted_str)
