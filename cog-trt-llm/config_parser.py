import yaml
from omegaconf import OmegaConf


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
        yaml_str = OmegaConf.to_yaml(config, resolve=True)

        # Print the YAML string with indents and no fancy formatting
        print("=========================================")
        print("You submitted this config:\n")
        print("=========================================\n\n")

        # Print the YAML string with indents and no fancy formatting
        print(yaml_str)
        print("=========================================\n\n")
