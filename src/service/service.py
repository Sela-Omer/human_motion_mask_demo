from abc import ABC, abstractmethod
from typing import Callable, Dict


class Service(ABC):
    """
    A class representing a service.
    :arg config: The configuration file for the service.

    """

    def __init__(self, config):
        self.config = config
        override_config_file = self.config['APP']['OVERWRITE_CONFIG_PATH']
        override_config_filename = override_config_file.split('/')[-1].split('.')[0]
        self.model_name = f"{self.config['APP']['ARCH']}-{override_config_filename}"

    @property
    @abstractmethod
    def scripts(self) -> Dict[str, Callable]:
        """
        Returns a dictionary of scripts.
        The dictionary contains a single key-value pair.
        The key is the name of the script, and the value is a function that implements the script.
        Returns:
            Dict[str, Callable]: A dictionary of scripts.
        """
        pass