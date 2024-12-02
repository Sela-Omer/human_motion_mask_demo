from typing import Callable, Dict

from src.script.fit_unet_1d_script import FitUnet1dScript
from src.service.service import Service


class ServiceFit(Service):
    """
    This class is responsible for initializing a fit service.

    """

    @property
    def scripts(self) -> Dict[str, Callable]:
        """
        Returns a dictionary of scripts.

        The dictionary contains a single key-value pair.
        The key is the name of the ARCH , and the value is an instance of the Fit Script class.
        """
        # Create a dictionary with the ARCH name and its corresponding instance
        script_dict = {
            'UNET_1D': FitUnet1dScript(self),
        }

        return script_dict
