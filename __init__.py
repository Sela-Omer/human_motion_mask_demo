import argparse

import torch

from src.config.config import config
from src.helper.param_helper import convert_param_to_list
from src.service.service_eval import ServiceEval
from src.service.service_fit import ServiceFit


def parse_arguments(config):
    """
    Generate parser for command line arguments based on config.ini sections and options.

    Args:
        config (configparser.ConfigParser): The configparser object containing the configuration.

    Returns:
        argparse.Namespace: The parsed command line arguments.

    """
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description="Override config.ini settings")

    # Iterate over the sections and options in the config
    for section in config.sections():
        for key in config[section]:
            # Generate a command line argument for each config setting
            arg_name = f"--{section}_{key}"
            help_msg = f"Override {section} {key}"
            parser.add_argument(arg_name, type=str, help=help_msg)

    # Parse the command line arguments
    return parser.parse_args()


def override_config(config, args):
    """
    Override config.ini settings with any specified command line arguments.

    Args:
        config (ConfigParser): The config parser object.
        args (Namespace): The command line arguments.

    Returns:
        None
    """
    # Iterate over the command line arguments
    for arg_key, arg_value in vars(args).items():
        # Check if the argument value is not None
        if arg_value is not None:
            # Split the argument key to get the section and key
            arg_lst = arg_key.split('_')
            found_arg = False
            for i in range(1, len(arg_lst)):
                a1 = '_'.join(arg_lst[:i])
                a2 = '_'.join(arg_lst[i:])
                if config.has_option(a1, a2):
                    config.set(a1, a2, arg_value)
                    found_arg = True
            if not found_arg:
                raise ValueError(f"Invalid command line argument was passed. {arg_key} does not exist in config.ini")


if __name__ == "__main__":
    # Print the number of GPUs and their names
    print(f"cuda is available: {torch.cuda.is_available()}")
    print(f"cuda is initialized: {torch.cuda.is_initialized()}")
    if not torch.cuda.is_initialized():
        print("initializing manually...")
        torch.cuda.init()
        print(f"cuda is initialized: {torch.cuda.is_initialized()}")

    gpus = torch.cuda.device_count()
    print(f"Number of GPUs: {gpus}")
    gpu_names = [torch.cuda.get_device_name(i) for i in range(gpus)]
    print(f"GPU names: {gpu_names}")

    # Parse the command line arguments
    args = parse_arguments(config)
    # Override the config settings with the command line arguments
    override_config(config, args)

    # Convert the APP.MODE parameter to a list
    app_mode_lst = convert_param_to_list(config['APP']['MODE'])
    # Get the APP.ARCH parameter
    app_arch = config['APP']['ARCH']

    # Create a dictionary of service objects
    service_dict = {
        "FIT": ServiceFit,
        "EVAL": ServiceEval,
    }
    for app_mode in app_mode_lst:
        # Create the service object
        service = service_dict[app_mode](config)

        # Run the script
        script = service.scripts[app_arch]
        script()
