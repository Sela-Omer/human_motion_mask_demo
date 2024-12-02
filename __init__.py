import argparse

import torch

from src.config.config import config
from src.helper.param_helper import convert_param_to_list
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
    }
    for app_mode in app_mode_lst:
        # Create the service object
        service = service_dict[app_mode](config)

        # Run the script
        script = service.scripts[app_arch]
        script()

#
# import pytorch_lightning as pl
# import torch
# from pytorch_lightning.callbacks import ModelCheckpoint
# from pytorch_lightning.loggers import CSVLogger
#
# from old_c import LOG_DIR, TRAIN_DIR, VALID_DIR, TEST_DIR, BATCH_SIZE
# from src.datamodule.amass_data_module import AMASSDataModule
# from src.ligtning_module.unet_1d import UNet1D
#
# if __name__ == '__main__':
#     # # Initialize the TransformerVAE model
#     # model = TransformerVAE(input_dim=156 + 52, output_dim=156, latent_dim=256,
#     #                        num_frames=frames_per_sample)  # Adjust as needed
#     data_module = AMASSDataModule(TRAIN_DIR, VALID_DIR, TEST_DIR, batch_size=BATCH_SIZE)
#     data_module.prepare_data()
#     data_module.setup()
#
#     tdl = data_module.train_dataloader()
#     vdl = data_module.val_dataloader()
#
#
#     model = UNet1D(in_channels=156 + 52, out_channels=156)
#
#     # Configure the Trainer
#     logger = CSVLogger(save_dir=LOG_DIR, name="TransformerVAE")
#     checkpoint_callback = ModelCheckpoint(
#         monitor="train_loss",  # Monitor training loss for saving the best model
#         save_top_k=1,
#         mode="min",
#         filename="transformer_vae-{epoch:02d}-{train_loss:.4f}",
#     )
#     trainer = pl.Trainer(
#         max_epochs=10,
#         logger=logger,
#         callbacks=[checkpoint_callback],
#         accelerator="gpu" if torch.cuda.is_available() else "cpu",
#         devices=1,
#         log_every_n_steps=1,  # Print logs for every batch
#         enable_progress_bar=True,  # Progress will be shown in the notebook
#     )
#
#
#     # Assuming you have a DataModule instance called `data_module`
#     trainer.fit(model, data_module)
