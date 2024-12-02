import os.path
from abc import ABC

import numpy as np
import pytorch_lightning as pl
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from src.helper.param_helper import convert_param_to_type
from src.script.script import Script


class EvalScript(Script, ABC):

    def create_trainer(self, callbacks: list):
        """
        Create a trainer with specified configurations.

        Args:
            callbacks (list): A list of callbacks to be used during training.

        Returns:
            pl.Trainer: The created trainer object.
        """
        # Create the trainer with specified configurations
        trainer = pl.Trainer(
            max_epochs=1,
            accelerator=self.service.config['APP']['ACCELERATOR'],
            log_every_n_steps=1,
            callbacks=callbacks,
            logger=None,
            devices=convert_param_to_type(self.service.config['APP']['DEVICES']),
            num_nodes=convert_param_to_type(self.service.config['APP']['NUM_NODES']),
            strategy=self.service.config['APP']['STRATEGY'],
        )
        return trainer

    def _get_model_checkpoint(self):
        """
        This method returns the path to the model checkpoint.
        :return: The path to the model checkpoint.

        """
        model_dir = f'model/{self.service.model_name}'
        assert os.path.isdir(model_dir), f"Model directory {model_dir} does not exist."
        version_lst = os.listdir(model_dir)

        version_dict = {}
        for version in version_lst:
            version_dir_lst = version.split('_')
            if len(version_dir_lst) != 2 or not version_dir_lst[1].isdigit() or version_dir_lst[0] != 'version':
                continue
            version_ind = int(version_dir_lst[1])
            version_dict[version_ind] = version
        assert len(version_dict) > 0, f"No model versions found in {model_dir}."

        version = None
        if self.service.config['EVAL']['CHECKPOINT_VERSION'] == 'highest':
            version = max(version_dict.keys())
        if self.service.config['EVAL']['CHECKPOINT_VERSION'] == 'loweset':
            version = min(version_dict.keys())
        if self.service.config['EVAL']['CHECKPOINT_VERSION'].isdigit():
            version = int(self.service.config['EVAL']['CHECKPOINT_VERSION'])
        assert version in version_dict, f"Version {version} not found in {model_dir}."
        checkpoint_dir = f'{model_dir}/{version_dict[version]}/checkpoints'
        assert os.path.isdir(checkpoint_dir), f"Checkpoint directory {checkpoint_dir} does not exist."
        checkpoint_lst = os.listdir(checkpoint_dir)
        checkpoint_lst = [checkpoint for checkpoint in checkpoint_lst if checkpoint.endswith('.ckpt')]
        assert len(checkpoint_lst) > 0, f"No checkpoints found in {checkpoint_dir}."
        if len(checkpoint_lst) > 1:
            print(f"Multiple checkpoints found in {checkpoint_dir}. Using the first one: {checkpoint_lst[0]}")
        return f'{checkpoint_dir}/{checkpoint_lst[0]}'

    def plot_predictions(self, y_hat, y, mask):
        y_hat = y_hat.detach().cpu().numpy()
        y = y.detach().cpu().numpy()
        N_FRAMES, JOINTS = y_hat.shape

        fig, axes = plt.subplots(JOINTS // 3, 3, figsize=(15, 4 * JOINTS // 3))
        for joint in range(0, JOINTS):
            # plot y,y_hat for each joint
            ax = axes[joint // 3, joint % 3]
            y_line = y[:, joint]
            y_hat_line = y_hat[:, joint]
            m = mask[:, joint // 3]

            mean = (y_line.mean() + y_hat_line.mean()) / 2
            bottom,top = mean - 0.5, mean + 0.5
            ax.set_ylim(bottom, top)

            ax.plot(range(N_FRAMES), y_line, label='y')
            ax.plot(range(N_FRAMES), y_hat_line, label='y_hat')

            x = np.arange(len(m))
            ax.plot(x[m == 1], np.ones_like(x[m == 1]) * bottom, '|', color='red', markersize=8, label='mask')
            ax.legend()
            ax.set_title(f'Joint {joint // 3}, Axis {joint % 3}')
        plt.tight_layout()
        return fig

    def __call__(self):
        """
        This method orchestrates the training process.
        It creates the data module, architecture, callbacks, and trainer,
        and then fits the model using the trainer.
        """
        # Create the data module
        datamodule = self.create_datamodule()

        # Create the architecture
        arch = self.create_architecture(datamodule)

        # load the model from the checkpoint
        arch.load_state_dict(torch.load(self._get_model_checkpoint())['state_dict'])

        # Get the dataloader
        train_dataloader = datamodule.train_dataloader()

        for i, batch in tqdm(enumerate(train_dataloader)):
            x_masked = torch.cat((batch['masked_poses'], batch['mask']), dim=-1)
            y = batch['poses']
            x_masked = x_masked.permute(0, 2, 1)  # (BATCH, N_FRAMES, JOINTS) -> (BATCH, JOINTS, N_FRAMES)
            y_hat = arch(x_masked)
            y_hat = y_hat.permute(0, 2, 1)  # (BATCH, JOINTS, N_FRAMES) -> (BATCH, N_FRAMES, JOINTS)
            for b in range(y_hat.size(0)):
                self.plot_predictions(y_hat[b], y[b], batch['mask'][b])
                f = self._get_model_checkpoint().replace('.ckpt', f'-sample-{i}-{b}.png')
                plt.savefig(f)
                break
