from abc import ABC

import pytorch_lightning as pl

from src.datamodule.amass_data_module import AMASSDataModule
from src.ligtning_module.unet_1d import UNet1D
from src.script.script import Script


class Unet1dScript(Script, ABC):
    def create_architecture(self, datamodule: pl.LightningDataModule):
        model = UNet1D(in_channels=156 + 52, out_channels=156, features=(64, 128, 256, 512), )
        return model

    def create_datamodule(self):
        """
        Create the data module for the script.
        :return: The data module for the script.
        """

        datamodule = AMASSDataModule(self.service, self.service.config['DATA']['TRAIN_DIR'],
                                     self.service.config['DATA']['VALID_DIR'],
                                     self.service.config['DATA']['TEST_DIR'],
                                     batch_size=int(self.service.config['APP']['BATCH_SIZE']))
        datamodule.prepare_data()
        datamodule.setup()

        return datamodule
