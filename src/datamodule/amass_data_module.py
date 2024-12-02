import glob

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from src.dataloader.frame_collate_fn import FrameCollateFn
from src.dataset.amass_dataset import AMASSDataset
from src.transform.mask_tfm import MaskTfm


class AMASSDataModule(pl.LightningDataModule):
    def __init__(self, service, train_dir, val_dir, test_dir, batch_size=32):
        """
        A PyTorch Lightning DataModule for the AMASS dataset.

        Args:
            train_dir (str): Path to the training directory.
            val_dir (str): Path to the validation directory.
            test_dir (str): Path to the test directory.
            batch_size (int): Batch size for DataLoader.
        """
        super().__init__()
        self.service = service
        self.mask_temporal_window = int(self.service.config['DATA']['MASK_TEMPORAL_WINDOW'])
        self.frames_per_sample = int(self.service.config['DATA']['FRAMES_PER_SAMPLE'])
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir
        self.batch_size = batch_size
        self.collate_fn = FrameCollateFn(self.frames_per_sample)

    def get_amass_files(self, directory):
        """
        Get a list of .npz files in the specified directory.

        Args:
            directory (str): Path to the directory containing .npz files.

        Returns:
            list: List of .npz files in the directory.
        """
        return sorted(glob.glob(f'{directory}/*/*poses.npz'))

    def prepare_data(self) -> None:
        """
        Prepare the data. This method is called only once to prepare the data.
        """
        self.train_files = self.get_amass_files(self.train_dir)
        self.val_files = self.get_amass_files(self.val_dir)
        self.test_files = self.get_amass_files(self.test_dir)

    def setup(self, stage=None):
        """
        Split the data and prepare datasets.
        """
        self.train_dataset = AMASSDataset(self.train_files, tfms=[MaskTfm(self.mask_temporal_window)])
        self.val_dataset = AMASSDataset(self.val_files, tfms=[MaskTfm(self.mask_temporal_window)])
        self.test_dataset = AMASSDataset(self.test_files, tfms=[MaskTfm(self.mask_temporal_window)])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, collate_fn=self.collate_fn, shuffle=True,
                          num_workers=1)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, collate_fn=self.collate_fn, num_workers=1)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, collate_fn=self.collate_fn, num_workers=1)
