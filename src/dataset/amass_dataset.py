from typing import List, Callable

import numpy as np
import torch
from torch.utils.data import Dataset


class AMASSDataset(Dataset):
    def __init__(self, npz_file_paths, tfms: List[Callable] = []):
        """
        A PyTorch Dataset for loading AMASS .npz files.

        Args:
            npz_file_paths (list of str): List of paths to .npz files.
            transform (callable, optional): Optional transform to apply to the data.
        """
        self.file_paths = npz_file_paths
        self.tfms = tfms

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        """
        Loads data from a given index.

        Args:
            idx (int): Index of the data.

        Returns:
            dict: A dictionary containing keys like 'poses', 'betas', etc., from the .npz file.
        """
        npz_data = np.load(self.file_paths[idx])

        # Extract relevant data
        sample = {
            # 'trans': torch.FloatTensor(npz_data['trans']),  # Global translation
            'gender': npz_data['gender'],  # Gender
            # 'mocap_framerate': npz_data['mocap_framerate'],  # Mocap framerate
            # 'betas': npz_data['betas'],  # Shape coefficients
            # 'dmpls': npz_data['dmpls'],  # Dynamic PCA components
            'poses': torch.FloatTensor(npz_data['poses']),  # Pose data
            'idx': idx,  # Index of the sample
        }

        # Apply transform if provided
        if self.tfms:
            for tfm in self.tfms:
                sample = tfm(sample)

        return sample
