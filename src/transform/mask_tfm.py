from typing import Callable

import numpy as np
import torch


class MaskTfm(Callable):
    def __init__(self, temporal_window, mask_prob=0.3):
        """
        A callable transform to mask pose data.

        Args:
            temporal_window (int): Number of frames to mask.
            mask_prob (float): Probability of masking a frame.
        """
        self.temporal_window = temporal_window
        self.mask_prob = mask_prob

    def __call__(self, sample: dict):
        """
        Mask the pose data in the sample.

        Args:
            sample (dict): A dictionary containing keys like 'poses', 'betas', etc.

        Returns:
            dict: The sample dictionary with 'poses' replaced by masked poses.
        """
        pose_index = sample['idx']
        # Mask the pose data
        poses = sample['poses'].clone()

        T, num_features = poses.shape
        J = num_features // 3  # Number of joints

        # Initialize mask as all ones (no masking initially)
        mask = torch.ones(T, J, dtype=torch.float32, device=poses.device)

        # Use the pose_index to seed the random number generator
        rng = np.random.default_rng(seed=pose_index)

        # Structured temporal masking
        for t in range(0, T, self.temporal_window):
            for j in range(J):
                if rng.random() < self.mask_prob:
                    # Randomly sample joints to mask
                    # num_joints_to_mask = rng.integers(1, J + 1)
                    # joints_to_mask = rng.choice(J, size=num_joints_to_mask, replace=False)

                    # Apply the same mask for the temporal window
                    mask[t:t + self.temporal_window, j] = 0

        # Expand mask to (x, y, z) coordinates
        expanded_mask = mask.unsqueeze(-1).repeat(1, 1, 3).view(T, num_features)

        # Apply mask to the pose array
        masked_poses = poses * expanded_mask

        sample['masked_poses'] = masked_poses
        sample['mask'] = mask
        return sample
