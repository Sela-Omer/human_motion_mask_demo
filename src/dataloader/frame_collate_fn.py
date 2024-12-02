from typing import Callable, List

import torch

from config import frames_per_sample


class FrameCollateFn(Callable):
    def __init__(self, frames_per_sample=frames_per_sample):
        self.frames_per_sample = frames_per_sample

    def tensor_to_frame_lst(self, tensor):
        assert tensor.ndim == 2, f"Expected tensor to have 2 dimensions (N_SAMPLES x FEATURES), got {tensor.ndim}"
        if tensor.shape[0] < self.frames_per_sample:
            return []
        # assert tensor.shape[
        #            0] >= self.frames_per_sample, f"Expected tensor to have at least {self.frames_per_sample} frames, got {tensor.shape}"
        frame_lst = []
        for i in range(0, tensor.shape[0] - self.frames_per_sample, self.frames_per_sample):
            frame = tensor[i:i + self.frames_per_sample]
            frame_lst.append(frame)
        last_frame = tensor[-self.frames_per_sample:]
        frame_lst.append(last_frame)
        return frame_lst

    def __call__(self, batch: List[dict]):
        collate_dict = {key: [] for key in batch[0].keys()}

        for item in batch:
            for key, value in item.items():
                if isinstance(value, torch.Tensor):
                    collate_dict[key] += self.tensor_to_frame_lst(value)
                elif hasattr(value, 'len'):
                    collate_dict[key] += value
                else:
                    collate_dict[key].append(value)

        for key, value in collate_dict.items():
            if isinstance(value[0], torch.Tensor):
                # print([v.shape for v in value])
                collate_dict[key] = torch.stack(value, dim=0)
        return collate_dict
