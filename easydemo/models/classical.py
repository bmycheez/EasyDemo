from typing import Optional

import torch
import torch.nn as nn

from ..registry import MODELS


def torch_dtype(dtype_str):
    TORCH_DTYPE = {
        'float64': torch.float64,
        'float32': torch.float32,
        'float16': torch.float16,
        'uint8': torch.uint8
    }
    return TORCH_DTYPE[dtype_str]


@MODELS.register_module()
class NR3D(nn.Module):
    def __init__(self,
                 threshold: float,
                 dtype: str = 'float16') -> None:
        super().__init__()
        self.dtype = torch_dtype(dtype)
        self.threshold = threshold
        self.diff = None

    def _frame_avg(self, frames: torch.Tensor) -> torch.Tensor:
        return torch.mean(frames, dim=0, keepdim=True)

    def _get_diff(self, frames: torch.Tensor) -> torch.Tensor:
        for i in range(frames.size()[0] - 1):
            if self.diff is not None and i != frames.size()[0] - 2:
                continue
            else:
                diff_per_frm = torch.abs(frames[i + 1] - frames[i])
                diff_per_frm /= torch.max(diff_per_frm)
                diff_per_frm = torch.where(diff_per_frm > self.threshold, 1, 0)
                self.diff = \
                    torch.logical_or(self.diff, diff_per_frm) \
                    if self.diff is not None else diff_per_frm
        return self.diff

    def de_afterimage(self,
                      diff: Optional[torch.Tensor],
                      frame: torch.Tensor,
                      avg: torch.Tensor):
        return torch.where(diff is not None and diff == 1, frame, avg)

    def forward(self, frames):
        avg = self._frame_avg(frames)
        avg = self.de_afterimage(
            diff=self._get_diff(frames),
            frame=frames[len(frames) - 1],
            avg=avg)
        return avg
