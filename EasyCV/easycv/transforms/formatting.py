from typing import Optional, Sequence, Tuple

import torch
import numpy as np

from easycv.registry import TRANSFORMS
from .base import BaseTransform, BaseMultiTransform


@TRANSFORMS.register_module()
class ToTensor(BaseMultiTransform):
    """Convert some results to :obj:`torch.Tensor` by given keys.

    Args:
        keys (Sequence[str]): Keys that need to be converted to Tensor.
        device (str): torch.Tensor device.
    """

    def __init__(self,
                 keys: Sequence[str],
                 device: str = 'cpu'):
        super().__init__(keys)
        self.device = device

    def transform_key(self,
                      results: dict,
                      transform_info: dict,
                      key: str) -> Tuple[dict, dict]:
        results[key] = torch.from_numpy(results[key]).to(self.device)
        return results, transform_info

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(keys={self.keys}, '
        repr_str += f'device={self.device})'
        return repr_str


@TRANSFORMS.register_module()
class ToDevice(BaseMultiTransform):
    def __init__(self,
                 keys: Sequence[str],
                 device: str):
        super().__init__(keys)
        self.device = device

    def transform_key(self,
                      results: dict,
                      transform_info: dict,
                      key: str) -> Tuple[dict, dict]:
        results[key] = results[key].to(self.device)
        return results, transform_info

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(keys={self.keys}, '
        repr_str += f'device={self.device})'
        return repr_str


@TRANSFORMS.register_module()
class ToNumpyImage(BaseMultiTransform):
    def __init__(self, keys: Sequence[str]):
        super().__init__(keys)

    def transform_key(self,
                      results: dict,
                      transform_info: dict,
                      key: str) -> Tuple[dict, dict]:
        img = results[key]
        if img.dim() == 4:
            img = img.squeeze()
        assert img.dim() == 3, "image dimension must be 3 or 4"
        results[key] = img.permute(1, 2, 0).detach().cpu().numpy()
        return results, transform_info

    def __repr__(self):
        return self.__class__.__name__ + f'(keys={self.keys})'


@TRANSFORMS.register_module()
class KeyMapping(BaseTransform):
    def __init__(self,
                 mapping: Optional[dict] = None) -> None:
        super().__init__()
        self.mapping = mapping

    def transform(self, results: dict) -> dict:
        for key, value in self.mapping.items():
            results[value] = results[key]
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__ + f'(mapping={self.mapping})'
        return repr_str


@TRANSFORMS.register_module()
class NumpyTypecast(BaseMultiTransform):
    def __init__(self,
                 keys: Sequence[str],
                 input_type: str,
                 output_type: str) -> None:
        super().__init__(keys)
        self.input_type = input_type
        self.output_type = output_type

    def transform_key(self,
                      results: dict,
                      transform_info: dict,
                      key: str) -> Tuple[dict, dict]:
        img = results[key]
        if self.input_type == 'uint12' and self.output_type == 'float':
            new_img = img.astype(np.float64)
            new_img /= 4095
        elif self.input_type == 'uint8' and self.output_type == 'float':
            new_img = img.astype(np.float64)
            new_img /= 255
        elif self.input_type == 'float' and self.output_type == 'uint8':
            new_img = img * 255
            new_img = new_img.astype(np.uint8)
        elif self.input_type == 'float' and self.output_type == 'uint12':
            new_img = img * 4095
            new_img = new_img.astype(np.uint16)
        else:
            raise NotImplementedError
        results[key] = new_img
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(keys={self.keys}, '
        repr_str += f'input_type={self.input_type}, '
        repr_str += f'output_type={self.output_type})'
        return repr_str


@TRANSFORMS.register_module()
class TensorTypecast(BaseMultiTransform):
    def __init__(self,
                 keys: Sequence[str],
                 input_type: str,
                 output_type: str) -> None:
        super().__init__(keys)
        self.input_type = input_type
        self.output_type = output_type

    def transform_key(self,
                      results: dict,
                      transform_info: dict,
                      key: str) -> Tuple[dict, dict]:
        img = results[key]
        if self.input_type == 'float32' and self.output_type == 'float16':
            new_img = img.to(torch.float16)
        else:
            raise NotImplementedError
        results[key] = new_img
        return results, transform_info

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(keys={self.keys}, '
        repr_str += f'input_type={self.input_type}, '
        repr_str += f'output_type={self.output_type})'
        return repr_str
