from typing import Optional, Sequence, Tuple

import cv2
import numpy as np
import torch.nn.functional as F

from easycv.registry import TRANSFORMS
from .base import BaseMultiTransform


@TRANSFORMS.register_module()
class CvtColor(BaseMultiTransform):
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
        if self.input_type == 'bgr' and self.output_type == 'rgb':
            new_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif self.input_type == 'rgb' and self.output_type == 'bgr':
            new_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
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


@TRANSFORMS.register_module()
class TorchUnsqueeze(BaseMultiTransform):
    def __init__(self, keys: Sequence[str]) -> None:
        super().__init__(keys)

    def transform_key(self,
                      results: dict,
                      transform_info: dict,
                      key: str) -> Tuple[dict, dict]:
        results[key] = results[key].unsqueeze(0)
        return results, transform_info

    def __repr__(self):
        repr_str = self.__class__.__name__ + f'(keys={self.keys})'
        return repr_str


@TRANSFORMS.register_module()
class NumpyUnsqueeze(BaseMultiTransform):
    def __init__(self, keys: Sequence[str]) -> None:
        super().__init__(keys)

    def transform_key(self,
                      results: dict,
                      transform_info: dict,
                      key: str) -> Tuple[dict, dict]:
        results[key] = np.expand_dims(results[key], 0)
        return results, transform_info

    def __repr__(self):
        repr_str = self.__class__.__name__ + f'(keys={self.keys})'
        return repr_str


@TRANSFORMS.register_module()
class PadTensor(BaseMultiTransform):
    def __init__(self,
                 keys: Sequence[str],
                 pad: Tuple[int],
                 mode: str = 'constant',
                 value: Optional[float] = None) -> None:
        super().__init__(keys)
        self.pad = pad
        self.mode = mode
        self.value = value if value is not None else 0

    def transform_key(self,
                      results: dict,
                      transform_info: dict,
                      key: str) -> Tuple[dict, dict]:
        img = results[key]
        pad_img = F.pad(img, self.pad, self.mode, self.value)
        results[key] = pad_img
        return results, transform_info

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(pad={self.pad}, '
        repr_str += f'keys={self.keys}, '
        repr_str += f'mode={self.mode}, '
        repr_str += f'value={self.value})'
        return repr_str


@TRANSFORMS.register_module()
class PadNumpy(BaseMultiTransform):
    def __init__(self,
                 keys: Sequence[str],
                 pad: Tuple[int],
                 mode: str = 'constant',
                 value: Optional[float] = None) -> None:
        super().__init__(keys)
        self.pad = pad
        self.mode = mode
        self.value = value if value is not None else 0

    def transform_key(self,
                      results: dict,
                      transform_info: dict,
                      key: str) -> Tuple[dict, dict]:
        img = results[key]
        pad_img = np.pad(
            img,
            ((self.pad[2], self.pad[3]), (self.pad[0], self.pad[1])),
            'constant', constant_values=self.value)
        results[key] = pad_img
        return results, transform_info

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(pad={self.pad}, '
        repr_str += f'keys={self.keys}, '
        repr_str += f'mode={self.mode}, '
        repr_str += f'value={self.value})'
        return repr_str


@TRANSFORMS.register_module()
class CropTensor(BaseMultiTransform):
    def __init__(self,
                 keys: Sequence[str],
                 crop: Tuple[int]) -> None:
        super().__init__(keys)
        self.crop = crop

    def transform_key(self,
                      results: dict,
                      transform_info: dict,
                      key: str) -> Tuple[dict, dict]:
        img = results[key]
        _, _, h, w = img.shape
        crop_img = img[:, :,
                       self.crop[2]:h-self.crop[3],
                       self.crop[0]:w-self.crop[1]]
        results[key] = crop_img
        return results, transform_info

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(keys={self.keys}, '
        repr_str += f'crop={self.crop})'
        return repr_str


@TRANSFORMS.register_module()
class CropNumpy(BaseMultiTransform):
    def __init__(self,
                 keys: Sequence[str],
                 crop: Tuple[int]) -> None:
        super().__init__(keys)
        self.crop = crop

    def transform_key(self,
                      results: dict,
                      transform_info: dict,
                      key: str) -> Tuple[dict, dict]:
        img = results[key]
        h, w, _ = img.shape
        crop_img = img[self.crop[2]:h-self.crop[3],
                       self.crop[0]:w-self.crop[1], :]
        results[key] = crop_img
        return results, transform_info

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(keys={self.keys}, '
        repr_str += f'crop={self.crop})'
        return repr_str
