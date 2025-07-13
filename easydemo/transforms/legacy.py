from typing import List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import time

from ..registry import TRANSFORMS
from .base import BaseTransform, BaseMultiTransform


@TRANSFORMS.register_module()
class GetRawFramefromCamera(BaseTransform):
    """Get a RAW Frame from Camera.

    This pipeline get a RAW frame image from Camera and
    returns the frame.

    Added Keys:

    - img_frame

    Args:
        device_num (int): Camera device number.
        size (tuple[int]): (w, h)
    """
    def __init__(self,
                 device_num: int = 0,
                 size: Tuple[int, int] = (1920, 1080)) -> None:
        super().__init__()
        self.device_num = device_num
        self.size = size
        self._setting_cam()  # setting cam

    def _setting_cam(self) -> None:
        """Setting Camera."""
        self.cam = cv2.VideoCapture(self.device_num, cv2.CAP_V4L2)
        self.cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"GB12"))
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, self.size[0])
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, self.size[1])

    def _postprocess(self, frame: np.ndarray) -> np.ndarray:
        """Postprocess the frame obtained from camera.

        Args:
            frame (np.ndarray): A frame obtained from camera.

        Returns:
            np.ndarray: A postprocessed frame.
        """
        # flatten frame
        frame_flatten = frame.flatten()

        # uint16 to uint12
        raw_16 = frame_flatten.view(np.uint16)[: self.size[0] * self.size[1]]
        raw_12 = raw_16 >> 4

        # reshape and type (uint12 -> float32)
        raw_reshape = raw_12.reshape(
            1, self.size[1], self.size[0]).astype(np.float32)
        return raw_reshape

    def transform(self, results: dict) -> dict:
        """Transform function to get a RAW frame from camera.

        Args:
            results (dict): Result dict.

        Returns:
            dict: The result dict contains the frame taken from camera.
        """
        _, frame = self.cam.read()  # read frame from camera
        frame = self._postprocess(frame)  # postprocess
        results['img_frame'] = frame
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(device_num={self.device_num}, '
        repr_str += f'size={self.size})'
        return repr_str


@TRANSFORMS.register_module()
class GetRandomFrame(BaseTransform):
    """Get a Random Frame.

    This pipeline get a Random Frame.

    Added Keys:

    - img_frame

    Args:
        size (tuple[int]): (w, h)
    """
    def __init__(self,
                 size: Tuple[int, int] = (1920, 1080)) -> None:
        super().__init__()
        self.size = size

    def transform(self, results: dict) -> dict:
        """Transform function to get a RAW frame from camera.

        Args:
            results (dict): Result dict.

        Returns:
            dict: The result dict contains the random frame.
        """
        frame = np.random.randint(2**12 - 1, size=self.size, dtype=np.uint16)
        results['img_frame'] = frame.reshape(
            1, self.size[1], self.size[0]).astype(np.float32)
        time.sleep(1/60)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'size={self.size})'
        return repr_str


@TRANSFORMS.register_module()
class CollectFramesfor3DNR(BaseTransform):
    """Collect Frames for 3DNR.

    This transform collects ``frame_cnt`` number of images and
    returns a frame list to be used in 3DNR.

    Required Keys:

    - img_frame


    Added Keys:

    - frame_cnt
    - img_frame_list

    Args:
        frame_cnt (int): The number of collected image frames. Defaults to 2.
    """
    def __init__(self, frame_cnt: int = 2) -> None:
        super().__init__()
        self.frame_cnt = frame_cnt
        self._frame_list = []  # frame_list

    @property
    def frame_list(self) -> List[Union[np.ndarray, torch.Tensor]]:
        """List: collected frame list."""
        return self._frame_list

    def transform(self, results: dict) -> dict:
        """Transform function to collect ``frame_cnt`` number of images.

        Args:
            results (dict): Result dict.

        Returns:
            dict: Updated result dict.
        """
        results['frame_cnt'] = self.frame_cnt
        # The length of the frame_list should be no more than frame_cnt.
        if len(self._frame_list) >= self.frame_cnt:
            self._frame_list = self._frame_list[-self.frame_cnt + 1:]

        # Add the current frame to frame_list
        img_frame = results['img_frame']
        while len(self._frame_list) < self.frame_cnt:
            self._frame_list.append(img_frame)

        # img_frame can be np.ndarray or torch.Tensor
        if isinstance(img_frame, np.ndarray):
            results['img_frame_list'] = np.array(self._frame_list)
        elif isinstance(img_frame, torch.Tensor):
            raise NotImplementedError  # TODO
        else:
            raise NotImplementedError
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(frame_cnt={self.frame_cnt})'


@TRANSFORMS.register_module()
class BlackWhiteLevel(BaseTransform):
    def __init__(self,
                 key: str,
                 black_level: float,
                 white_level: float) -> None:
        super().__init__()
        self.key = key
        self.black_level = black_level
        self.white_level = white_level

    def transform(self, results: dict) -> dict:
        img = results[self.key]
        new_img = img.sub_(self.black_level).div_(self.white_level).clamp(0, 1)
        results[self.key] = new_img
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(key={self.key}, '
        repr_str += f'black_level={self.black_level}, '
        repr_str += f'white_level={self.white_level})'
        return repr_str


@TRANSFORMS.register_module()
class BayerFormatting4Channel(BaseTransform):
    def __init__(self,
                 key: str,
                 input: str,
                 output: str) -> None:
        super().__init__()
        self.key = key
        self.input = input
        self.output = output

    def transform(self, results: dict) -> dict:
        img = results[self.key]
        if self.input == 'gbrg' and self.output == 'rggb':
            new_img = torch.cat((
                img[:, :, 1::2, 0::2],
                img[:, :, 0::2, 0::2],
                img[:, :, 1::2, 1::2],
                img[:, :, 0::2, 1::2]), dim=1)
        elif self.input == self.output:
            new_img = torch.cat((
                img[:, :, 0::2, 0::2],
                img[:, :, 0::2, 1::2],
                img[:, :, 1::2, 0::2],
                img[:, :, 1::2, 1::2]), dim=1)
        else:
            raise NotImplementedError
        results[self.key] = new_img
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(key={self.key}, '
        repr_str += f'input={self.input}, '
        repr_str += f'output={self.output})'
        return repr_str


@TRANSFORMS.register_module()
class MeanBaseBrightScaling(BaseTransform):
    def __init__(self,
                 scale: float,
                 patch_size: Optional[int] = None) -> None:
        super().__init__()
        self.scale = scale
        self.patch_size = patch_size

    def transform(self, results: dict) -> dict:
        img = results['img']
        _, _, h, w = img.shape
        if self.patch_size is not None:
            mid_h, mid_w = (h // 2, w // 2)
            mid_size = self.patch_size // 2
            scale = self.scale / \
                torch.mean(img[:, :,
                               mid_h - mid_size:mid_h + mid_size,
                               mid_w - mid_size:mid_w + mid_size])
        else:
            scale = self.scale / torch.mean(img)
        scale_img = (img * scale).clamp(0, 1)
        results['img'] = scale_img
        results['scale'] = scale
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(scale={self.scale}, '
        repr_str += f'patch_size={self.patch_size})'
        return repr_str


@TRANSFORMS.register_module()
class NR3DSplit(BaseTransform):
    def __init__(self) -> None:
        super().__init__()

    def transform(self, results: dict) -> dict:
        b, _, h, w = results['noisy'].size()
        results['display_input'] = results['noisy'][b // 2, :, :, :]
        results['noisy'] = results['noisy'].view(1, -1, h, w)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


@TRANSFORMS.register_module()
class NR3DMerge(BaseTransform):
    def __init__(self) -> None:
        super().__init__()

    def transform(self, results: dict) -> dict:
        results['noisy'] = results['display_input'].float()
        results['denoised'] = results['denoised'].float()
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


@TRANSFORMS.register_module()
class NaiveDemosaicing(BaseMultiTransform):
    def __init__(self,
                 keys: Sequence[str],
                 scale: float) -> None:
        super().__init__(keys)
        self.scale = scale

    def transform_key(self,
                      results: dict,
                      transform_info: dict,
                      key: str) -> Tuple[dict, dict]:
        img = results[key]
        g = img[:, 0, :, :]
        b = img[:, 1, :, :]
        r = img[:, 2, :, :]
        # r = img[:, 0, :, :]
        # g = img[:, 1, :, :]
        # b = img[:, 3, :, :]
        new_r = (r * torch.mean(g) / torch.mean(r)).clamp(0, 1)
        new_g = g.clamp(0, 1)
        new_b = (b * torch.mean(g) / torch.mean(b)).clamp(0, 1)
        new_img = torch.stack([new_r, new_g, new_b], dim=1).clamp(0, 1)
        new_img = new_img*(self.scale/torch.mean(new_img))
        new_img = new_img.clamp(0, torch.max(new_g))
        results[key] = new_img
        return results, transform_info

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(keys={self.keys}, '
        repr_str += f'scale={self.scale})'
        return repr_str


@TRANSFORMS.register_module()
class Demosaicing3x3Bilinear(BaseTransform):
    def __init__(self, key: str = 'img') -> None:
        super().__init__()
        self.key = key
        self.g_at_r = torch.Tensor([[0, 2, 0],
                                    [2, 0, 2],
                                    [0, 2, 0]]).float().view(1, 1, 3, 3) / 8
        self.g_at_b = self.g_at_r.clone()
        self.r_at_g1 = torch.Tensor([[0, 0, 0],
                                     [4, 0, 4],
                                     [0, 0, 0]]).float().view(1, 1, 3, 3) / 8
        self.r_at_g2 = torch.Tensor([[0, 4, 0],
                                     [0, 0, 0],
                                     [0, 4, 0]]).float().view(1, 1, 3, 3) / 8
        self.r_at_b = torch.Tensor([[2, 0, 2],
                                    [0, 0, 0],
                                    [2, 0, 2]]).float().view(1, 1, 3, 3) / 8
        self.b_at_g1 = self.r_at_g2.clone()
        self.b_at_g2 = self.r_at_g1.clone()
        self.b_at_r = self.r_at_b.clone()

    def transform(self, results: dict) -> dict:
        img = results[self.key]
        self.g_at_r = self.g_at_r.to(img.device)
        self.g_at_b = self.g_at_b.to(img.device)
        self.r_at_g1 = self.r_at_g1.to(img.device)
        self.r_at_g2 = self.r_at_g2.to(img.device)
        self.r_at_b = self.r_at_b.to(img.device)
        self.b_at_g1 = self.b_at_g1.to(img.device)
        self.b_at_g2 = self.b_at_g2.to(img.device)
        self.b_at_r = self.b_at_r.to(img.device)

        g00 = F.conv2d(img, self.g_at_r, padding=1, stride=2)
        g01 = img[:, :, 1::2, 1::2]
        g10 = img[:, :, 0::2, 0::2]
        g11 = F.conv2d(img.flip(dims=(2, 3)), self.g_at_b, padding=1, stride=2)
        g11 = g11.flip(dims=(2, 3))
        g = F.pixel_shuffle(torch.cat([g00, g01, g10, g11], dim=1), 2)
        r00 = img[:, :, 1::2, 0::2]
        r01 = F.conv2d(img.flip(dims=(3,)), self.r_at_g1, padding=1, stride=2)
        r01 = r01.flip(dims=(3,))
        r10 = F.conv2d(img.flip(dims=(2,)), self.r_at_g2, padding=1, stride=2)
        r10 = r10.flip(dims=(2,))
        r11 = F.conv2d(img.flip(dims=(2, 3)), self.r_at_b, padding=1, stride=2)
        r11 = r11.flip(dims=(2, 3))
        r = F.pixel_shuffle(torch.cat([r00, r01, r10, r11], dim=1), 2)
        b00 = F.conv2d(img, self.b_at_r, padding=1, stride=2)
        b01 = F.conv2d(img.flip(dims=(3,)), self.b_at_g1, padding=1, stride=2)
        b01 = b01.flip(dims=(3,))
        b10 = F.conv2d(img.flip(dims=(2,)), self.b_at_g2, padding=1, stride=2)
        b10 = b10.flip(dims=(2,))
        b11 = img[:, :, 0::2, 1::2]
        b = F.pixel_shuffle(torch.cat([b00, b01, b10, b11], dim=1), 2)
        new_img = torch.clip(torch.cat([r, g, b], dim=1), 0, 1)
        results[self.key] = new_img
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__ + f'(key={self.key})'
        return repr_str


@TRANSFORMS.register_module()
class Demosaicing5x5Malvar(BaseTransform):
    def __init__(self, key: str = 'img') -> None:
        super().__init__()
        self.key = key
        self.g_at_r = (
            torch.Tensor(
                [
                    [0, 0, -1, 0, 0],
                    [0, 0, 2, 0, 0],
                    [-1, 2, 4, 2, -1],
                    [0, 0, 2, 0, 0],
                    [0, 0, -1, 0, 0],
                ]
            )
            .float()
            .view(1, 1, 5, 5)
            / 8
        )

        self.g_at_b = self.g_at_r.clone()

        self.r_at_g1 = (
            torch.Tensor(
                [
                    [0, 0, 0.5, 0, 0],
                    [0, -1, 0, -1, 0],
                    [-1, 4, 5, 4, -1],
                    [0, -1, 0, -1, 0],
                    [0, 0, 0.5, 0, 0],
                ]
            )
            .float()
            .view(1, 1, 5, 5)
            / 8
        )

        self.r_at_g2 = (
            torch.Tensor(
                [
                    [0, 0, -1, 0, 0],
                    [0, -1, 4, -1, 0],
                    [0.5, 0, 5, 0, 0.5],
                    [0, -1, 4, -1, 0],
                    [0, 0, -1, 0, 0],
                ]
            )
            .float()
            .view(1, 1, 5, 5)
            / 8
        )

        self.r_at_b = (
            torch.Tensor(
                [
                    [0, 0, -1.5, 0, 0],
                    [0, 2, 0, 2, 0],
                    [-1.5, 0, 6, 0, -1.5],
                    [0, 2, 0, 2, 0],
                    [0, 0, -1.5, 0, 0],
                ]
            )
            .float()
            .view(1, 1, 5, 5)
            / 8
        )
        self.b_at_g1 = self.r_at_g2.clone()
        self.b_at_g2 = self.r_at_g1.clone()
        self.b_at_r = self.r_at_b.clone()

    def transform(self, results: dict) -> dict:
        x = results[self.key]
        self.g_at_r = self.g_at_r.to(x.device)
        self.g_at_b = self.g_at_b.to(x.device)
        self.r_at_g1 = self.r_at_g1.to(x.device)
        self.r_at_g2 = self.r_at_g2.to(x.device)
        self.r_at_b = self.r_at_b.to(x.device)
        self.b_at_g1 = self.b_at_g1.to(x.device)
        self.b_at_g2 = self.b_at_g2.to(x.device)
        self.b_at_r = self.b_at_r.to(x.device)

        x = F.pixel_shuffle(x, 2)
        g00 = F.conv2d(x, self.g_at_r, padding=2, stride=2)
        g01 = x[:, :, 1::2, 1::2]
        g10 = x[:, :, 0::2, 0::2]
        g11 = F.conv2d(x.flip(dims=(2, 3)), self.g_at_b, padding=2, stride=2)
        g11 = g11.flip(dims=(2, 3))
        g = F.pixel_shuffle(torch.cat([g00, g01, g10, g11], dim=1), 2)

        r00 = x[:, :, 1::2, 0::2]
        r01 = F.conv2d(x.flip(dims=(3,)), self.r_at_g1, padding=2, stride=2)
        r01 = r01.flip(dims=(3,))
        r10 = F.conv2d(x.flip(dims=(2,)), self.r_at_g2, padding=2, stride=2)
        r10 = r10.flip(dims=(2,))
        r11 = F.conv2d(x.flip(dims=(2, 3)), self.r_at_b, padding=2, stride=2)
        r11 = r11.flip(dims=(2, 3))
        r = F.pixel_shuffle(torch.cat([r00, r01, r10, r11], dim=1), 2)

        b00 = F.conv2d(x, self.b_at_r, padding=2, stride=2)
        b01 = F.conv2d(x.flip(dims=(3,)), self.b_at_g1, padding=2, stride=2)
        b01 = b01.flip(dims=(3,))
        b10 = F.conv2d(x.flip(dims=(2,)), self.b_at_g2, padding=2, stride=2)
        b10 = b10.flip(dims=(2,))
        b11 = x[:, :, 0::2, 1::2]
        b = F.pixel_shuffle(torch.cat([b00, b01, b10, b11], dim=1), 2)
        new_img = torch.cat([r, g, b], dim=1)
        results[self.key] = new_img
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__ + f'(key={self.key})'
        return repr_str


@TRANSFORMS.register_module()
class GrayWorldTorch(BaseMultiTransform):
    def __init__(self, keys: Sequence[str]) -> None:
        super().__init__(keys)

    def transform_key(self,
                      results: dict,
                      transform_info: dict,
                      key: str) -> Tuple[dict, dict]:
        img = results[key]
        if img.ndim == 4:
            r = img[:, 0, :, :]
            g = img[:, 1, :, :]
            b = img[:, 2, :, :]
        elif img.ndim == 3:
            r = img[0, :, :]
            g = img[1, :, :]
            b = img[2, :, :]
        new_r = (r * torch.mean(g) / torch.mean(r)).clamp(0, 1)
        new_g = g.clamp(0, 1)
        new_b = (b * torch.mean(g) / torch.mean(b)).clamp(0, 1)
        new_img = torch.stack([new_r, new_g, new_b], dim=-3)
        new_img = torch.clip(new_img, 0, torch.max(new_g))
        results[key] = new_img
        return results, transform_info

    def __repr__(self):
        repr_str = self.__class__.__name__ + f'(keys={self.keys})'
        return repr_str


@TRANSFORMS.register_module()
class DarkMode(BaseTransform):
    def __init__(self,
                 digital_gain: float = 0.1,
                 dark_mode: bool = True) -> None:
        super().__init__()
        self.dark_mode = dark_mode
        self.digital_gain = digital_gain

    def transform(self, results: dict) -> dict:
        denoised = results['denoised']
        noisy = results['noisy']
        org_mean = results['meta_info']['org_mean'].item()

        dm_scale = max(self.digital_gain, org_mean)
        if isinstance(noisy, torch.Tensor):
            stacked_mean = torch.mean(torch.cat([noisy, denoised], dim=0))
            denoised_scale = dm_scale / stacked_mean
            noisy_scale = org_mean / stacked_mean if self.dark_mode \
                else denoised_scale
            noisy = (noisy * noisy_scale).clamp(0, 1)
            denoised = (denoised * denoised_scale).clamp(0, 1)
        elif isinstance(noisy, np.ndarray):
            stacked_mean = np.mean(np.concatenate([noisy, denoised], axis=0))
            denoised_scale = dm_scale / stacked_mean
            noisy_scale = org_mean / stacked_mean if self.dark_mode \
                else denoised_scale
            noisy = np.clip(noisy * noisy_scale, 0, 1)
            denoised = np.clip(denoised * denoised_scale, 0, 1)
        else:
            raise NotImplementedError
        results['denoised'] = denoised
        results['noisy'] = noisy
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(dark_mode={self.dark_mode}, '
        repr_str += f'digital_gain={self.digital_gain})'
        return repr_str


@TRANSFORMS.register_module()
class NumpyResize(BaseMultiTransform):
    def __init__(self,
                 size: Sequence[int],
                 keys: Sequence[str]):
        super().__init__(keys)
        self.size = size

    def transform_key(self,
                      results: dict,
                      transform_info: dict,
                      key: str) -> Tuple[dict, dict]:
        new_img = cv2.resize(results[key],
                             tuple(self.size),
                             interpolation=cv2.INTER_LINEAR)
        results[key] = new_img
        return results, transform_info

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(keys={self.keys}, '
        repr_str += f'size={self.size})'
        return repr_str


@TRANSFORMS.register_module()
class GetOriginMean(BaseTransform):
    def __init__(self, key: str = 'img') -> None:
        super().__init__()
        self.key = key

    def transform(self, results: dict) -> dict:
        results['meta_info']['org_mean'] = results[self.key].mean()
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__ + f'(key={self.key})'
        return repr_str


@TRANSFORMS.register_module()
class GetRAWGreenMean(BaseTransform):
    def __init__(self, key: str = 'img') -> None:
        super().__init__()
        self.key = key

    def transform(self, results: dict) -> dict:
        raw = results[self.key]
        results['meta_info']['raw_g_mean'] = \
            np.mean(raw[:, :, 0::2, 0::2] + raw[:, :, 1::2, 1::2])
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__ + f'(key={self.key})'
        return repr_str


@TRANSFORMS.register_module()
class ConcatNumpyImage(BaseTransform):
    def __init__(self,
                 keys: Sequence[str],
                 output_key: str) -> None:
        super().__init__()
        self.keys = keys
        self.out_key = output_key

    def transform(self, results: dict) -> dict:
        img_list = [results[key] for key in self.keys]
        results[self.out_key] = np.hstack(img_list)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(keys={self.keys}, '
        repr_str += f'output_key={self.out_key})'
        return repr_str


@TRANSFORMS.register_module()
class LegacyRAW2RAWSplit(BaseTransform):
    def __init__(self, key: str) -> None:
        super().__init__()
        self.key = key

    def transform(self, results: dict) -> dict:
        results['noisy'] = results[self.key][:, 1, :, :].permute(2, 0, 1)
        results['denoised'] = results[self.key][:, 0, :, :].permute(2, 0, 1)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__ + f'(key={self.key})'
        return repr_str


@TRANSFORMS.register_module()
class MixFeature(BaseTransform):
    def __init__(self,
                 input_key: str,
                 output_key: str,
                 alpha: float) -> None:
        super().__init__()
        self.alpha = alpha
        self.input_key = input_key
        self.output_key = output_key

    def transform(self, results: dict) -> dict:
        input_img = results[self.input_key]
        output_img = results[self.output_key]
        results[self.output_key] = self.alpha * output_img + \
            (1 - self.alpha) * input_img
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(input_key={self.input_key}, '
        repr_str += f'output_key={self.output_key}, '
        repr_str += f'alpha={self.alpha})'
        return repr_str
