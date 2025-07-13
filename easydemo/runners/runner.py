import copy
import time
import subprocess
import cv2
from typing import Dict, Optional, Sequence, Union
import numpy as np

from easycv.config import Config, ConfigDict

from easydemo.registry import (RUNNERS, )
from easydemo.press_key import PressKeyManager
from easydemo.transforms.wrappers import Compose

ConfigType = Union[Dict, Config, ConfigDict]


@RUNNERS.register_module()
class Runner:
    def __init__(
        self,
        fps: float,
        gain_list: Sequence[int],
        exposure_list: Sequence[float],
        gain_index: int = 0,
        exposure_index: int = 0,
        use_auto_exposure: bool = False,
        auto_exposure_target: Optional[float] = None,
        auto_exposure_speed: Optional[float] = None,
        auto_exposure_range: Optional[float] = None,
        pipeline: Optional[ConfigType] = None,
        cfg: Optional[ConfigType] = None
    ) -> None:
        super().__init__()
        # recursively copy the `cfg` because `self.cfg` will be modified
        # everywhere.
        if cfg is not None:
            if isinstance(cfg, Config):
                self.cfg = copy.deepcopy(cfg)
            elif isinstance(cfg, dict):
                self.cfg = Config(cfg)
        else:
            self.cfg = Config(dict())

        # fps, gain, exposure
        self._fps = fps * 1000000
        self.gain_list = gain_list
        self.exposure_list = exposure_list
        self._gain_index = gain_index
        self._exposure_index = exposure_index

        # auto exposure
        self.use_auto_exposure = use_auto_exposure
        self.auto_exposure_target = auto_exposure_target
        self.auto_exposure_range = auto_exposure_range
        self.auto_exposure_speed = auto_exposure_speed
        self._auto_exposure = exposure_list[exposure_index]
        self._auto_gain = gain_list[gain_index]

        # pipeline
        self.pipeline = Compose(pipeline)

    @property
    def fps(self) -> int:
        return self._fps

    @property
    def gain_index(self) -> int:
        return self._gain_index

    @property
    def exposure_index(self) -> int:
        return self._exposure_index

    @property
    def gain(self) -> int:
        if self.use_auto_exposure:
            return int(self._auto_gain)
        return int(self.gain_list[self.gain_index])

    @property
    def exposure(self) -> int:
        if self.use_auto_exposure:
            return int(self._auto_exposure * 1000)
        return int(self.exposure_list[self.exposure_index] * 1000)

    @classmethod
    def from_cfg(cls, cfg: ConfigType) -> 'Runner':
        """Build a runner from config.

        Args:
            cfg (ConfigType): A config used for building runner. Keys of
                ``cfg`` can see :meth:`__init__`.

        Returns:
            Runner: A runner build from ``cfg``.
        """
        cfg = copy.deepcopy(cfg)
        runner = cls(
            fps=cfg['fps'],
            gain_list=cfg['gain_list'],
            exposure_list=cfg['exposure_list'],
            gain_index=cfg['gain_index'],
            exposure_index=cfg['exposure_index'],
            use_auto_exposure=cfg['use_auto_exposure'],
            auto_exposure_target=cfg['auto_exposure_target'],
            auto_exposure_speed=cfg['auto_exposure_speed'],
            auto_exposure_range=cfg['auto_exposure_range'],
            pipeline=cfg['pipeline'],
            cfg=cfg,
        )

        return runner

    def _auto_alpha(self,
                    output: np.ndarray,
                    raw_g_mean: np.ndarray,
                    key: str = 'a') -> None:
        prev = PressKeyManager.get_instance(key).auto_state
        if raw_g_mean > 1700:
            PressKeyManager.get_instance(key).auto_state = 0
        elif raw_g_mean < 500:
            PressKeyManager.get_instance(key).auto_state = 3
        elif raw_g_mean < 800:
            PressKeyManager.get_instance(key).auto_state = 2
        elif raw_g_mean < 1100:
            PressKeyManager.get_instance(key).auto_state = 1
        else:
            PressKeyManager.get_instance(key).auto_state = 0
        cur = PressKeyManager.get_instance(key).auto_state

        if prev != cur:
            print(f'When {raw_g_mean}, Alpha {prev} changed to {cur}')

        alpha = PressKeyManager.get_instance('a').get_state_name()
        cv2.putText(output,
                    f'Auto Alpha: {alpha}',
                    (10, 40), cv2.FONT_ITALIC, 1, (0, 0, 255), 2)
        return output

    def _check_key(self, key) -> bool:
        return PressKeyManager.check_instance_created(key)

    def run(self) -> None:
        subprocess.call([
            "v4l2-ctl", "-d", "0", "-c",
            f"frame_rate={self.fps}"])
        subprocess.call([
            "v4l2-ctl", "-d", "0", "-c",
            f"exposure={self.exposure}," +
            f"gain={self.gain}"])
        time.sleep(2)

        while True:
            results = dict(meta_info={})
            results = self.pipeline(results)

            output = results['output']
            raw_g_mean = results['meta_info']['raw_g_mean']

            # auto exposure
            if self.use_auto_exposure:
                base = np.abs(self.auto_exposure_target - raw_g_mean)

                if base > self.auto_exposure_range:
                    exponent = \
                        np.sign(self.auto_exposure_target - raw_g_mean) \
                        * self.auto_exposure_speed
                    step = pow(base, exponent)
                    self._auto_exposure = self._auto_exposure * step
                    self._auto_gain = self._auto_gain * step

                if self._auto_exposure > 30:
                    self._auto_exposure = 30
                if self._auto_gain > 300:
                    self._auto_gain = 300

                exp = self.exposure
                gain = self.gain
                subprocess.call(["v4l2-ctl", "-d", "0", "-c",
                                 f"exposure={exp},gain={gain}"])

            # cmd
            cmd = cv2.waitKey(1)

            # exposure ctrl
            if cmd == ord("e"):
                self._exposure_index = (self._exposure_index + 1) % \
                    len(self.exposure_list)
                gain = self.gain
                exp = self.exposure
                subprocess.call(["v4l2-ctl", "-d", "0", "-c",
                                 f"exposure={exp},gain={gain}"])
                cv2.putText(output,
                            f'Exposure : {exp / 1000}ms / Gain : {gain}',
                            (10, 80), cv2.FONT_ITALIC, 1, (0, 0, 255), 2)

            # gain ctrl
            elif cmd == ord("g"):
                self._gain_index = (self._gain_index + 1) % \
                    len(self.gain_list)
                gain = self.gain
                exp = self.exposure
                subprocess.call(["v4l2-ctl", "-d", "0", "-c",
                                 f"exposure={exp},gain={gain}"])
                cv2.putText(output,
                            f'Exposure : {exp / 1000}ms / Gain : {gain}',
                            (10, 80), cv2.FONT_ITALIC, 1, (0, 0, 255), 2)

            # model alpha ctrl : pipeline
            elif cmd == ord("a") and self._check_key("a"):
                PressKeyManager.get_instance('a').press_key()
                if not PressKeyManager.get_instance('a').is_auto_state:
                    alpha = PressKeyManager.get_instance('a').get_state_name()
                    cv2.putText(output,
                                f'Manual Alpha : {alpha}',
                                (10, 80), cv2.FONT_ITALIC, 1, (0, 0, 255), 2)

            # dark view on / off : pipeline
            elif cmd == ord("d") and self._check_key("d"):
                PressKeyManager.get_instance('d').press_key()
                dark_state = PressKeyManager.get_instance('d').state
                dark_state_str = 'On' if dark_state == 1 else 'Off'
                cv2.putText(output,
                            f'Dark: {dark_state_str}',
                            (10, 80), cv2.FONT_ITALIC, 1, (0, 0, 255), 2)

            # digital zoom on / off : pipeline
            elif cmd == ord("z") and self._check_key("z"):
                PressKeyManager.get_instance('z').press_key()
                zoom_state = PressKeyManager.get_instance('z').state
                zoom_state_str = 'On' if zoom_state == 1 else 'Off'
                cv2.putText(output,
                            f'Zoom: {zoom_state_str}',
                            (10, 80), cv2.FONT_ITALIC, 1, (0, 0, 255), 2)

            # quit
            elif cmd == ord("q"):
                break

            # auto_alpha
            if PressKeyManager.get_instance('a').is_auto_state:
                output = self._auto_alpha(output, raw_g_mean, 'a')

            cv2.imshow("output", output)


@RUNNERS.register_module()
class FPSTestRunner:
    def __init__(
        self,
        fps: float,
        gain_list: Sequence[int],
        exposure_list: Sequence[float],
        gain_index: int = 0,
        exposure_index: int = 0,
        use_auto_exposure: bool = False,
        auto_exposure_target: Optional[float] = None,
        auto_exposure_speed: Optional[float] = None,
        auto_exposure_range: Optional[float] = None,
        pipeline: Optional[ConfigType] = None,
        cfg: Optional[ConfigType] = None
    ) -> None:
        super().__init__()
        # recursively copy the `cfg` because `self.cfg` will be modified
        # everywhere.
        if cfg is not None:
            if isinstance(cfg, Config):
                self.cfg = copy.deepcopy(cfg)
            elif isinstance(cfg, dict):
                self.cfg = Config(cfg)
        else:
            self.cfg = Config(dict())

        # pipeline
        self.pipeline = Compose(pipeline)

    @classmethod
    def from_cfg(cls, cfg: ConfigType) -> 'Runner':
        """Build a runner from config.

        Args:
            cfg (ConfigType): A config used for building runner. Keys of
                ``cfg`` can see :meth:`__init__`.

        Returns:
            Runner: A runner build from ``cfg``.
        """
        cfg = copy.deepcopy(cfg)
        runner = cls(
            fps=cfg['fps'],
            gain_list=cfg['gain_list'],
            exposure_list=cfg['exposure_list'],
            gain_index=cfg['gain_index'],
            exposure_index=cfg['exposure_index'],
            use_auto_exposure=cfg['use_auto_exposure'],
            auto_exposure_target=cfg['auto_exposure_target'],
            auto_exposure_speed=cfg['auto_exposure_speed'],
            auto_exposure_range=cfg['auto_exposure_range'],
            pipeline=cfg['pipeline'],
            cfg=cfg,
        )

        return runner

    def _auto_alpha(self,
                    output: np.ndarray,
                    raw_g_mean: np.ndarray,
                    key: str = 'a') -> None:
        prev = PressKeyManager.get_instance(key).auto_state
        if raw_g_mean > 1700:
            PressKeyManager.get_instance(key).auto_state = 0
        elif raw_g_mean < 500:
            PressKeyManager.get_instance(key).auto_state = 3
        elif raw_g_mean < 800:
            PressKeyManager.get_instance(key).auto_state = 2
        elif raw_g_mean < 1100:
            PressKeyManager.get_instance(key).auto_state = 1
        else:
            PressKeyManager.get_instance(key).auto_state = 0
        cur = PressKeyManager.get_instance(key).auto_state

        if prev != cur:
            print(f'When {raw_g_mean}, Alpha {prev} changed to {cur}')

        alpha = PressKeyManager.get_instance('a').get_state_name()
        cv2.putText(output,
                    f'Auto Alpha: {alpha}',
                    (10, 40), cv2.FONT_ITALIC, 1, (0, 0, 255), 2)
        return output

    def _check_key(self, key) -> bool:
        return PressKeyManager.check_instance_created(key)

    def run(self) -> None:
        while True:
            results = dict(meta_info={})
            results = self.pipeline(results)

            output = results['output']
            raw_g_mean = results['meta_info']['raw_g_mean']

            # cmd
            cmd = cv2.waitKey(1)

            # model alpha ctrl : pipeline
            if cmd == ord("a") and self._check_key("a"):
                PressKeyManager.get_instance('a').press_key()
                if not PressKeyManager.get_instance('a').is_auto_state:
                    alpha = PressKeyManager.get_instance('a').get_state_name()
                    cv2.putText(output,
                                f'Manual Alpha : {alpha}',
                                (10, 40), cv2.FONT_ITALIC, 1, (0, 0, 255), 2)

            # dark view on / off : pipeline
            elif cmd == ord("d") and self._check_key("d"):
                PressKeyManager.get_instance('d').press_key()
                dark_state = PressKeyManager.get_instance('d').state
                dark_state_str = 'On' if dark_state == 1 else 'Off'
                cv2.putText(output,
                            f'Dark: {dark_state_str}',
                            (10, 40), cv2.FONT_ITALIC, 1, (0, 0, 255), 2)

            # digital zoom on / off : pipeline
            elif cmd == ord("z") and self._check_key("z"):
                PressKeyManager.get_instance('z').press_key()
                zoom_state = PressKeyManager.get_instance('z').state
                zoom_state_str = 'On' if zoom_state == 1 else 'Off'
                cv2.putText(output,
                            f'Zoom: {zoom_state_str}',
                            (10, 40), cv2.FONT_ITALIC, 1, (0, 0, 255), 2)

            # 3dnr on / off : pipeline
            elif cmd == ord("m") and self._check_key("m"):
                PressKeyManager.get_instance('m').press_key()
                dnr_state = PressKeyManager.get_instance('m').state
                dnr_state_str = 'On' if dnr_state == 1 else 'Off'
                cv2.putText(output,
                            f'3DNR: {dnr_state_str}',
                            (10, 40), cv2.FONT_ITALIC, 1, (0, 0, 255), 2)

            # quit
            elif cmd == ord("q"):
                break

            # auto_alpha
            if PressKeyManager.get_instance('a').is_auto_state:
                output = self._auto_alpha(output, raw_g_mean, 'a')

            # cv2.imshow("output", output)
