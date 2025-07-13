import time
from typing import Callable, Dict, List, Optional, Sequence, Union

from .base import BaseTransform
from ..registry import TRANSFORMS
from ..press_key import PressKeyManager

# Define type of transform or transform config
Transform = Union[Dict, Callable[[Dict], Dict]]


@TRANSFORMS.register_module()
class Compose(BaseTransform):
    """Compose multiple transforms sequentially.

    Args:
        transforms (list[dict | callable]): Sequence of transform object or
            config dict to be composed.

    Examples:
        >>> pipeline = [
        >>>     dict(type='Compose',
        >>>         transforms=[
        >>>             dict(type='LoadImageFromFile'),
        >>>             dict(type='Normalize')
        >>>         ]
        >>>     )
        >>> ]
    """

    def __init__(self, transforms: Union[Transform, Sequence[Transform]]):
        super().__init__()

        if not isinstance(transforms, Sequence):
            transforms = [transforms]
        self.transforms: List = []
        self.transforms_name: List = []
        for transform in transforms:
            if isinstance(transform, dict):
                self.transforms_name.append(transform['type'])
                transform = TRANSFORMS.build(transform)
                self.transforms.append(transform)
            elif callable(transform):
                self.transforms.append(transform)
            else:
                raise TypeError('transform must be callable or a dict, but got'
                                f' {type(transform)}')

    def __iter__(self):
        """Allow easy iteration over the transform sequence."""
        return iter(self.transforms)

    def transform(self, results: Dict) -> Optional[Dict]:
        """Call function to apply transforms sequentially.

        Args:
            results (dict): A result dict contains the results to transform.

        Returns:
            dict or None: Transformed results.
        """
        for t, n in zip(self.transforms, self.transforms_name):
            start = time.time()
            results = t(results)  # type: ignore
            if 'inference_time' in results:
                self._logger(f'{n},cuda,{results["inference_time"]:.4f}\n')
                del results["inference_time"]
            else:
                self._logger(f'{n},time,{(time.time() - start) * 1000:.4f}\n')
            if results is None:
                return None
        return results
    
    def _logger(self, string):
        with open("log.csv", "a+t") as f:
            f.write(f"{string}")

    def __repr__(self):
        """Compute the string representation."""
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += f'\n    {t}'
        format_string += '\n)'
        return format_string


@TRANSFORMS.register_module()
class BaseTransformWrapper(BaseTransform):
    def __init__(self,
                 name: Optional[str] = None,
                 preprocessing: Optional[
                     Union[Transform, Sequence[Transform]]] = None,
                 postprocessing: Optional[
                     Union[Transform, Sequence[Transform]]] = None):
        super().__init__()

        self.name = name if name is not None else self.__class__.__name__
        # init preprocessing, postprocessing
        self.preprocessing = Compose(preprocessing) \
            if preprocessing is not None else None
        self.postprocessing = Compose(postprocessing) \
            if postprocessing is not None else None

    def transform_core(self, results: Dict) -> Optional[Dict]:
        return results

    def transform(self, results: Dict) -> Optional[Dict]:
        """Call function to apply transforms sequentially.

        Args:
            results (dict): A result dict contains the results to transform.

        Returns:
            dict or None: Transformed results.
        """
        if self.preprocessing is not None:
            results = self.preprocessing(results)
        results = self.transform_core(results)
        if self.postprocessing is not None:
            results = self.postprocessing(results)
        return results

    def __repr__(self):
        """Compute the string representation."""
        format_string = self.name + '('
        format_string += f'\n    preprocessing={self.preprocessing}'
        format_string += f'\n    postprocessing={self.postprocessing}'
        format_string += '\n)'
        return format_string


@TRANSFORMS.register_module()
class PressKeyWrapper(BaseTransformWrapper):
    def __init__(self,
                 key: str,
                 init_state: int,
                 num_states: int,
                 custom_state_name: List[str] = None,
                 use_auto_state: bool = False,
                 preprocessing: Optional[
                     Union[Transform, Sequence[Transform]]] = None,
                 postprocessing: Optional[
                     Union[Transform, Sequence[Transform]]] = None):
        super().__init__(
            name=self.__class__.__name__,
            preprocessing=preprocessing,
            postprocessing=postprocessing)

        self.key = key
        self.init_state = init_state
        self.num_states = num_states
        self.custom_state_name = custom_state_name
        self.use_auto_state = use_auto_state
        if self.use_auto_state:
            self.num_states += 1

        # check whether key is used
        assert not PressKeyManager.check_instance_created(key), \
            f"key \'{key}\' is used."
        # Add Key in key manager
        PressKeyManager.get_instance(key,
                                     num_states=self.num_states,
                                     init_state=init_state,
                                     custom_state_name=custom_state_name,
                                     use_auto_state=use_auto_state)

    def current_state(self):
        return PressKeyManager.get_instance(self.key).state

    def __repr__(self):
        """Compute the string representation."""
        format_string = self.name + '('
        format_string += f'\n    key={self.key}, '
        format_string += f'init_state={self.init_state}, '
        format_string += f'num_states={self.num_states}'
        format_string += f'custom_state_name={self.custom_state_name}'
        format_string += f'use_auto_state={self.use_auto_state}'
        format_string += f'\n    preprocessing={self.preprocessing}'
        format_string += f'\n    postprocessing={self.postprocessing}'
        format_string += '\n)'
        return format_string


@TRANSFORMS.register_module()
class OnOffPressKeyWrapper(PressKeyWrapper):
    def __init__(self,
                 key: str,
                 custom_state_name: List[str] = None,
                 init_state: Union[str, int] = 'on',
                 transform: Optional[
                     Union[Transform, Sequence[Transform]]] = None,
                 transform_off: Optional[
                     Union[Transform, Sequence[Transform]]] = None,
                 preprocessing: Optional[
                     Union[Transform, Sequence[Transform]]] = None,
                 postprocessing: Optional[
                     Union[Transform, Sequence[Transform]]] = None):

        if isinstance(init_state, str):
            if init_state.lower() == 'on':
                init_state = 1
            elif init_state.lower() == 'off':
                init_state = 0
            else:
                raise ValueError('init_state must be on/off, but got'
                                 f' {init_state}')
        elif isinstance(init_state, int):
            init_state = init_state % 2
        else:
            raise TypeError('init_state must be string or a int, but got'
                            f' {type(init_state)}')
        num_states = 2

        super().__init__(
            key=key,
            custom_state_name=custom_state_name,
            init_state=init_state,
            num_states=num_states,
            preprocessing=preprocessing,
            postprocessing=postprocessing)

        # build transform_on, transform_off
        self.transform_on = Compose(transform) \
            if transform is not None else None
        self.transform_off = Compose(transform_off) \
            if transform_off is not None else None

    def _transform_on(self, results: Dict) -> Optional[Dict]:
        if self.transform_on is not None:
            results = self.transform_on(results)
        return results

    def _transform_off(self, results: Dict) -> Optional[Dict]:
        if self.transform_off is not None:
            results = self.transform_off(results)
        return results

    def transform_core(self, results: Dict) -> Optional[Dict]:
        current_state = self.current_state()
        if current_state == 1:
            results = self._transform_on(results)
        elif current_state == 0:
            results = self._transform_off(results)
        else:
            raise ValueError('on/off state must be 0 or 1, but got'
                             f' {current_state}')
        return results

    def __repr__(self):
        """Compute the string representation."""
        format_string = self.name + '('
        format_string += f'\n    key={self.key}, '
        format_string += f'init_state={self.init_state}, '
        format_string += f'num_states={self.num_states}'
        format_string += f'custom_state_name={self.custom_state_name}'
        format_string += f'\n    transform_on={self.transform_on}'
        format_string += f'\n    transform_off={self.transform_off}'
        format_string += f'\n    preprocessing={self.preprocessing}'
        format_string += f'\n    postprocessing={self.postprocessing}'
        format_string += '\n)'
        return format_string


@TRANSFORMS.register_module()
class GatePressKeyWrapper(PressKeyWrapper):
    def __init__(self,
                 key: str,
                 custom_state_name: List[str] = None,
                 init_state: int = 0,
                 use_off_state: bool = False,
                 use_auto_state: bool = False,
                 transforms: Optional[
                     Union[Transform, Sequence[Transform]]] = None,
                 preprocessing: Optional[
                     Union[Transform, Sequence[Transform]]] = None,
                 postprocessing: Optional[
                     Union[Transform, Sequence[Transform]]] = None):

        if not isinstance(transforms, Sequence):
            transforms = [transforms]

        if use_off_state:
            transforms.append(None)
        self.use_off_state = use_off_state

        num_states = len(transforms)
        init_state = init_state % num_states

        super().__init__(
            key=key,
            custom_state_name=custom_state_name,
            init_state=init_state,
            num_states=num_states,
            use_auto_state=use_auto_state,
            preprocessing=preprocessing,
            postprocessing=postprocessing)

        # build transforms
        self.transforms: List = []
        for transform in transforms:
            if isinstance(transform, dict):
                transform = TRANSFORMS.build(transform)
                self.transforms.append(transform)
            elif isinstance(transform, Sequence):
                transform = Compose(transform)
                self.transforms.append(transform)
            elif callable(transform):
                self.transforms.append(transform)
            elif transform is None:
                self.transforms.append(None)
            else:
                raise TypeError('transform must be sequence, callable, a dict'
                                f' or none type , but got {type(transform)}')

    def transform_core(self, results: Dict) -> Optional[Dict]:
        current_state = self.current_state()
        current_transform = self.transforms[current_state]
        if current_transform is not None:
            results = current_transform(results)
        return results

    def __repr__(self):
        """Compute the string representation."""
        format_string = self.name + '('
        format_string += f'\n    key={self.key}, '
        format_string += f'init_state={self.init_state}, '
        format_string += f'num_states={self.num_states}, '
        format_string += f'use_off_state={self.use_off_state}'
        format_string += f'use_auto_state={self.use_auto_state}'
        format_string += f'custom_state_name={self.custom_state_name}'
        format_string += f'use_auto_state={self.use_auto_state}'
        for t in self.transforms:
            format_string += f'\n    {t}'
        format_string += f'\n    preprocessing={self.preprocessing}'
        format_string += f'\n    postprocessing={self.postprocessing}'
        format_string += '\n)'
        return format_string
