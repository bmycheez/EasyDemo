from typing import Callable, Dict, List, Optional, Sequence, Union

import mmengine
import numpy as np

from .base import BaseTransform
from easycv.registry import TRANSFORMS
from .utils import cache_randomness

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
        for transform in transforms:
            if isinstance(transform, dict):
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
        for t in self.transforms:
            results = t(results)  # type: ignore
            if results is None:
                return None
        return results

    def __repr__(self):
        """Compute the string representation."""
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += f'\n    {t}'
        format_string += '\n)'
        return format_string


@TRANSFORMS.register_module()
class RandomChoice(BaseTransform):
    """Process data with a randomly chosen transform from given candidates.

    Args:
        transforms (list[list]): A list of transform candidates, each is a
            sequence of transforms.
        prob (list[float], optional): The probabilities associated
            with each pipeline. The length should be equal to the pipeline
            number and the sum should be 1. If not given, a uniform
            distribution will be assumed.

    Examples:
        >>> # config
        >>> pipeline = [
        >>>     dict(type='RandomChoice',
        >>>         transforms=[
        >>>             [dict(type='RandomHorizontalFlip')],  # subpipeline 1
        >>>             [dict(type='RandomRotate')],  # subpipeline 2
        >>>         ]
        >>>     )
        >>> ]
    """

    def __init__(self,
                 transforms: List[Union[Transform, List[Transform]]],
                 prob: Optional[List[float]] = None):

        super().__init__()

        if prob is not None:
            assert mmengine.is_seq_of(prob, float)
            assert len(transforms) == len(prob), \
                '``transforms`` and ``prob`` must have same lengths. ' \
                f'Got {len(transforms)} vs {len(prob)}.'
            assert sum(prob) == 1

        self.prob = prob
        self.transforms = [Compose(transforms) for transforms in transforms]

    def __iter__(self):
        return iter(self.transforms)

    @cache_randomness
    def random_pipeline_index(self) -> int:
        """Return a random transform index."""
        indices = np.arange(len(self.transforms))
        return np.random.choice(indices, p=self.prob)

    def transform(self, results: Dict) -> Optional[Dict]:
        """Randomly choose a transform to apply."""
        idx = self.random_pipeline_index()
        return self.transforms[idx](results)

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(transforms = {self.transforms}'
        repr_str += f'prob = {self.prob})'
        return repr_str


@TRANSFORMS.register_module()
class RandomApply(BaseTransform):
    """Apply transforms randomly with a given probability.

    Args:
        transforms (list[dict | callable]): The transform or transform list
            to randomly apply.
        prob (float): The probability to apply transforms. Default: 0.5

    Examples:
        >>> # config
        >>> pipeline = [
        >>>     dict(type='RandomApply',
        >>>         transforms=[dict(type='HorizontalFlip')],
        >>>         prob=0.3)
        >>> ]
    """

    def __init__(self,
                 transforms: Union[Transform, List[Transform]],
                 prob: float = 0.5):

        super().__init__()
        self.prob = prob
        self.transforms = Compose(transforms)

    def __iter__(self):
        return iter(self.transforms)

    @cache_randomness
    def random_apply(self) -> bool:
        """Return a random bool value indicating whether apply the
        transform."""
        return np.random.rand() < self.prob

    def transform(self, results: Dict) -> Optional[Dict]:
        """Randomly apply the transform."""
        if self.random_apply():
            return self.transforms(results)  # type: ignore
        else:
            return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(transforms = {self.transforms}'
        repr_str += f', prob = {self.prob})'
        return repr_str
