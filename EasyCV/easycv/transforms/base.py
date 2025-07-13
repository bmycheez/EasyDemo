from abc import ABCMeta, abstractmethod
from typing import Dict, Sequence, Tuple


class BaseTransform(metaclass=ABCMeta):
    """Base class for all transformations."""

    def __call__(self,
                 results: Dict) -> Dict:

        return self.transform(results)

    @abstractmethod
    def transform(self,
                  results: Dict) -> Dict:
        """The transform function. All subclass of BaseTransform should
        override this method.

        This function takes the result dict as the input, and can add new
        items to the dict or modify existing items in the dict. And the result
        dict will be returned in the end, which allows to concate multiple
        transforms into a pipeline.

        Args:
            results (dict): The result dict.

        Returns:
            dict: The result dict.
        """


class BaseMultiTransform(BaseTransform, metaclass=ABCMeta):
    """Base class for multiple transformations."""

    def __init__(self,
                 keys: Sequence[str]) -> None:
        super().__init__()
        self.keys = keys

    def transform(self,
                  results: Dict) -> Dict:
        transform_info = dict()
        results, transform_info = self.transform_pre(results, transform_info)
        for key in self.keys:
            results, transform_info = \
                self.transform_key(results, transform_info, key)
        results, transform_info = self.transform_post(results, transform_info)
        return results

    def transform_pre(self,
                      results: Dict,
                      transform_info: Dict) -> Tuple[Dict, Dict]:
        return results, transform_info

    def transform_post(self,
                       results: Dict,
                       transform_info: Dict) -> Tuple[Dict, Dict]:
        return results, transform_info

    @abstractmethod
    def transform_key(self,
                      results: Dict,
                      transform_info: Dict,
                      key: str) -> Tuple[Dict, Dict]:
        pass
