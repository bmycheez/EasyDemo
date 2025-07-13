from easycv.registry import Registry
from easycv.registry import TRANSFORMS as EASYCV_TRANSFORMS

TRANSFORMS = Registry(
    'transform',
    parent=EASYCV_TRANSFORMS,
    locations=['easydemo.transforms'])
MODELS = Registry('model')  # TODO : EasyCV
RUNNERS = Registry('runner')
