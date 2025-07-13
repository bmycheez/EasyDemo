from .registry import Registry

TRANSFORMS = Registry('transform', scope='easycv',
                      locations=['easycv.transforms'])
