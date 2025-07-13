from typing import Callable, Dict, List, Optional, Sequence, Union

import torch
import torch.nn as nn
import tensorrt as trt

from ..registry import TRANSFORMS, MODELS
from .wrappers import BaseTransformWrapper

# Define type of transform or transform config
Transform = Union[Dict, Callable[[Dict], Dict]]


def torch_dtype_from_trt(dtype):
    if dtype == trt.int8:
        return torch.int8
    elif dtype == trt.bool:
        return torch.bool
    elif dtype == trt.int32:
        return torch.int32
    elif dtype == trt.float16:
        return torch.float16
    elif dtype == trt.float32:
        return torch.float32
    else:
        raise TypeError("%s is not supported by torch" % dtype)


def torch_device_from_trt(device):
    if device == trt.TensorLocation.DEVICE:
        return torch.device("cuda")
    elif device == trt.TensorLocation.HOST:
        return torch.device("cpu")
    else:
        raise TypeError("%s is not supported by torch" % device)


def load_trt_engine(engine_path: str):
    with open(engine_path, "rb") as stream:
        serialized = stream.read()
    logger = trt.Logger(trt.Logger.ERROR)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(serialized)
    return engine


@TRANSFORMS.register_module()
class TRTModelWrapper(BaseTransformWrapper):
    def __init__(self,
                 load_from: Optional[str] = None,
                 input_key: str = 'noisy',
                 output_key: str = 'denoised',
                 preprocessing: Optional[
                     Union[Transform, Sequence[Transform]]] = None,
                 postprocessing: Optional[
                     Union[Transform, Sequence[Transform]]] = None):
        super().__init__(
            name=self.__class__.__name__,
            preprocessing=preprocessing,
            postprocessing=postprocessing)

        self.load_from = load_from
        self.input_key = input_key
        self.output_key = output_key

        self.model = load_trt_engine(load_from)
        self.model_ctx = self.model.create_execution_context()
        
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)

    @torch.no_grad()
    def run_engine(self, input_tensor):
        idx = self.model.get_binding_index("output")
        device = torch_device_from_trt(self.model.get_location(idx))
        dtype = torch_dtype_from_trt(self.model.get_binding_dtype(idx))
        shape = tuple(self.model.get_binding_shape(idx))

        input_tensor = input_tensor.to(device=device, dtype=dtype)

        bindings = [None] * 2
        bindings[0] = input_tensor.contiguous().data_ptr()

        output = torch.empty(size=shape, dtype=dtype, device=device)
        bindings[idx] = output.data_ptr()

        self.model_ctx.execute_async_v2(
            bindings, torch.cuda.current_stream().cuda_stream
        )
        torch.cuda.synchronize()
        return output

    def transform_core(self, results: Dict) -> Optional[Dict]:
        self.start_event.record()
        results[self.output_key] = self.run_engine(results[self.input_key])
        self.end_event.record()
        torch.cuda.synchronize()
        results['inference_time'] = self.start_event.elapsed_time(self.end_event)
        return results

    def __repr__(self):
        """Compute the string representation."""
        format_string = self.name + '('
        format_string += f'\n    load_from={self.preprocessing}, '
        format_string += f'input_key={self.input_key}'
        format_string += f'output_key={self.output_key}'
        format_string += f'\n    preprocessing={self.preprocessing}'
        format_string += f'\n    postprocessing={self.postprocessing}'
        format_string += '\n)'
        return format_string


@TRANSFORMS.register_module()
class TorchModelWrapper(BaseTransformWrapper):
    def __init__(self,
                 model: Union[nn.Module, Dict],
                 load_from: Optional[str] = None,
                 device: str = 'cpu',
                 alpha: float = 1.0,
                 input_key: str = 'img',
                 output_key: str = 'img',
                 preprocessing: Optional[
                     Union[Transform, Sequence[Transform]]] = None,
                 postprocessing: Optional[
                     Union[Transform, Sequence[Transform]]] = None):
        super().__init__(
            name=self.__class__.__name__,
            preprocessing=preprocessing,
            postprocessing=postprocessing)

        self.load_from = load_from
        self.input_key = input_key
        self.output_key = output_key
        self.device = device

        if isinstance(model, nn.Module):
            self.model = model
        elif isinstance(model, dict):
            self.model = MODELS.build(model)
        else:
            raise TypeError('model should be a nn.Module object or dict, '
                            f'but got {model}')
        
        self.type = model["type"]
        if load_from is not None and self.type != 'Unet21to3':
            self.model.load_state_dict(torch.load(load_from))
        elif load_from is not None and self.type == 'Unet21to3':
            state_dict = torch.load(load_from)["state_dict"]

            for key in list(state_dict.keys()):
                state_dict[key.replace("model.", "")] = state_dict.pop(key)

            self.model.load_state_dict(state_dict)
        self.model.to(device)
        self.model.eval()
        self.model._set_alpha(alpha)

    def transform_core(self, results: Dict) -> Optional[Dict]:
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        with torch.no_grad():
            start_event.record()
            results[self.output_key] = self.model(results[self.input_key])
            end_event.record()
        torch.cuda.synchronize()
        results['inference_time'] = start_event.elapsed_time(end_event)
        return results

    def __repr__(self):
        """Compute the string representation."""
        format_string = self.name + '('
        format_string += f'\n    load_from={self.preprocessing}, '
        format_string += f'input_key={self.input_key}'
        format_string += f'output_key={self.output_key}'
        format_string += f'\n    model={self.model}'
        format_string += f'\n    preprocessing={self.preprocessing}'
        format_string += f'\n    postprocessing={self.postprocessing}'
        format_string += '\n)'
        return format_string
