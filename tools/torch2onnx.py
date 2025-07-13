import torch

from easydemo.models.ezmodel import Unet4to4, Unet3to3, Unet21to3

if __name__ == "__main__":
    task = 'rgb2rgb-torch-3dnr'
    dummy_input = torch.randn(1, 21, 528, 928)
    torch_model = Unet21to3()
    if task == 'rgb2rgb-torch-3dnr':
        state_dict = torch.load('weights/rgb2rgb-torch-3dnr/Unet21to3-8ch.ckpt')["state_dict"]

        for key in list(state_dict.keys()):
            state_dict[key.replace("model.", "")] = state_dict.pop(key)

        torch_model.load_state_dict(state_dict)
    else:
        torch_model.load_state_dict(torch.load('weights/rgb2rgb-torch-3dnr/Unet21to3-8ch.ckpt'))
    for i in [0.4, 0.8, 1.2, 1.6]:
        torch_model._set_alpha(i)
        torch.onnx.export(
                torch_model,
                dummy_input,
                f'weights/rgb2rgb-torch-3dnr/Unet21to3-8ch_alpha-{i}.onnx',
                opset_version=13,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}})

