import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import MODELS


# raw2raw-torch-2dnr
@MODELS.register_module()
class Unet4to4(nn.Module):
    def __init__(self, k=8, bias=False):
        super().__init__()
        # self.quant = torch.quantization.QuantStub()
        self.conv1_1 = nn.Conv2d(4, 1 * k, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=bias)
        self.conv1_2 = nn.Conv2d(1 * k, 1 * k, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=bias)
        self.conv2_1 = nn.Conv2d(1 * k, 2 * k, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=bias)
        self.conv2_2 = nn.Conv2d(2 * k, 2 * k, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=bias)
        self.conv3_1 = nn.Conv2d(2 * k, 4 * k, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=bias)
        self.conv3_2 = nn.Conv2d(4 * k, 4 * k, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=bias)
        self.conv4_1 = nn.Conv2d(4 * k, 8 * k, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=bias)
        self.conv4_2 = nn.Conv2d(8 * k, 8 * k, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=bias)
        self.conv5_1 = nn.Conv2d(8 * k, 16 * k, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=bias)
        self.conv5_2 = nn.Conv2d(16 * k, 16 * k, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=bias)
        self.upv6 = nn.ConvTranspose2d(16 * k, 8 * k, kernel_size=(2, 2), stride=(2, 2))
        self.conv6_1 = nn.Conv2d(16 * k, 8 * k, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=bias)
        self.conv6_2 = nn.Conv2d(8 * k, 8 * k, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=bias)
        self.upv7 = nn.ConvTranspose2d(8 * k, 4 * k, kernel_size=(2, 2), stride=(2, 2))
        self.conv7_1 = nn.Conv2d(8 * k, 4 * k, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=bias)
        self.conv7_2 = nn.Conv2d(4 * k, 4 * k, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=bias)
        self.upv8 = nn.ConvTranspose2d(4 * k, 2 * k, kernel_size=(2, 2), stride=(2, 2))
        self.conv8_1 = nn.Conv2d(4 * k, 2 * k, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=bias)
        self.conv8_2 = nn.Conv2d(2 * k, 2 * k, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=bias)
        self.upv9 = nn.ConvTranspose2d(2 * k, 1 * k, kernel_size=(2, 2), stride=(2, 2))
        self.conv9_1 = nn.Conv2d(2 * k, 1 * k, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=bias)
        self.conv9_2 = nn.Conv2d(1 * k, 1 * k, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=bias)
        self.conv10_1 = nn.Conv2d(1 * k, 4, kernel_size=(1, 1), stride=(1, 1), bias=bias)
        self.pool1_1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.pool1_2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.pool1_3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.l_relu = nn.LeakyReLU(0.2)
        # self.dequant = torch.quantization.DeQuantStub()

        self.sum_1 = torch.nn.quantized.FloatFunctional()
        self.sum_2 = torch.nn.quantized.FloatFunctional()
        self.sum_3 = torch.nn.quantized.FloatFunctional()
        self.sum_4 = torch.nn.quantized.FloatFunctional()
        self.sum_5 = torch.nn.quantized.FloatFunctional()
        self.sum_6 = torch.nn.quantized.FloatFunctional()

        self.cat_1 = torch.nn.quantized.FloatFunctional()
        self.cat_2 = torch.nn.quantized.FloatFunctional()
        self.cat_3 = torch.nn.quantized.FloatFunctional()
        self.cat_4 = torch.nn.quantized.FloatFunctional()

        self.alpha = None
    
    def _set_alpha(self, alpha):
        self.alpha = alpha
        
    def forward(self, x, alpha = 1.6):
        # x = self.quant(x)
        # x = x[:, :, 12:-12, :]
        # x = x.sub_(240).div_(2**12)
        # x = x.clamp(0, 1)
        # x = torch.cat((x[:, :, 0::2,0::2],
        #                 x[:, :, 0::2,1::2],
        #                 x[:, :, 1::2,0::2],
        #                 x[:, :, 1::2,1::2]), dim=1)
        
        # scale = 0.3/torch.mean(x)
        # x = x*scale
        input_x = x

        x = x - torch.mean(x, dim=[1,2, 3], keepdim=True)

        conv1_1 = self.conv1_1(x, )
        # mul = torch.mul(0.2, conv1_1)
        max_1 = self.l_relu(conv1_1)
        conv1_2 = self.conv1_2(max_1, )
        # mul_1 = torch.mul(0.2, conv1_2)
        max_2 = self.l_relu(conv1_2)
        max_2 = self.sum_1.add(max_1, max_2)
        pool1 = self.pool1(max_2, )
        conv2_1 = self.conv2_1(pool1, )
        # mul_2 = torch.mul(0.2, conv2_1)
        max_3 = self.l_relu(conv2_1)
        conv2_2 = self.conv2_2(max_3, )
        # mul_3 = torch.mul(0.2, conv2_2)
        max_4 = self.l_relu(conv2_2)
        max_4 = self.sum_2.add(max_4, max_3)
        pool1_1 = self.pool1_1(max_4, )
        conv3_1 = self.conv3_1(pool1_1, )
        # mul_4 = torch.mul(0.2, conv3_1)
        max_5 = self.l_relu(conv3_1)
        conv3_2 = self.conv3_2(max_5, )
        # mul_5 = torch.mul(0.2, conv3_2)
        max_6 = self.l_relu(conv3_2)
        max_6 = self.sum_3.add(max_6, max_5)
        pool1_2 = self.pool1_2(max_6, )
        conv4_1 = self.conv4_1(pool1_2, )
        # mul_6 = torch.mul(0.2, conv4_1)
        max_7 = self.l_relu(conv4_1)
        conv4_2 = self.conv4_2(max_7, )
        # mul_7 = torch.mul(0.2, conv4_2)
        max_8 = self.l_relu(conv4_2)
        max_8 = self.sum_4.add(max_8, max_7)
        pool1_3 = self.pool1_3(max_8, )
        conv5_1 = self.conv5_1(pool1_3, )
        # mul_8 = torch.mul(0.2, conv5_1)
        max_9 = self.l_relu(conv5_1)
        conv5_2 = self.conv5_2(max_9, )
        # mul_9 = torch.mul(0.2, conv5_2)
        max_10 = self.l_relu(conv5_2)
        max_10 = self.sum_5.add(max_10, conv5_2)
        upv6 = self.upv6(max_10, )
        cat = self.cat_1.cat([upv6, max_8], 1)
        conv6_1 = self.conv6_1(cat, )
        # mul_10 = torch.mul(0.2, conv6_1)
        max_11 = self.l_relu(conv6_1)
        conv6_2 = self.conv6_2(max_11, )
        # mul_11 = torch.mul(0.2, conv6_2)
        max_12 = self.l_relu(conv6_2)
        upv7 = self.upv7(max_12, )
        cat_1 = self.cat_2.cat([upv7, max_6], 1)
        conv7_1 = self.conv7_1(cat_1, )
        # mul_12 = torch.mul(0.2, conv7_1)
        max_13 = self.l_relu( conv7_1)
        conv7_2 = self.conv7_2(max_13, )
        # mul_13 = torch.mul(0.2, conv7_2)
        max_14 = self.l_relu(conv7_2)
        upv8 = self.upv8(max_14, )
        cat_2 = self.cat_3.cat([upv8, max_4], 1)
        conv8_1 = self.conv8_1(cat_2, )
        # mul_14 = torch.mul(0.2, conv8_1)
        max_15 = self.l_relu(conv8_1)
        conv8_2 = self.conv8_2(max_15, )
        # mul_15 = torch.mul(0.2, conv8_2)
        max_16 = self.l_relu(conv8_2)
        upv9 = self.upv9(max_16, )
        cat_3 = self.cat_4.cat([upv9, max_2], 1)
        conv9_1 = self.conv9_1(cat_3, )
        # mul_16 = torch.mul(0.2, conv9_1)
        max_17 = self.l_relu(conv9_1)
        conv9_2 = self.conv9_2(max_17, )
        # mul_17 = torch.mul(0.2, conv9_2)
        max_18 = self.l_relu(conv9_2)
        conv10_1 = self.conv10_1(max_18, )

        noise = conv10_1 - torch.mean(conv10_1, dim=[2, 3], keepdim=True)

        # un_norm
        out = self.sum_6.add(noise*self.alpha, input_x)
        # out = torch.cat((out, input_x), dim=0)#/scale
        # out = self.isp(out)
        # r = r/g.mean()
        # out = out.permute(2, 0, 3, 1)#.mul(1.5).clamp(0, 1).pow(1/2.2).mul_(255.0)
        # out = out.reshape(out.shape[0], -1, out.shape[-1])
        return out
    
    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for o_name, o_param in own_state.items():
            for s_name, s_param in state_dict.items():
                # if o_name.find('res_enc')>=0:
                #     o_name = o_name[4:]
                if o_name == s_name and o_param.shape == s_param.shape:
                    # print('{} -> {}'.format(o_name, s_name))
                    # print('{} -> {}'.format(o_param.shape, s_param.shape))
                    own_state[o_name].copy_(state_dict[s_name])
                    break


# rgb2rgb-torch-2dnr
@MODELS.register_module()
class Unet3to3(nn.Module):
    def __init__(self,
                 k: int = 4,
                 bias: bool = True):
        super().__init__()
        self.conv1_1 = nn.Conv2d(
            3, 2 * k,
            kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=bias)
        self.conv1_2 = nn.Conv2d(
            2 * k, 2 * k,
            kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=bias)
        self.conv2_1 = nn.Conv2d(
            2 * k, 2 * k,
            kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=bias)
        self.conv2_2 = nn.Conv2d(
            2 * k, 2 * k,
            kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=bias)
        self.conv3_1 = nn.Conv2d(
            2 * k, 4 * k,
            kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=bias)
        self.conv3_2 = nn.Conv2d(
            4 * k, 4 * k,
            kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=bias)
        self.conv4_1 = nn.Conv2d(
            4 * k, 8 * k,
            kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=bias)
        self.conv4_2 = nn.Conv2d(
            8 * k, 8 * k,
            kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=bias)
        self.conv5_1 = nn.Conv2d(
            8 * k, 8 * k,
            kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=bias)
        self.conv5_2 = nn.Conv2d(
            8 * k, 8 * k,
            kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=bias)
        self.upv6 = nn.ConvTranspose2d(
            8 * k, 8 * k, kernel_size=(2, 2), stride=(2, 2))
        self.conv6_1 = nn.Conv2d(
            16 * k, 8 * k,
            kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=bias)
        self.conv6_2 = nn.Conv2d(
            8 * k, 8 * k,
            kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=bias)
        self.upv7 = nn.ConvTranspose2d(
            8 * k, 4 * k,
            kernel_size=(2, 2), stride=(2, 2))
        self.conv7_1 = nn.Conv2d(
            8 * k, 4 * k,
            kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=bias)
        self.conv7_2 = nn.Conv2d(
            4 * k, 4 * k,
            kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=bias)
        self.upv8 = nn.ConvTranspose2d(
            4 * k, 2 * k, kernel_size=(2, 2), stride=(2, 2))
        self.conv8_1 = nn.Conv2d(
            4 * k, 2 * k,
            kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=bias)
        self.conv8_2 = nn.Conv2d(
            2 * k, 2 * k,
            kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=bias)
        self.upv9 = nn.ConvTranspose2d(
            2 * k, 2 * k, kernel_size=(2, 2), stride=(2, 2))
        self.conv9_1 = nn.Conv2d(
            4 * k, 2 * k,
            kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=bias)
        self.conv9_2 = nn.Conv2d(
            2 * k, 2 * k,
            kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=bias)
        self.conv10_1 = nn.Conv2d(
            2 * k, 3,
            kernel_size=(1, 1), stride=(1, 1), bias=bias)
        self.pool1_1 = nn.MaxPool2d(
            kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.pool1 = nn.MaxPool2d(
            kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.pool1_2 = nn.MaxPool2d(
            kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.pool1_3 = nn.MaxPool2d(
            kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.l_relu = nn.LeakyReLU(0.2)

        self.sum_1 = torch.nn.quantized.FloatFunctional()
        self.sum_2 = torch.nn.quantized.FloatFunctional()
        self.sum_3 = torch.nn.quantized.FloatFunctional()
        self.sum_4 = torch.nn.quantized.FloatFunctional()
        self.sum_5 = torch.nn.quantized.FloatFunctional()
        self.sum_6 = torch.nn.quantized.FloatFunctional()

        self.cat_1 = torch.nn.quantized.FloatFunctional()
        self.cat_2 = torch.nn.quantized.FloatFunctional()
        self.cat_3 = torch.nn.quantized.FloatFunctional()
        self.cat_4 = torch.nn.quantized.FloatFunctional()

        self.alpha = None
    
    def _set_alpha(self, alpha):
        self.alpha = alpha

    def forward(self,
                x: torch.Tensor,
                alpha: float = 1.0):
        conv1_1 = self.conv1_1(x)
        max_1 = self.l_relu(conv1_1)
        conv1_2 = self.conv1_2(max_1, )
        # mul_1 = torch.mul(0.2, conv1_2)
        max_2 = self.l_relu(conv1_2)
        max_2 = self.sum_1.add(max_1, max_2)
        pool1 = self.pool1(max_2, )
        conv2_1 = self.conv2_1(pool1, )
        # mul_2 = torch.mul(0.2, conv2_1)
        max_3 = self.l_relu(conv2_1)
        conv2_2 = self.conv2_2(max_3, )
        # mul_3 = torch.mul(0.2, conv2_2)
        max_4 = self.l_relu(conv2_2)
        max_4 = self.sum_2.add(max_4, max_3)
        pool1_1 = self.pool1_1(max_4, )
        conv3_1 = self.conv3_1(pool1_1, )
        # mul_4 = torch.mul(0.2, conv3_1)
        max_5 = self.l_relu(conv3_1)
        conv3_2 = self.conv3_2(max_5, )
        # mul_5 = torch.mul(0.2, conv3_2)
        max_6 = self.l_relu(conv3_2)
        max_6 = self.sum_3.add(max_6, max_5)
        pool1_2 = self.pool1_2(max_6, )
        conv4_1 = self.conv4_1(pool1_2, )
        # mul_6 = torch.mul(0.2, conv4_1)
        max_7 = self.l_relu(conv4_1)
        conv4_2 = self.conv4_2(max_7, )
        # mul_7 = torch.mul(0.2, conv4_2)
        max_8 = self.l_relu(conv4_2)
        max_8 = self.sum_4.add(max_8, max_7)
        pool1_3 = self.pool1_3(max_8, )
        conv5_1 = self.conv5_1(pool1_3, )
        # mul_8 = torch.mul(0.2, conv5_1)
        max_9 = self.l_relu(conv5_1)
        conv5_2 = self.conv5_2(max_9, )
        # mul_9 = torch.mul(0.2, conv5_2)
        max_10 = self.l_relu(conv5_2)
        max_10 = self.sum_5.add(max_10, conv5_2)
        upv6 = self.upv6(max_10, )
        cat = self.cat_1.cat([upv6, max_8], 1)
        conv6_1 = self.conv6_1(cat, )
        # mul_10 = torch.mul(0.2, conv6_1)
        max_11 = self.l_relu(conv6_1)
        conv6_2 = self.conv6_2(max_11, )
        # mul_11 = torch.mul(0.2, conv6_2)
        max_12 = self.l_relu(conv6_2)
        upv7 = self.upv7(max_12, )
        cat_1 = self.cat_2.cat([upv7, max_6], 1)
        conv7_1 = self.conv7_1(cat_1, )
        # mul_12 = torch.mul(0.2, conv7_1)
        max_13 = self.l_relu(conv7_1)
        conv7_2 = self.conv7_2(max_13, )
        # mul_13 = torch.mul(0.2, conv7_2)
        max_14 = self.l_relu(conv7_2)
        upv8 = self.upv8(max_14, )
        cat_2 = self.cat_3.cat([upv8, max_4], 1)
        conv8_1 = self.conv8_1(cat_2, )
        # mul_14 = torch.mul(0.2, conv8_1)
        max_15 = self.l_relu(conv8_1)
        conv8_2 = self.conv8_2(max_15, )
        # mul_15 = torch.mul(0.2, conv8_2)
        max_16 = self.l_relu(conv8_2)
        upv9 = self.upv9(max_16, )
        cat_3 = self.cat_4.cat([upv9, max_2], 1)
        conv9_1 = self.conv9_1(cat_3, )
        # mul_16 = torch.mul(0.2, conv9_1)
        max_17 = self.l_relu(conv9_1)
        conv9_2 = self.conv9_2(max_17, )
        # mul_17 = torch.mul(0.2, conv9_2)
        max_18 = self.l_relu(conv9_2)
        conv10_1 = self.conv10_1(max_18, )
        return conv10_1+x


# rgb2rgb-torch-3dnr
class LayerNormChannel(nn.Module):
    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels).view(1, num_channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(num_channels).view(1, num_channels, 1, 1))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight * x + self.bias
        return x

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class Block(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0., num_shift=0):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        if num_shift != 0:
            self.sg = ShiftSimpleGate(num_shift)
        else:
            self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNormChannel(c)
        self.norm2 = LayerNormChannel(c)

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        return y + x * self.gamma


class DownBlock(nn.Module):
    def __init__(self, c, type="conv"):
        super().__init__()

        self.type = type

        if type == "conv":
            self.down = nn.Conv2d(c, c*2, 2, 2)
        elif type == "maxpool":
            self.down = nn.Sequential(
                nn.Conv2d(c, c*2, 1, padding=0, bias=False),
                nn.MaxPool2d(2, 2)
            )
        elif type == "avgpool":
            self.down = nn.Sequential(
                nn.Conv2d(c, c*2, 1, padding=0, bias=False),
                nn.AvgPool2d(2, 2),
            )
        elif type == "upsample":
            self.down = nn.Sequential(
                nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True),
                nn.Conv2d(c, c*2, 1, padding=0, bias=False),
            )
        else:
            raise NotImplementedError

    def forward(self, x):
        return self.down(x)

class UpBlock(nn.Module):
    def __init__(self, c, type="upsample"):
        super().__init__()

        if type == "upsample":
            self.up = nn.Sequential(
                nn.Conv2d(c, c//2, 3, padding=1, bias=False),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            )
        elif type == "convtranspose":
            self.up = nn.ConvTranspose2d(c, c//2, 2, 2)
        elif type == "pixelshuffle":
            self.up = nn.Sequential(
                nn.Conv2d(c, c*2, 1, padding=0, bias=False),
                nn.PixelShuffle(2),
            )
        else:
            raise NotImplementedError

    def forward(self, x):
        return self.up(x)

@MODELS.register_module()
class Unet21to3(nn.Module):
    def __init__(self, in_ch=21, bias=False):
        super().__init__()

        down = "conv"
        up = "upsample"

        c = 8

        self.embed = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch, bias=True),
            nn.Conv2d(in_ch, c, 1, padding=0, bias=True),
        )

        self.enc1 = Block(c, num_shift=0)

        self.down1 = DownBlock(c, down)

        self.enc2 = Block(2*c, num_shift=0)

        self.down2 = DownBlock(2*c, down)

        self.enc3 = Block(4*c, num_shift=0)

        self.down3 = DownBlock(4*c, down)

        self.enc4 = nn.Sequential(
            Block(8*c, num_shift=0),
        )

        self.up3 = UpBlock(8*c, up)
    
        self.dec3 = Block(4*c)

        self.up2 = UpBlock(4*c, up)
  
        self.dec2 = Block(2*c)

        self.up1 = UpBlock(2*c, up)

        self.dec1 = Block(c)

        self.up0 = nn.ConvTranspose2d(c, c, 2, 2, 0, bias=False)

        self.dec_out = nn.Conv2d(c, 3, 3, 1, 1, bias=bias)
        
        self.alpha = None
    
    def _set_alpha(self, alpha):
        self.alpha = alpha

    def forward(self, x):

        x =  F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)

        x = self.embed(x)

        enc1 = self.enc1(x)
        down1 = self.down1(enc1)

        enc2 = self.enc2(down1)
        down2 = self.down2(enc2)

        enc3 = self.enc3(down2)
        down3 = self.down3(enc3)

        enc4 = self.enc4(down3)

        up3 = self.up3(enc4)
        cat3 = enc3 + up3
        dec3 = self.dec3(cat3)

        up2 = self.up2(dec3)
        cat2 = enc2 + up2
        dec2 = self.dec2(cat2)

        up1 = self.up1(dec2)
        cat1 = enc1 + up1
        dec1 = self.dec1(cat1)

        up0 = self.up0(dec1)

        dec_out = self.dec_out(up0)

        return dec_out

