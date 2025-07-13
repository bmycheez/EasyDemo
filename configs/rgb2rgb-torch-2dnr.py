# runner_type = 'FPSTestRunner'
fps = 33
gain_list = [10, 100, 200, 300]
exposure_list = [0.5, 1, 5, 10, 15, 20, 25, 30]
gain_index = 3
exposure_index = 7
use_auto_exposure = True
auto_exposure_target = 2400
auto_exposure_range = 400
auto_exposure_speed = 0.005
pipeline = [
    # Preprocessing
    dict(type='GetRawFramefromCamera', size=(1920, 1080)),
    # dict(type='GetRandomFrame', size=(1920, 1080)),
    dict(type='NumpyUnsqueeze', keys=['img_frame']),
    dict(type='GetRAWGreenMean', key='img_frame'),
    dict(type='ToTensor', keys=['img_frame'], device='cuda'),
    dict(type='KeyMapping', mapping={'img_frame': 'img'}),
    # ISP
    dict(type='BlackWhiteLevel', key='img', black_level=240, white_level=2**12),
    dict(type='CropTensor', keys=['img'], crop=(32, 32, 12, 12)),
    dict(type='BayerFormatting4Channel', key='img', input='gbrg', output='gbrg'),
    dict(type='GetOriginMean'),
    dict(type='MeanBaseBrightScaling', scale=0.1, patch_size=None),
    dict(type='KeyMapping', mapping={'img': 'noisy'}),
    # RAW2RGB
    dict(type='NaiveDemosaicing', keys=['noisy'], scale=0.1),
    # dict(type='Demosaicing5x5Malvar', key='noisy'),
    # 2DNR
    dict(type='GatePressKeyWrapper',
         key='a',
         init_state=4,
         use_auto_state=True,
         custom_state_name=["0.4", "0.8", "1.2", "1.6"],
         transforms=[
             dict(type='TorchModelWrapper', model=dict(type='Unet3to3'), device='cuda',
                  alpha=0.4, load_from='weights/rgb2rgb-torch-2dnr/Unet3to3-4ch.pt',
                  input_key='noisy', output_key='denoised'),
             dict(type='TorchModelWrapper', model=dict(type='Unet3to3'), device='cuda',
                  alpha=0.8, load_from='weights/rgb2rgb-torch-2dnr/Unet3to3-4ch.pt',
                  input_key='noisy', output_key='denoised'),
             dict(type='TorchModelWrapper', model=dict(type='Unet3to3'), device='cuda',
                  alpha=1.2, load_from='weights/rgb2rgb-torch-2dnr/Unet3to3-4ch.pt',
                  input_key='noisy', output_key='denoised'),
             dict(type='TorchModelWrapper', model=dict(type='Unet3to3'), device='cuda',
                  alpha=1.6, load_from='weights/rgb2rgb-torch-2dnr/Unet3to3-4ch.pt',
                  input_key='noisy', output_key='denoised')]),
    # Postprocessing
    dict(type='ToNumpyImage', keys=['noisy', 'denoised']),
    dict(type='CvtColor', keys=['noisy', 'denoised'], input_type='bgr', output_type='rgb'),
    # Visualization
    dict(type='OnOffPressKeyWrapper',
         key='d',
         init_state='off',
         transform=dict(type='DarkMode', digital_gain=0.1, dark_mode=True),
         transform_off=dict(type='DarkMode', digital_gain=0.1, dark_mode=False)),
    dict(type='OnOffPressKeyWrapper',
         key='z',
         init_state='off',
         transform=[
             dict(type='CropNumpy', keys=['noisy', 'denoised'], crop=(300, 300, 180, 180)),
             dict(type='NumpyResize', keys=['noisy', 'denoised'], size=(928, 528))],
         transform_off=dict(type='NumpyResize', keys=['noisy', 'denoised'], size=(928, 528))),
    dict(type='ConcatNumpyImage', keys=['noisy', 'denoised'], output_key='output')]
