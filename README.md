# EasyDemo
## Installation
1. Download EasyDemo
```
git clone --recursive git@github.com:ENERZAi/EasyDemo.git
```
2. You Need EasyCV to install EasyDemo!
```
cd EasyCV
pip install -e .
```
3. install EasyDemo
```
cd ..
pip install -e.
```
## Run Demo
```
python tools/demo.py {config path}
```
## TRT Build
```
/usr/src/tensorrt/bin/trtexec --onnx={onnx path} --saveEngine={save path} {--fp16 --int8}
```