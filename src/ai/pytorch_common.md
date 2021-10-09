# Pytorch常用代码总结

- [基本配置](#基本配置)
    - [导入包和版本查询](#导入包和版本查询)
    - [可复现性](#可复现性)
    - [显卡设置](#显卡设置)
- [Tensor处理](#tensor处理)
    - [张量的数据类型](#张量的数据类型)
    - [张量基本信息](#张量基本信息)


---
## 基本配置

### 导入包和版本查询
```python
import torch
import torch.nn as nn
import torchvision
print(torch.__version__)
print(torch.version.cuda)
print(torch.backends.cudnn.version())
print(torch.cuda.get_device_name(0))
```
### 可复现性
在硬件设备(CPU、GPU)不同时，完全的可复现性无法保证，即使随机种子相同。但是，在同一个设备上，应该保证可复现性。具体做法是，在程序开始的时候固定torch的随机种子，同时也把numpy的随机种子固定。
```python
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

### 显卡设置
如果只需要一张显卡
```python
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```
如果需要指定多张显卡，比如`0`, `1`号显卡。
```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
```
也可以在命令行运行代码时设置显卡：
```python
CUDA_VISIBLE_DEVICES=0,1 python train.py
```
清除显存
```python
torch.cuda.empty_cache()
```
也可以使用在命令行重置GPU的指令
```python
nvidia-smi --gpu-reset -i [gpu_id]
```
---
## Tensor处理
### 张量的数据类型
[PyTorch有10种CPU张量类型和9种GPU张量类型.](https://pytorch.org/docs/stable/tensors.html?highlight=tensor#torch.Tensor)    

|  Data type   | dtype | CPU tensor | GPU tensor |
|  ----  | ----  | ----  | ----  |
| 32-bit floating point  | torch.float32 or torch.float | torch.FloatTensor | torch.cuda.FloatTensor |
| 64-bit floating point | torch.float64 or torch.double | torch.DoubleTensor | torch.cuda.DoubleTensor |
| 16-bit floating point `[1]` | torch.float16 or torch.half | torch.HalfTensor | torch.cuda.HalfTensor |
| 16-bit floating point `[2]` | torch.bfloat16 | torch.BFloat16Tensor | torch.cuda.BFloat16Tensor |
| 32-bit complex | torch.complex32 |  |  |
| 64-bit complex | torch.complex64 |  |  |
| 128-bit complex | torch.complex128 or torch.cdouble |  |  |
| 8-bit integer (unsigned) | torch.uint8 | torch.ByteTensor | torch.cuda.ByteTensor |
| 8-bit integer (signed) | torch.int8 | torch.CharTensor | torch.cuda.CharTensor |
| 16-bit integer (signed) | torch.int16 or torch.short | torch.ShortTensor | torch.cuda.ShortTensor |
| 32-bit integer (signed) | torch.int32 or torch.int | torch.IntTensor | torch.cuda.IntTensor |
| 64-bit integer (signed) | torch.int64 or torch.long | torch.LongTensor | torch.cuda.LongTensor |
| Boolean | torch.bool | torch.BoolTensor | torch.cuda.BoolTensor |
| quantized 8-bit integer (unsigned) | torch.quint8 | torch.ByteTensor | / |
| quantized 8-bit integer (signed) | torch.qint8 | torch.CharTensor | / |
| quantized 32-bit integer (signed) | torch.qfint32 | torch.IntTensor | / |
| quantized 4-bit integer (unsigned) `[3]` | torch.quint4x2 | torch.ByteTensor | / |

> [1] Sometimes referred to as binary16: uses 1 sign, 5 exponent, and 10 significand bits. Useful when precision is important at the expense of range.    
> [2] Sometimes referred to as Brain Floating Point: uses 1 sign, 8 exponent, and 7 significand bits. Useful when range is important, since it has the same number of exponent bits as `float32`    
> [3] quantized 4-bit integer is stored as a 8-bit signed integer. Currently it’s only supported in EmbeddingBag operator.    

`torch.Tensor` is an alias for the default tensor type (`torch.FloatTensor`).

### 张量基本信息




---
## 参考
> 1. [深度学习框架PyTorch 常用代码段总结](https://zhuanlan.zhihu.com/p/419063125)   
> 2. [Pytorch Tensor Data types](https://pytorch.org/docs/stable/tensors.html?highlight=tensor#torch.Tensor)   
> 3. 
