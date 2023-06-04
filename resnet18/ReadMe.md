# ResNet18 训练与数据增强效果的展示
## 该项目的实现主要是来自[源项目](https://github.com/zgcr/SimpleAICV_pytorch_training_examples)，根据课程要求实现以下需求：
* 在CIFAR100数据集上训练并测试分类模型 ResNet-18;
* 对比cutmix, cutout, mixup三种方法以及baseline方法在CIFAR-100图像分类任务中的性能表现;
* 使用 [wandb](https://github.com/wandb/wandb) 可视化训练和测试的loss曲线

## 环境配置
```
torch==1.10.0
torchvision==0.11.1
torchaudio==0.10.0
onnx==1.11.0
onnx-simplifier==0.3.6
numpy
Cython
pycocotools
opencv-python
tqdm
thop==0.0.31.post2005241907
yapf
apex
wandb
```

## 数据集配置
下载 CIFAR100 数据集，确保格式如下
```
CIFAR100
|
|-----train unzip from cifar-100-python.tar.gz
|-----test  unzip from cifar-100-python.tar.gz
|-----meta  unzip from cifar-100-python.tar.gz
```
更改 `tools/path.py` 文件中的path目录

## 测试模型
* 下载对应的checkpoint
* 在 `classification_training/cifar100/resnet18cifar/train_config` 中更改模型地址 `trained_model_path`. 
* `cd classification_training/cifar100/resnet18cifar/ && ./test.sh`

## 训练模型
* 在 `classification_training/cifar100/resnet18cifar/train_config` 中更改数据增强的方式 `AUG = 'none', 'mixup', 'cutout', 'cutmix'`. 
* 删掉 checkpoint 中对应数据增强的 checkpoint 子目录
* `cd classification_training/cifar100/resnet18cifar/ && ./train.sh`

