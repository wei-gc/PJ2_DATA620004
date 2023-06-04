# FCOS for PASCAL VOC
## 该项目的实现主要是来自[源项目](https://github.com/zhenghao977/FCOS-PyTorch-37.2AP)，根据课程要求实现以下需求：
* 在 VOC-2012 数据集上训练并测试分类模型 ResNet-FCOS, 以 ResNet-50 作为backbone;
* 使用 [wandb](https://github.com/wandb/wandb) 可视化训练和测试的loss曲线
## Requirements  
* opencv-python  
* pytorch >= 1.0  
* torchvision >= 0.4. 
* matplotlib
* cython
* numpy == 1.17
* Pillow
* tqdm
* pycocotools
* wandb

## prepare dataset
更改 train_voc.py 文件中的 `DATASETPATH` 变量。

## backbone parameter
运行代码，将自动下载训练好的 resnet-50 参数到 `model/backbone/` 下面.

## train for PASCAL VOC
运行 train_voc.py, 共 30 epoch. 可以指定使用的显卡个数。

## Detect Image   
You can run the detect.py to detect images.
