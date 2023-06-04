# DATA620004 第二次课程作业
## 内容与代码结构
本次作业共有两个大题，三组实验：
1. 使用CNN网络模型(自己设计或使用现有的CNN架构，如AlexNet，ResNet-18)作为baseline在CIFAR-100上训练并测试；对比cutmix, cutout, mixup三种方法以及baseline方法在CIFAR-100图像分类任务中的性能表现；对三张训练样本分别经过cutmix, cutout, mixup后进行可视化，一共show 9张图像。
2. 在VOC数据集上训练并测试目标检测模型 Faster R-CNN 和 FCOS；在四张测试图像上可视化Faster R-CNN第一阶段的proposal box；
两个训练好后的模型分别可视化三张不在VOC数据集内，但是包含有VOC中类别物体的图像的检测结果（类别标签，得分，boundingbox），并进行对比，一共show六张图像；

对应的代码结构：
* 第一道题目的全部训练脚本与运行所需的环境在 `resnet18` 目录下
* 第二道题 Faster-R-CNN 的训练脚本与依赖环境在 `faster_rcnn` 目录下
* 第二道题 FCOS 的训练脚本与依赖环境在 `fcos` 目录下.

## 模型参数
所有的模型训练结果都可以在 [google drive](https://drive.google.com/file/d/14SHHXpEK1VNKG7V3lgAhoJFYXBo-YxX6/view?usp=sharing) 下载.