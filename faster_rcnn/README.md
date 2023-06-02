# Faster RCNN训练与可视化
## 该项目的实现主要是来自[源项目](https://github.com/WZMIAOMIAO/deep-learning-for-image-processing/tree/master/pytorch_object_detection/faster_rcnn)，根据课程要求实现以下需求：
* 在VOC数据集上训练并测试目标检测模型 Faster R-CNN，用tensorboard储存train loss, test loss, mAP曲线；
* 在四张测试图像上可视化Faster R-CNN第一阶段的proposal box；
* 可视化三张不在VOC数据集内，但是包含有VOC中类别物体的图像的检测结果（类别标签，得分，boundingbox）


## 环境配置：
* Python3.7
* Pytorch1.7.1(注意：必须是1.6.0或以上，因为使用官方提供的混合精度训练1.6.0后才支持)
* pycocotools(Linux:`pip install pycocotools`)
* Ubuntu
* 使用GPU训练
* 详细环境配置见`requirements.txt`

## 数据准备
* 数据集：Pascal VOC2012 train/val数据集下载地址：http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
* pretrain weights: 
Resnet50 weights(下载后重命名为`resnet50.pth`，然后放到`bakcbone`文件夹下): https://download.pytorch.org/models/resnet50-0676ba61.pth;
ResNet50+FPN COCO pretrain weights(下载后重命名为`fasterrcnn_resnet50_fpn_coco.pth`，然后放到`bakcbone`文件夹下): https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth

## 训练方法
训练脚本为：
```
python train_res50_fpn.py --data-path voc/root/path --batch_size 4 --load_coco_pretrain True --tensorboard_log './logs_voc_frompretrain' 
```
注意要将`--data-path`(VOC_root)设置为自己存放`VOCdevkit`文件夹所在的**根目录**；将`--tensorboard_log`设置为tensorboard可视化文件储存路径

`--load_coco_pretrain`代表是否用COCO pretrain weights开始训练

## 可视化
可视化proposal box使用脚本为：
```
python pred_proposal.py --img_file img/folder/path --weights save_weights/resNetFpn-model-final.pth
```
注意需下载通过上一步训练得到的[checkpoint](https://drive.google.com/file/d/14SHHXpEK1VNKG7V3lgAhoJFYXBo-YxX6/view?usp=share_link)，并设置在`--weights`参数中。

可视化检测结果使用脚本为：
```
python predict.py --img_file img/folder/path --weights save_weights/resNetFpn-model-final.pth
```
可视化结果图都储存在`img_file`原路径中。