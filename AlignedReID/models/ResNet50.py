#-*-coding:utf-8-*-
from __future__ import absolute_import

import torch
from torch import nn
from torch.nn import functional as F
import torchvision
from models.HorizontalMaxPool2D import HorizontalMaxPool2d
from IPython import embed

class ResNet50(nn.Module):
    # loss参数设置：损失默认为分类损失和度量损失结合的方式
    # aligned=False 默认为False 表示是否使用local分支
    #  num_classes 分类数
    def __init__(self, num_classes, loss={'softmax','metric'}, aligned=False, **kwargs):
        # 用父类的初始化方法来初始化继承的属性 继承自nn.module
        super(ResNet50, self).__init__()
        self.loss = loss
        # pretrained =True 表示使用预训练模型
        resnet50 = torchvision.models.resnet50(pretrained=True)
        # 去掉平均池化层和fc层 保留前面那些层
        self.base = nn.Sequential(*list(resnet50.children())[:-2])
        # 设置自己的分类器
        self.classifier = nn.Linear(2048, num_classes)

        self.feat_dim = 2048 # feature dimension
        self.aligned = aligned
        # 如何aligned被激活，则加入local分支
        # 不被激活就是简单的度量学习
        if self.aligned:
            # 水平池化
            self.horizon_pool = HorizontalMaxPool2d()
            # 根据统计的mean 和var来对数据进行标准化 2048通道数
            self.bn = nn.BatchNorm2d(2048)
            # 下采样 inplace = True改变输入数据
            self.relu = nn.ReLU(inplace=True)

            # 2维卷积 pytorch无需手动定义网络层权重和偏置
            # in_channels 输入通道数2048
            # out_channels 输出通道数128 实现了特征通道的减少
            # kernel_size 卷积核为1
            # stride 步长为1
            # padding 图像填充
            self.conv1 = nn.Conv2d(2048,128,kernel_size=1,stride=1,padding=0,bias=True)

    # 设置前向传播 （反向传播不用自己定义）x=[32,3,256,128]
    def forward(self, x):
        # 1.ResNet50  计算global_feature
        x = self.base(x) # x.shape => [32,2048,8,4]
        # pool层
        # [32,2048,8,4] =>[32,2048,1,1]
        # 输入为（x ，（8，4））
        # 选择x.size（）的后两个参数
        global_feat = F.avg_pool2d(x, x.size()[2:])
        # 对X进行展平[32,2048,1,1]=>[32,2048]
        # 这样才可以通过分类层
        global_feat = global_feat.view(global_feat.size(0), -1)
        # 对feature map进行归一化处理
        # f = 1.*f / (torch.norm(f, 2, dim=-1, keepdim=True).expand_as(f)+1e-12)

        # 2.计算local_feature
        if not self.training:
            local_feat = self.horizon_pool(x)

        if self.aligned and self.training:
            local_feat = self.bn(x)
            local_feat = self.relu(local_feat)
            local_feat = self.horizon_pool(local_feat)
            local_feat = self.conv1(local_feat)

        if self.aligned or not self.training:
            local_feat = local_feat.view(local_feat.size()[0:3])
            local_feat = local_feat/torch.pow(local_feat,2).sum(dim=1,keepdim=True).clamp(min=1e-12).sqrt()

        # 3.返回特征 计算损失
        # 不是训练集的话 我们直接global_feature就结束了
        if not self.training:
            return global_feat,local_feat
        # 否则经过分类器
        y = self.classifier(global_feat)

        if self.loss == {'softmax'}:
            return y
        elif self.loss == {'softmax', 'metric'}:
            if self.aligned:return y,global_feat,local_feat
            return global_feat
        elif self.loss == {'metric'}:
            return global_feat,local_feat
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))


#
if __name__ == "__main__":
    model = ResNet50(num_classes=751,loss={'softmax','metric'},aligned=True)
    imgs = torch.Tensor(32,3,256,128)
    y,global_feat,local_feat = model(imgs)
    print(y.shape)
    print(local_feat.shape)
    print(global_feat.shape)