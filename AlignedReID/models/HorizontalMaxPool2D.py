#-*-coding:utf-8-*-
# 水平pooling算是论文的创新点 因此需要我们自己定义

import torch.nn as nn

import torch

class HorizontalMaxPool2d(nn.Module):

    def __init__(self):
        super(HorizontalMaxPool2d,self).__init__()

    # 输入是[N,C,H,W]的feature map
    def forward(self,f):
        input_size = f.size()
        return nn.functional.max_pool2d(input=f, kernel_size=(1,input_size[3]))


# if __name__ == "__main__":
#     x = torch.Tensor(32,2048,8,4)
#     hp = HorizontalMaxPool2d()
#     y = hp(x)
#     print(y.shape)