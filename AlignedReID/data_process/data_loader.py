#-*-coding:utf-8-*-
from __future__ import print_function, absolute_import
from PIL import Image
import numpy as np
import os.path as osp

import torch
from torch.utils.data import Dataset
from IPython import embed
from data_process import dataset_manager

# 设置图片读取方法
def read_image(image_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    # 标志位表示是否读取到图片
    got_image = False
    if not osp.exists(image_path):
        raise IOError("{} is not exists".format(image_path))
    # 没读到图片就一直读
    while not got_image:
        try:
            # 把读到的图片转化为RGB格式
            img = Image.open(image_path).convert('RGB')
            got_image = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(image_path))
            pass
        return img


# 重写dataset类
class ImageDataset(Dataset):
    """Image Person ReID Dataset"""
    def __init__(self,dataset,transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self,index):
        # 读取dataset的一行信息
        img_path,pid,camid =self.dataset[index]
        # 使用read_image读取图片
        img = read_image(img_path)
        # 判断是否进行数据增广
        if self.transform is not None:
            img = self.transform(img)
        return img, pid, camid



# 验证
if __name__ == "__main__":
    dataset =dataset_manager.init_img_dataset(root='/home/user/桌面/code/data',name="market1501")
    train_loader = ImageDataset(dataset.train)
    for batch_id,(img,pid,camid) in enumerate(train_loader):
        break
    print(batch_id,img,pid,camid)

