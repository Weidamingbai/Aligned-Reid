#-*-coding:utf-8-*-
# 此文件用于加载数据集Market1501
"""
主要步骤：
1.拼接文件夹路径
2.获取图片路径信息、行人ID（pid）、摄像头ID（camid）
3.统计行人、图片总数

"""
from __future__ import print_function, absolute_import
import os.path as osp
import glob
import re

from IPython import embed


class Market1501(object):
    """
    Market1501

    Reference:
    Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.

    URL: http://www.liangzheng.org/Project/project_reid.html

    Dataset statistics:
    # identities: 1501 (+1 for background)
    # images: 12936 (train) + 3368 (query) + 15913 (gallery)
    """
    # 数据集Market1501目录
    dataset_dir = "market1501"

    # 通过创建类对象完成对数据集的加载，因此把读取操作都放入 init方法
    # 默认传入参数root 为数据集所在根目录
    # 默认传入参数min_seq_len 为最小序列长度 默认值为0
    # **kwargs可能会有其他参数
    def __init__(self, root='/home/user/桌面/code/data', min_seq_len=0,**kwargs):
        # 1.加载几个文件夹目录 拼接路径
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test')

        # 检查是否成功加载
        self._check_before_run()

        # 调用目录处理方法
        # 2.获取图片路径信息、行人ID（pid）、摄像头ID（camid）
        # train: ('/home/dmb/Desktop/materials/data/market/bounding_box_train/0796_c3s2_089653_01.jpg', 420, 2)
        train,num_train_pids,num_train_imgs = self._process_dir(self.train_dir,relabel=True)
        query,num_query_pids,num_query_imgs = self._process_dir(self.query_dir,relabel=False)
        gallery,num_gallery_pids,num_gallery_imgs = self._process_dir(self.gallery_dir,relabel=False)
        # 3.统计行人、图片总数
        num_total_pids = num_train_pids + num_query_pids
        num_total_imgs = num_train_imgs + num_query_imgs + num_gallery_imgs
        # embed()
        # 打印信息
        print("=> Market1501 loaded")
        print("Dataset statistics:")
        print("  ------------------------------")
        print("  subset   | # ids | # images")
        print("  ------------------------------")
        print("  train    | {:5d} | {:8d}".format(num_train_pids, num_train_imgs))
        print("  query    | {:5d} | {:8d}".format(num_query_pids, num_query_imgs))
        print("  gallery  | {:5d} | {:8d}".format(num_gallery_pids, num_gallery_imgs))
        print("  ------------------------------")
        print("  total    | {:5d} | {:8d}".format(num_total_pids, num_total_imgs))
        print("  ------------------------------")

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids = num_train_pids
        self.num_query_pids = num_query_pids
        self.num_gallery_pids = num_gallery_pids

    def _check_before_run(self):
        # 定义检验加载是否成功方法
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("{} is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("{} is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("{} is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("{} is not available".format(self.gallery_dir))


    def _process_dir(self,dir_path,relabel=False):

        # 此函数返回一个符合glob匹配的pathname的list，返回结果有可能是空
        # 2.1匹配该路径下所有以.jpg结尾的文件，放入list
        img_paths = glob.glob(osp.join(dir_path,'*.jpg'))
        # 正则表达式设置匹配规则 只提取行人id以及摄像头id
        pattern = re.compile(r'([-\d]+)_c(\d)')

        # 2.2实现relabel
        # 原因是由于训练集只有751个行人，但标注是到1501，直接使用1501会使模型产生750个无效神经元
        # set集合存放的行人ID 后面会用的到
        # 使用set集合可以去重
        pid_container = set()
        # 遍历list集合中的图片名
        for img_path in img_paths:
            # 只关心每张图片的pid，其他值设置为缺省值
            # map() 会根据提供的函数对指定序列做映射。
            # 第一个参数 function 以参数序列中的每一个元素调用 function 函数，返回包含每次 function 函数返回值的新列表
            pid,_ = map(int,pattern.search(img_path).groups())

            # 跳过所有pid为-1的项
            if pid == -1:continue
            # 添加pid到列表
            pid_container.add(pid)
        pid2label = {pid:label for label,pid in enumerate(pid_container)}
        # embed()

        dataset = []
        for img_path in img_paths:
            pid,camid = map(int,pattern.search(img_path).groups())
            if pid == -1:continue
            assert 0 <= pid <=1501
            assert 1 <= camid <= 6
            camid -= 1
            # 这里有个判断 只有relabel = True 我才relabel
            if relabel :pid = pid2label[pid]
            dataset.append((img_path,pid,camid))

        num_pids = len(pid_container)
        num_imgs = len(dataset)
        # 返回值为dataset，图片id数量，图片数量
        return dataset,num_pids,num_imgs


"""Create dataset"""

__img_factory = {
    'market1501': Market1501,
    # 'cuhk03': CUHK03,
    # 'dukemtmcreid': DukeMTMCreID,
    # 'msmt17': MSMT17,
}

# __vid_factory = {
#     'mars': Mars,
#     'ilidsvid': iLIDSVID,
#     'prid': PRID,
#     'dukemtmcvidreid': DukeMTMCVidReID,
# }

def get_names():
    return __img_factory.keys()

def init_img_dataset(name, **kwargs):
    if name not in __img_factory.keys():
        raise KeyError("Invalid dataset, got '{}', but expected to be one of {}".format(name, __img_factory.keys()))
    return __img_factory[name](**kwargs)

# 验证
if __name__ == "__main__":
    init_img_dataset(root='/home/user/桌面/code/data',name="market1501")