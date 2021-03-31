from __future__ import print_function, absolute_import
import numpy as np
import copy
from collections import defaultdict
import sys

# distmat.shape [3368,15913]
def eval_market1501(distmat, q_pids, g_pids, q_camids, g_camids, max_rank):
    """Evaluation with market1501 metric
    Key: for each query identity, its gallery images from the same camera view are discarded.
    """
    # 获得query gallery图片（特征）的数目
    num_q, num_g = distmat.shape
    # 判断 如果gallery的数目小于rank 则吧gallery的数目给rank
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    # 将dismat中的元素从小到大排列，提取其对应的index(索引)，然后输出到indexs
    indices = np.argsort(distmat, axis=1)
    # 进行匹配，如果g_pids[indices]等于q_pids[:, np.newaxis]身份ID，则被置1。
    # matches[3368，15913]，排列之后的结果类似如下：
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    # compute cmc curve for each query
    all_cmc = []
    all_AP = [] # 记录query每张图像的AP
    num_valid_q = 0. # number of valid query 记录有效的query数量
    # 对每一个query中的图片进行处理
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        # 取出当前第q_idx张图片，在gallery中查询过后的排序结果
        # [3368,15913]-->[15913,]
        order = indices[q_idx]
        # 删除与查询具有相同pid和camid的gallery样本,也就是删除query和gallery中相同图片的结果
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        # 二进制向量，值为1的位置是正确的匹配
        orig_cmc = matches[q_idx][keep] # binary vector, positions with value 1 are correct matches
        if not np.any(orig_cmc):
            # 当查询标识未出现在图库中时，此条件为真
            # this condition is true when query identity does not appear in gallery
            continue
        # 计算一行中的累加值，如一个数组为[0,0,1,1,0,2,0]
        # 通过cumsum得到[0,0,1,2,2,4,4]
        cmc = orig_cmc.cumsum()
        # cmc > 1的位置，表示都预测正确了
        cmc[cmc > 1] = 1

        # 根据max_rank，添cmc到all_cmc之中
        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision 平均精度
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i+1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)
    # # 所有查询身份没有出现在图库（gallery）中则报错
    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"
    # 计算平均cmc精度
    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP

# 测试代码
def evaluate(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):

    return eval_market1501(distmat, q_pids, g_pids, q_camids, g_camids, max_rank)

