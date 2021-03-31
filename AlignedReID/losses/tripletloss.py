#-*-coding:utf-8-*-

import torch
from  torch import nn
from models.local_dist import batch_local_dist

from IPython import embed

# 寻找最难样本对 dist_mat距离矩阵 labels为标签
# return_inds为是否返回正负样本对id 默认为False
# 但是AlignReID两分支需要使用同一对正负样本对 所以为True
def hard_example_mining(dist_mat,labels,return_inds=False):
    """For each anchor, find the hardest positive and negative sample.
    Args:
      dist_mat: pytorch Variable, pair wise distance between samples, shape [N, N]
      labels: pytorch LongTensor, with shape [N]
      return_inds: whether to return the indices. Save time if `False`(?)
    Returns:
      dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
      dist_an: pytorch Variable, distance(anchor, negative); shape [N]
      p_inds: pytorch LongTensor, with shape [N];
        indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
      n_inds: pytorch LongTensor, with shape [N];
        indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
    NOTE: Only consider the case in which all labels have same num of samples,
      thus we can cope with all anchors in parallel.
    """
    # 断言
    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)

    # 距离矩阵的尺寸是 (batch_size, batch_size)  [N,N]
    N = dist_mat.size(0)

    # shape[N,N] 选出所有正负样本对
    is_pos = labels.expand(N,N).eq(labels.expand(N,N).t()) # 两两组合， 取label相同的a-p
    is_neg = labels.expand(N,N).ne(labels.expand(N,N).t()) # 两两组合， 取label不同的a-n

    # `dist_ap` means distance(anchor, positive)
    # both `dist_ap` and `relative_p_inds` with shape [N, 1]
    dist_ap,relative_p_inds = torch.max(dist_mat[is_pos].contiguous().view(N,-1),1,keepdim=True)
    # `dist_an` means distance(anchor, negative)
    # both `dist_an` and `relative_n_inds` with shape [N, 1]
    dist_an, relative_n_inds = torch.min(dist_mat[is_neg].contiguous().view(N, -1), 1, keepdim=True)

    # shape [N]
    # 去掉维数为1的的维度，squeeze(a)将a中所有为1的维度删掉，不为1的维度没有影响。a.squeeze(N) 就是去掉a中指定的维数为一的维度。
    dist_ap = dist_ap.squeeze(1)
    dist_an = dist_an.squeeze(1)

    if return_inds:
        # shape [N, N]
        ind = (labels.new().resize_as_(labels)
               .copy_(torch.arange(0, N).long())
               .unsqueeze(0).expand(N, N))
        # shape [N, 1]
        # 收集输入的特定维度指定位置的数值(input,dim,index)
        p_inds = torch.gather(
            ind[is_pos].contiguous().view(N, -1), 1, relative_p_inds.data)
        n_inds = torch.gather(
            ind[is_neg].contiguous().view(N, -1), 1, relative_n_inds.data)
        # shape [N]
        p_inds = p_inds.squeeze(1)
        n_inds = n_inds.squeeze(1)
        return dist_ap, dist_an, p_inds, n_inds

    return dist_ap, dist_an


class TripletAlignedReIDloss(nn.Module):
    """Triplet loss with hard positive/negative mining.

    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.

    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.

    Args:
        margin (float): margin for triplet.
    """
    def __init__(self,margin=0.3,mutual_flag = False):
        super(TripletAlignedReIDloss,self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        self.ranking_loss_local = nn.MarginRankingLoss(margin=margin)
        self.mutual = mutual_flag

    # input[N,()]
    def forward(self,inputs,targets,local_features):
        """
        Args:
            inputs: feature matrix with shape (batch_size, feat_dim)
            targets: ground truth labels with shape (num_classes)
        """
        n = inputs.size(0)
        # 1.
        # power是逐元素点乘，sum起来就是取模的平方, keepdim保持维度不变，不要求和成一个数
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()  # a^2 + b^2
        # a^2 + b^2 -2*ab
        dist.addmm_(1,-2,inputs,inputs.t()) # 做的是如(a1, a2, a, b) -> a1*dist + a2*a*b
        # for numerical stability
        # 将输入input张量每个元素的夹紧到区间 [min,max][min,max]，并返回结果到一个新张量，防止他等于0，再做一个开方
        dist = dist.clamp(min=1e-12).sqrt()
        dist_ap,dist_an,p_inds,n_inds = hard_example_mining(dist,targets,return_inds=True)
        p_inds,n_inds = p_inds.long(),n_inds.long()
        local_features = local_features.permute(0,2,1)
        p_local_feature = local_features[p_inds]
        n_local_feature = local_features[n_inds]
        local_dist_ap = batch_local_dist(local_features,p_local_feature)
        local_dist_an = batch_local_dist(local_features,n_local_feature)

        y = torch.ones_like(dist_an)
        global_loss = self.ranking_loss(dist_an,dist_ap,y)
        local_loss = self.ranking_loss_local(local_dist_an,local_dist_ap,y)
        if self.mutual:
            return global_loss+local_loss,dist
        return global_loss,local_loss

def DeepSupervision(criterion, xs, y):
    """
    Args:
        criterion: loss function
        xs: tuple of inputs
        y: ground truth
    """
    loss = 0.
    for x in xs:
        loss += criterion(x, y)
    return loss

if __name__ == "__main__":
    # 32
    target = [1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,6,6,6,6,7,7,7,7,8,8,8,8]
    target = torch.Tensor(target)
    features =torch.rand(32,2048)
    local_features = torch.randn(32,128,8)
    triloss = TripletAlignedReIDloss()
    global_loss,local_loss = triloss(features,target,local_features)
