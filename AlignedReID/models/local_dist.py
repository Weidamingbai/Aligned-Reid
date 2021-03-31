#-*-coding:utf-8-*-
# 实现最短路径的计算及选取最难样本对

import torch

# 计算欧式距离

def enclidean_dist(x,y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m,n = x.size(0),y.size(0)
    xx = torch.pow(x,2).sum(1,keepdim=True).expand(m,n)
    yy = torch.pow(y,2).sum(1,keepdim=True).expand(m,n)
    dist = xx + yy
    dist.addmm_(1,-2,x,y.t())
    dist = dist.clamp(min=1e-12).sqrt()
    return dist

def batch_enclidean_dist(x,y):
    """
    Args:
      x: pytorch Variable, with shape [Batch size, Local part, Feature channel]
      y: pytorch Variable, with shape [Batch size, Local part, Feature channel]
    Returns:
      dist: pytorch Variable, with shape [Batch size, Local part, Local part]
    """
    # 断言 在不满足条件时直接抛出异常
    assert len(x.size()) == 3
    assert len(y.size()) == 3
    assert x.size(0) == y.size(0)
    assert x.size(-1) == y.size(-1)

    N,m,d = x.size()
    N,n,d = y.size()

    # shape[N,m,n]
    xx = torch.pow(x,2).sum(-1,keepdim=True).expand(N,m,n)
    yy = torch.pow(y,2).sum(-1,keepdim=True).expand(N,n,m).permute(0,2,1)
    dist = xx + yy
    dist.baddbmm_(1,-2,x,y.permute(0,2,1))
    dist = dist.clamp(min=1e-12).sqrt()
    return dist

# 计算最短路径
def shortest_dist(dist_mat):
    """Parallel version.
    Args:
      dist_mat: pytorch Variable, available shape:
        1) [m, n]
        2) [m, n, N], N is batch size
        3) [m, n, *], * can be arbitrary additional dimensions
    Returns:
      dist: three cases corresponding to `dist_mat`:
        1) scalar
        2) pytorch Variable, with shape [N]
        3) pytorch Variable, with shape [*]
    """
    m,n = dist_mat.size()[:2]
    # Just offering some reference for accessing intermediate distance.
    # m×n次循环 创建数组
    dist = [[0 for _ in range(n)] for _ in range(m)]
    for i in range(m):
        for j in range(n):
            if(i==0) and (j==0):
                dist[i][j] = dist_mat[i,j]
            elif(i == 0) and (j > 0):
                dist[i][j] = dist[i][j-1] + dist_mat[i,j]
            elif(i>0) and (j == 0):
                dist[i][j] = dist[i-1][j] + dist_mat[i,j]
            else:
                dist[i][j] = torch.min(dist[i-1][j],dist[i][j-1]) + dist_mat[i,j]
    dist = dist[-1][-1]

    return dist

def batch_local_dist(x,y):
    """
    Args:
      x: pytorch Variable, with shape [N, m, d]
      y: pytorch Variable, with shape [N, n, d]
    Returns:
      dist: pytorch Variable, with shape [N]
    """
    assert len(x.size()) == 3
    assert len(y.size()) == 3
    assert x.size(0) == y.size(0)
    assert x.size(-1) == y.size(-1)

    # shape[N,m,n]
    dist_mat = batch_enclidean_dist(x,y)
    dist_mat = (torch.exp(dist_mat) - 1.)/(torch.exp(dist_mat) + 1.)

    # shape[N]
    dist = shortest_dist(dist_mat.permute(1,2,0))

    return dist






if __name__ == "__main__":
    x = torch.randn(32,2048)
    y = torch.randn(32,2048)
    dist_mat = enclidean_dist(x,y)
