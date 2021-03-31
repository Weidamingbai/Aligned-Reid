#-*-coding:utf-8-*-
from __future__ import absolute_import

import os
import sys
import os.path as osp
import time
import datetime
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torch.optim import lr_scheduler


# 导入自己创建的工具
import models
from losses.tripletloss import TripletAlignedReIDloss,DeepSupervision
from data_process import dataset_manager
from data_process.data_loader import ImageDataset
from utils.util import AverageMeter,Logger,save_checkpoint
from utils.optimizers import init_optim
from utils.samplers import RandomIdentitySampler
from utils import re_ranking
from utils.eval_metrics import evaluate

from IPython import embed


# 0.设置一些常见的选项
parser = argparse.ArgumentParser(description='Train AlignedReID with cross entropy loss and triplet hard loss')
# Datasets
# path
parser.add_argument('--root', type=str, default='/home/user/桌面/code/data', help="root path to data directory")
# dataset name
parser.add_argument('-d', '--dataset', type=str, default='market1501',
                    choices=dataset_manager.get_names())
# 多线程 4个
parser.add_argument('-j', '--workers', default=4, type=int,
                    help="number of data loading workers (default: 4)")
# image height
parser.add_argument('--height', type=int, default=256,
                    help="height of an image (default: 256)")
# image weight
parser.add_argument('--width', type=int, default=128,
                    help="width of an image (default: 128)")
# split-id 默认为0
parser.add_argument('--split-id', type=int, default=0, help="split index")

# Optimization options
parser.add_argument('--labelsmooth', action='store_true', help="label smooth")
# 默认使用adam优化
parser.add_argument('--optim', type=str, default='adam', help="optimization algorithm (see optimizers.py)")
# 总共300epoch
parser.add_argument('--max-epoch', default=300, type=int,
                    help="maximum epochs to run")
parser.add_argument('--start-epoch', default=0, type=int,
                    help="manual epoch number (useful on restarts)")
# batch size
parser.add_argument('--train-batch', default=32, type=int,
                    help="train batch size")
parser.add_argument('--test-batch', default=32, type=int, help="test batch size")
# 学习率初始值
parser.add_argument('--lr', '--learning-rate', default=0.0002, type=float,
                    help="initial learning rate")
# 步长
parser.add_argument('--stepsize', default=150, type=int,
                    help="stepsize to decay learning rate (>0 means this is enabled)")
# 学习率衰减系数
parser.add_argument('--gamma', default=0.1, type=float,
                    help="learning rate decay")
parser.add_argument('--weight-decay', default=5e-04, type=float,
                    help="weight decay (default: 5e-04)")

# triplet hard loss
parser.add_argument('--margin', type=float, default=0.3, help="margin for triplet loss")
parser.add_argument('--num-instances', type=int, default=4,
                    help="number of instances per identity")
parser.add_argument('--htri-only', action='store_true', default=False,
                    help="if this is True, only htri loss is used in training")
# Architecture
parser.add_argument('-a', '--arch', type=str, default='resnet50', choices=models.get_names())

# Miscs
parser.add_argument('--print-freq', type=int, default=10, help="print frequency")
parser.add_argument('--seed', type=int, default=1, help="manual seed")
parser.add_argument('--resume', type=str, default='', metavar='PATH')
parser.add_argument('--evaluate', action='store_true', help="evaluation only")
# 每20个epoch执行一次测试
parser.add_argument('--eval-step', type=int, default=20,
                    help="run evaluation for every N epochs (set to -1 to test after training)")
parser.add_argument('--start-eval', type=int, default=0, help="start to evaluate after specific epoch")
parser.add_argument('--save-dir', type=str, default='log')
parser.add_argument('--use_cpu', action='store_true', help="use cpu")
# 默认使用gup-0
parser.add_argument('--gpu-devices', default='0', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument('--reranking',action= 'store_true', help= 'result re_ranking')

parser.add_argument('--test_distance',type = str, default='global', help= 'test distance type')
parser.add_argument('--unaligned',action= 'store_true', help= 'test local feature with unalignment')

args = parser.parse_args()

# 主函数
def main():
    # 判断是否有GPU
    use_gpu = torch.cuda.is_available()
    # 使用cpu则Gpu关
    if args.use_cpu:use_gpu = False
    # 节省内存
    pin_memory =True if use_gpu else False
    # 日志输出
    if not args.evaluate:
        sys.stdout = Logger(osp.join(args.save_dir, 'log_train.txt'))
    else:
        sys.stdout = Logger(osp.join(args.save_dir, 'log_test.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    # 使用GPU的一些设置
    if use_gpu:
        print("Currently using GPU {}".format(args.gpu_devices))
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
        cudnn.benchmark = True
        # 确定随机初始化seed
        torch.cuda.manual_seed_all(args.seed)
    else:
        print("Currently using CPU (GPU is highly recommended)")

    # 1. 数据预处理
    # 1.1 初始化数据集
    print("Initializing dataset {}".format(args.dataset))
    dataset = dataset_manager.init_img_dataset(
        root=args.root, name=args.dataset
    )

    # 1.2 data augmentation
    # 训练集和测试集采用的方式不同
    # 测试集不需要数据增强 仅需要修改图片格式
    # 这里把图片转成的256，128 且为tensor
    transform_train = T.Compose([
        T.Resize((args.height,args.width)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
    ])

    transform_test = T.Compose([
        T.Resize((args.height,args.width)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
    ])
    # 1.3读入图片数据
    trainloader = DataLoader(
        ImageDataset(dataset.train,transform=transform_train),
        sampler=RandomIdentitySampler(dataset.train,num_instances=args.num_instances),
        batch_size=args.train_batch,num_workers=args.workers,
        pin_memory=pin_memory,
        # 丢掉不满足一个batch的数据
        drop_last=True,
    )

    queryloader = DataLoader(
        ImageDataset(dataset.query,transform=transform_test),
        # shuffle =False 不打乱顺序
        batch_size=args.test_batch,shuffle=False,num_workers=args.workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    galleryloader = DataLoader(
        ImageDataset(dataset.gallery, transform=transform_test),
        batch_size=args.test_batch, shuffle=False, num_workers=args.workers,
        pin_memory=pin_memory, drop_last=False,
    )

    # 2.加载模型
    print("Initializing model: {}".format(args.arch))
    model = models.init_model(name=args.arch,num_classes = dataset.num_train_pids,loss={'softmax','metric'},aligned =True,use_gpu=use_gpu)
    print("Model size: {:.5f}M".format(sum(p.numel() for p in model.parameters()) / 1000000.0))

    # 3.加载损失函数
    if args.labelsmooth:
        # overfit
        criterion_class = nn.CrossEntropyLabelSmooth(num_classes=dataset.num_train_pids, use_gpu=use_gpu)
    else:
        # 交叉熵损失(分类)
        criterion_class = nn.CrossEntropyLoss()
    # 三元损失（度量）
    criterion_metric = TripletAlignedReIDloss(margin=args.margin)

    # 4.加载模型优化器
    # 这里对模型的优化器进行了重构 可以根据参数调用不同的优化器
    # args.optim 决定使用的优化器
    # model.parameters()对所有的参数进行更新    model.conv1对模型的第一层进行更新
    # args.lr 初始学习率
    # args.wight_decay模型正则化参数
    optimizer = init_optim(args.optim,model.parameters(),args.lr,args.weight_decay)
    # 根据需求选择不同调整学习率方法
    # 学习率的衰减 避免模型震荡
    if args.stepsize > 0:
        scheduler = lr_scheduler.StepLR(optimizer,step_size=args.stepsize,gamma=args.gamma)

    start_epoch = args.start_epoch
    # 是否需要恢复模型
    if args.resume:
        print("Loading checkpoint from '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch']

    # 使用并行库
    if use_gpu:
        model = nn.DataParallel(model).cuda()
    # 测试
    if args.evaluate:
        print("Evaluate only")
        #    resnet50,query,        test
        test(model, queryloader, galleryloader, use_gpu)
        return 0

    # 5.模型训练
    start_time = time.time()
    train_time = 0
    best_rank1 = -np.inf
    best_epoch = 0
    print("==> Start training")
    # 5.1 从开始的epoch，到结束的epoch开始循环
    for epoch in range(start_epoch, args.max_epoch):
        start_train_time = time.time()
        # 5.2 调用训练函数进行训练
        train(epoch, model, criterion_class, criterion_metric, optimizer, trainloader, use_gpu)
        # 计算了一下训练的时间
        train_time += round(time.time() - start_train_time)
        # 学习率衰减
        if args.stepsize > 0: scheduler.step()
        # 测试
        if (epoch + 1) > args.start_eval and args.eval_step > 0 and (epoch + 1) % args.eval_step == 0 or (
                epoch + 1) == args.max_epoch:
            print("==> Test")
            rank1 = test(model, queryloader, galleryloader, use_gpu)
            is_best = rank1 > best_rank1
            if is_best:
                best_rank1 = rank1
                best_epoch = epoch + 1

            if use_gpu:
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            # 6.保存模型
            save_checkpoint({
                'state_dict': state_dict,
                'rank1': rank1,
                'epoch': epoch,
            }, is_best, osp.join(args.save_dir, 'checkpoint_ep' + str(epoch + 1) + '.pth.tar'))

    print("==> Best Rank-1 {:.1%}, achieved at epoch {}".format(best_rank1, best_epoch))

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    train_time = str(datetime.timedelta(seconds=train_time))
    print("Finished. Total elapsed time (h:m:s): {}. Training time (h:m:s): {}.".format(elapsed, train_time))

def train(epoch, model, criterion_class, criterion_metric, optimizer, trainloader, use_gpu):
    # 确定模型实在训练模式
    model.train()
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    xent_losses = AverageMeter()
    global_losses = AverageMeter()
    local_losses = AverageMeter()

    end = time.time()
    # 使用trainloader迭代器吐数据
    for batch_idx, (imgs, pids, _) in enumerate(trainloader):
        if use_gpu:
            imgs, pids = imgs.cuda(), pids.cuda()
        # measure data loading time
        data_time.update(time.time() - end)
        # [32,751],[32,128,8],[32,2048]
        outputs, global_features, local_features = model(imgs)
        # `htri`: triplet loss with hard positive/negative mining [4]
        if args.htri_only:
            # isinstance() 函数来判断一个对象是否是一个已知的类型，类似 type() tuple 元组 数组
            if isinstance(global_features, tuple):
                global_loss, local_loss = DeepSupervision(criterion_metric, global_features, pids, local_features)
            else:
                global_loss, local_loss = criterion_metric(global_features, pids, local_features)
        else:
            if isinstance(outputs, tuple):
                # `xent`: cross entropy + label smoothing regularizer
                xent_loss = DeepSupervision(criterion_class, outputs, pids)
            else:
                xent_loss = criterion_class(outputs, pids)

            if isinstance(global_features, tuple):
                global_loss, local_loss = DeepSupervision(criterion_metric, global_features, pids, local_features)
            else:
                global_loss, local_loss = criterion_metric(global_features, pids, local_features)
        # 计算损失
        loss = xent_loss + global_loss + local_loss
        # 清空优化器梯度
        optimizer.zero_grad()
        # 反向传播
        loss.backward()
        # 更新模型参数
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        losses.update(loss.item(), pids.size(0))
        xent_losses.update(xent_loss.item(), pids.size(0))
        global_losses.update(global_loss.item(), pids.size(0))
        local_losses.update(local_loss.item(), pids.size(0))

        if (batch_idx+1) % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'CLoss {xent_loss.val:.4f} ({xent_loss.avg:.4f})\t'
                  'GLoss {global_loss.val:.4f} ({global_loss.avg:.4f})\t'
                  'LLoss {local_loss.val:.4f} ({local_loss.avg:.4f})\t'.format(
                   epoch+1, batch_idx+1, len(trainloader), batch_time=batch_time,data_time=data_time,
                   loss=losses,xent_loss=xent_losses, global_loss=global_losses, local_loss = local_losses))

def test(model, queryloader, galleryloader, use_gpu, ranks=[1, 5, 10, 20]):
    batch_time = AverageMeter()
    # 和model.train相对 表示不使用BatchNormalization和Dropout，保证BN和dropout不变化
    model.eval()
    # 反向传播时都不会自动求导。volatile可以实现一定速度的提升，并节省一半的显存，因为其不需要保存梯度
    with torch.no_grad():
        # 1.queryloader处理
        # list列表数据类型，列表是一种可变序列
        qf, q_pids, q_camids, lqf = [], [], [], []
        # 从迭代器中取数据
        for batch_idx, (imgs, pids, camids) in enumerate(queryloader):
            # 转化为cuda模式
            if use_gpu: imgs = imgs.cuda()

            end = time.time()
            global_features, local_features = model(imgs)
            batch_time.update(time.time() - end)
            # 将GPU上的tensor转化为cpu上从而进行一些只能在cpu上进行的运算
            global_features = global_features.data.cpu()
            local_features = local_features.data.cpu()
            # 添加到列表
            qf.append(global_features)
            lqf.append(local_features)
            q_pids.extend(pids)
            q_camids.extend(camids)
        # torch.cat是将两个张量（tensor）拼接在一起 #按维数0（行）拼接
        qf = torch.cat(qf, 0)
        lqf = torch.cat(lqf,0)
        q_pids = np.asarray(q_pids)
        q_camids = np.asarray(q_camids)

        print("Extracted features for query set, obtained {}-by-{} matrix".format(qf.size(0), qf.size(1)))

        # 2.galleryloader处理
        gf, g_pids, g_camids, lgf = [], [], [], []
        end = time.time()
        for batch_idx, (imgs, pids, camids) in enumerate(galleryloader):
            if use_gpu: imgs = imgs.cuda()

            end = time.time()
            features, local_features = model(imgs)
            batch_time.update(time.time() - end)

            features = features.data.cpu()
            local_features = local_features.data.cpu()
            gf.append(features)
            lgf.append(local_features)
            g_pids.extend(pids)
            g_camids.extend(camids)
        gf = torch.cat(gf, 0)
        lgf = torch.cat(lgf,0)
        g_pids = np.asarray(g_pids)
        g_camids = np.asarray(g_camids)


        print("Extracted features for gallery set, obtained {}-by-{} matrix".format(gf.size(0), gf.size(1)))

    print("==> BatchTime(s)/BatchSize(img): {:.3f}/{}".format(batch_time.avg, args.test_batch))
    # feature normlization
    qf = 1. * qf / (torch.norm(qf, 2, dim = -1, keepdim=True).expand_as(qf) + 1e-12)
    gf = 1. * gf / (torch.norm(gf, 2, dim = -1, keepdim=True).expand_as(gf) + 1e-12)
    m, n = qf.size(0), gf.size(0)
    # 求距离a^2+b^2-2*a*b
    distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
              torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    distmat.addmm_(1, -2, qf, gf.t())
    distmat = distmat.numpy()
    print(distmat.shape,distmat.size)
    if not args.test_distance== 'global':
        print("Only using global branch")
        from utils.distance import low_memory_local_dist
        lqf = lqf.permute(0,2,1)
        lgf = lgf.permute(0,2,1)
        local_distmat = low_memory_local_dist(lqf.numpy(),lgf.numpy(),aligned= not args.unaligned)
        if args.test_distance== 'local':
            print("Only using local branch")
            distmat = local_distmat
        if args.test_distance == 'global_local':
            print("Using global and local branches")
            distmat = local_distmat+distmat
    print("Computing CMC and mAP")
    cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids)

    print("Results ----------")
    print("mAP: {:.1%}".format(mAP))
    print("CMC curve")
    for r in ranks:
        print("Rank-{:<3}: {:.1%}".format(r, cmc[r - 1]))
    print("------------------")

    if args.reranking:
        if args.test_distance == 'global':
            print("Only using global branch for reranking")
            distmat = re_ranking(qf,gf,k1=20, k2=6, lambda_value=0.3)
        else:
            local_qq_distmat = low_memory_local_dist(lqf.numpy(), lqf.numpy(),aligned= not args.unaligned)
            local_gg_distmat = low_memory_local_dist(lgf.numpy(), lgf.numpy(),aligned= not args.unaligned)
            local_dist = np.concatenate(
                [np.concatenate([local_qq_distmat, local_distmat], axis=1),
                 np.concatenate([local_distmat.T, local_gg_distmat], axis=1)],
                axis=0)
            if args.test_distance == 'local':
                print("Only using local branch for reranking")
                distmat = re_ranking(qf,gf,k1=20,k2=6,lambda_value=0.3,local_distmat=local_dist,only_local=True)
            elif args.test_distance == 'global_local':
                print("Using global and local branches for reranking")
                distmat = re_ranking(qf,gf,k1=20,k2=6,lambda_value=0.3,local_distmat=local_dist,only_local=False)
        print("Computing CMC and mAP for re_ranking")
        cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids, use_metric_cuhk03=args.use_metric_cuhk03)

        print("Results ----------")
        print("mAP(RK): {:.1%}".format(mAP))
        print("CMC curve(RK)")
        for r in ranks:
            print("Rank-{:<3}: {:.1%}".format(r, cmc[r - 1]))
        print("------------------")
    return cmc[0]

if __name__ == "__main__":
    main()