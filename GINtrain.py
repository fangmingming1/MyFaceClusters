
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os
import os.path as osp
import sys
import time
import argparse
import numpy as np

import torch
import torch.nn as nn
from torch.backends import cudnn
#from torch.utils.data import DataLoader

import model
from feeder.feeder import Feeder
from utils import to_numpy
from utils.logging import Logger 
from utils.meters import AverageMeter
from utils.serialization import save_checkpoint

from sklearn.metrics import precision_score, recall_score

from feeder.feederforGIN import GINFeeder
from feeder.feederforGIN_onechanged import GINFeederTWO
from model.gin import GIN
from torch_geometric.loader import DataLoader

def main(args, device):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.benchmark = True
    # sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))

    trainset = GINFeederTWO(args, args.feat_path,
                      args.knn_graph_path, 
                      args.label_path, 
                      args.seed, 
                      args.k_at_hop,
                      args.active_connection)

    trainloader = DataLoader(
            trainset, batch_size=args.batch_size,
            num_workers=args.workers, shuffle=True, pin_memory=True)

    net = GIN(args)
    net = net.to(device)
    opt = torch.optim.SGD(net.parameters(), args.lr, 
                          momentum=args.momentum, 
                          weight_decay=args.weight_decay) 

    # 原本是有个.cuda()
    criterion = nn.CrossEntropyLoss().to(device)

    save_checkpoint({
        'state_dict':net.state_dict(),
        'epoch': 0,}, False, 
        fpath=osp.join(args.logs_dir, 'epoch_{}.ckpt'.format(0)))
    for epoch in range(args.epochs):
        adjust_lr(opt, epoch)

        train(trainloader, net, criterion, opt, epoch, device)
        save_checkpoint({ 
            'state_dict':net.state_dict(),
            'epoch': epoch+1,}, False, 
            fpath=osp.join(args.logs_dir, 'epoch_{}.ckpt'.format(epoch+1)))
        

def train(loader, net, crit, opt, epoch, device):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accs = AverageMeter()
    precisions = AverageMeter()
    recalls = AverageMeter()

    net.train()
    end = time.time()
    # feat特征矩阵，adj邻接矩阵，h1id第一跳矩阵，cid 中心点id
    for i, databatch in enumerate(loader):

        data_time.update(time.time() - end)

        # >>>s = [1,2,3]
        # >>>list(map(lambda x:x+1,s))
        # >>>[2,3,4]            将参数送入gpu
        # feat, adj, cid, h1id, gtmat = map(lambda x: x.to(device),
        #                         (feat, adj, cid, h1id, gtmat))
        # print('feat{}, adj{}, cid{}, h1id{}, gtmat{}'.format(feat.shape, adj.shape, cid.shape, h1id.shape, gtmat.shape))
        # feattorch.Size([32, 2201, 512]), adjtorch.Size([32, 2201, 2201]), cidtorch.Size([32, 1]), h1idtorch.Size([32, 200]), gtmattorch.Size([32, 200])
        # h1id 是第一跳邻居索引，cid 是枢轴中心点id  gtmat是中心点和邻居是否连边的edge_labels[]
        '''
        本文使用的GCN是由ReLU函数激活的四个图卷积层的堆栈。然后利用softmax激活后的交叉熵损失作为优化的目标函数。
        在实践中，我们只反向传播1跳邻居节点的梯度，因为我们只考虑主节点和它的1跳邻居之间的联系。
        '''
        # pred = net(feat, adj, h1id) # pred.shape = 6400,2
        feat = databatch.x.to(device)
        A = databatch.edge_index.to(device)
        h1id = databatch.one_hop_idcs.to(device)
        batch = (databatch.batch[-1] + 1).to(device)
        pred = net(feat, A, h1id, batch)

        gtmat = databatch.edge_label.to(device)
        # gtmat 是返回的edge的标签矩阵

        labels = make_labels(gtmat).long() # labels.shape = 32*200 =6400,1
        loss = crit(pred, labels)

        p, r, acc = accuracy(pred, labels)
        
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        losses.update(loss.item(), feat.size(0))
        accs.update(acc.item(), feat.size(0))
        precisions.update(p, feat.size(0))
        recalls.update(r, feat.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        #if i % args.print_freq == 0:
        print('Epoch:[{0}][{1}/{2}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
              'Loss {losses.val:.3f} ({losses.avg:.3f})\t'
              'Accuracy {accs.val:.3f} ({accs.avg:.3f})\t'
              'Precison {precisions.val:.3f} ({precisions.avg:.3f})\t'
              'Recall {recalls.val:.3f} ({recalls.avg:.3f})'.format(
                    epoch, i, len(loader), batch_time=batch_time,
                    data_time=data_time, losses=losses, accs=accs,
                    precisions=precisions, recalls=recalls))



def make_labels(gtmat):
    return gtmat.view(-1) # 将tensor矩阵转换为一行

def adjust_lr(opt, epoch):
    scale = 0.1
    print('Current lr {}'.format(args.lr))
    if epoch in [1,2,3,4]:
        args.lr *=0.1
        print('Change lr to {}'.format(args.lr))
        for param_group in opt.param_groups:
            param_group['lr'] = param_group['lr'] * scale

    

def accuracy(pred, label):
    pred = torch.argmax(pred, dim=1).long() # print(pred.shape)  6400   从pred矩阵（6400*2）中按y轴找到一个最大值 并且返回最大值的索引 得到6400*1
    #print('pred is',pred)
    #print('label is',label)
    acc = torch.mean((pred == label).float()) #
    pred = to_numpy(pred)
    label = to_numpy(label)
    p = precision_score(label, pred, zero_division=1)
    r = recall_score(label, pred)
    return p,r,acc 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # misc
    working_dir = osp.dirname(osp.abspath(__file__)) 
    parser.add_argument('--logs-dir', type=str, metavar='PATH', 
                        default=osp.join(working_dir, 'logs'))
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--workers', default=2, type=int)
    parser.add_argument('--print_freq', default=200, type=int)

    # Optimization args
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=4)
    
    # Training args
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--feat_path', type=str, metavar='PATH',
                        default=osp.join(working_dir, './ms1m/transformed/prat0_train_features.npy'))
    parser.add_argument('--knn_graph_path', type=str, metavar='PATH',
                        default=osp.join(working_dir, './ms1m/transformed/prat0_train_knns.npy'))
    parser.add_argument('--label_path', type=str, metavar='PATH',
                        default=osp.join(working_dir, './ms1m/transformed/prat0_train_labels.npy'))
    parser.add_argument('--k_at_hop', type=int, nargs='+', default=[79, 10])
    parser.add_argument('--active_connection', type=int, default=10)
    parser.add_argument(
        "--filters-1",
        type=int,
        default=256,
        help="Filters (neurons) in 1st convolution. Default is 64.",
    )

    parser.add_argument(
        "--filters-2",
        type=int,
        default=256,
        help="Filters (neurons) in 2nd convolution. Default is 32.",
    )

    parser.add_argument(
        "--filters-3",
        type=int,
        default=256,
        help="Filters (neurons) in 3rd convolution. Default is 16.",
    )
    parser.add_argument(
        "--filters-4",
        type=int,
        default=128,
        help="Filters (neurons) in 3rd convolution. Default is 16.",
    )
    parser.add_argument(
        "--filters-5",
        type=int,
        default=128,
        help="Filters (neurons) in 3rd convolution. Default is 16.",
    )
    parser.add_argument(
        "--dropout", type=float, default=0, help="Dropout probability. Default is 0."
    )
    parser.add_argument("--train", default=True)

    args = parser.parse_args()
    print('args',args)
    # 是否用GPU训练
    train_on_gpu = torch.cuda.is_available()

    if not train_on_gpu:
        print('CUDA is not available.  Training on CPU ...')
    else:
        print('CUDA is available!  Training on GPU ...')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    main(args,device)
