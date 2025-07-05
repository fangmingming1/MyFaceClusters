
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
import torch.nn.functional as F
from torch.backends import cudnn
#from torch.utils.data import DataLoader

import model
from feeder.feeder import Feeder
from utils import to_numpy
from utils.meters import AverageMeter
from utils.serialization import load_checkpoint
from utils.utils import bcubed,bcubed1,fowlkes_mallows_score
from utils.graph import bfs_clustering, graph_propagation_soft, graph_propagation_naive

from sklearn.metrics import normalized_mutual_info_score, precision_score, recall_score

from torch_geometric.loader import DataLoader
from feeder.feederforGIN_onechanged import GINFeederTWO
from model.gin import GIN

def single_remove(Y, pred):
    single_idcs = np.zeros_like(pred)  # 创建一个和预测标签一样长的ndarray数组
    pred_unique = np.unique(pred)  # 生成预测标签的唯一值数组
    for u in pred_unique:
        idcs = pred == u  # 判断一下预测标签列表里，重复的值,得到里面值为True或者False的数组
        if np.sum(idcs) == 1:  # 统计一下True值得数目  如果等于1，则表明当前节点被分到了单独的聚类图
            single_idcs[np.where(idcs)[0][
                0]] = 1  # np.where(condition)参数表示条件，当条件成立时，where返回的是每个符合condition条件元素的坐标,返回的是以元组的形式。 所以[0][0]是元组中的第一个元素，的第一个值
    remain_idcs = [i for i in range(len(pred)) if
                   not single_idcs[i]]  # 遍历pred的索引，如果single_idcs[i]的值不是1的话，将i加入到remain_idcs中
    remain_idcs = np.asarray(remain_idcs)
    return Y[remain_idcs], pred[remain_idcs]


def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.benchmark = True

    valset = GINFeederTWO(args,
                    args.val_feat_path,
                    args.val_knn_graph_path,
                    args.val_label_path,
                    args.seed,
                    args.k_at_hop,
                    args.active_connection,
                    train=False)

    valloader = DataLoader(
        valset, batch_size=args.batch_size,
        num_workers=args.workers, shuffle=False, pin_memory=True)

    ckpt = load_checkpoint(args.checkpoint)
    net = GIN(args)
    net.load_state_dict(ckpt['state_dict'])
    net = net.cuda()


    '''
    knn_graph_dict = list()
    knn_graph_dict.append(dict())
    knn_graph_dict.append(dict())
    knn_graph_dict[0][1] = []
    knn_graph_dict[0][0] = []
    print(knn_graph_dict)

    knn_graph_dict[-1][1] = []
    knn_graph_dict[-1][0] = []
    -----------
    [{1: [], 0: []}, {}]

    [{}, {1: [], 0: []}]
    所以[-1]是list集合里面的dict集合的下标-1代表从最后一个开始，[n]是要给dict传入的key值，[]是要给dict传入的value值
    '''
    criterion = nn.CrossEntropyLoss().cuda()
    edges, scores = validate(valloader, net, criterion, args)

    np.save('edges', edges)  # 363420*2
    np.save('scores', scores)  # 363420*1
    # edges = np.load('./测试中间结果/msmt17/edges.npy')
    # scores = np.load('./测试中间结果/msmt17/scores.npy')

    # 将得到的边集和对这些边集的链接可能性的预测输入
    clusters = bfs_clustering(edges, scores, max_sz=500, step=0.95, pool='max', beg_th=0.9)


    final_pred = clusters2labels(clusters, len(valset))  # final_pred是所有顶点(18171)的所归属子图的标签
    labels = valset.labels  # 得到所有顶点对应的真实的（聚类子图标签？）
    np.save('labels_pred.npy', final_pred)

    print('------------------------------------')
    print('Number of nodes: ', len(labels))
    print('Precision      Recall     Bcubed F-Score     avg_pre        avg_rec   Parewised F-Score     NMI')
    p, r, f = bcubed1(labels, final_pred)  # 将预测的聚类子图标签和真实的子图标签传入，得到precision, recall, and F-sore
    nmi = normalized_mutual_info_score(final_pred, labels)
    avg_pre, avg_rec, fscore = fowlkes_mallows_score(final_pred, labels)
    print(('{:.4f}         ' * 7).format(p, r, f, avg_pre, avg_rec, fscore, nmi))

    # labels, final_pred = single_remove(labels, final_pred)
    # print('------------------------------------')
    # print('After removing singleton culsters, number of nodes: ', len(labels))
    # print('Precision      Recall     Bcubed F-Score     avg_pre        avg_rec   Parewised F-Score     NMI')
    # p, r, f = bcubed1(labels, final_pred)
    # nmi = normalized_mutual_info_score(final_pred, labels)
    # avg_pre, avg_rec, fscore = fowlkes_mallows_score(final_pred, labels)
    # print(('{:.4f}         ' * 7).format(p, r, f, avg_pre, avg_rec, fscore, nmi))


def clusters2labels(clusters, n_nodes):
    labels = (-1) * np.ones((n_nodes,))  # 创建一个维度为所有顶点数的全-1数组
    for ci, c in enumerate(clusters):  # c为clusters中的set集合（set里面存放的是 聚类子图顶点的 Data对象） ci为list的索引值从0开始，所以ci也就是聚类子图的标签号
        for xid in c:  # xid 是顶点Data对象
            labels[xid.name] = ci  # 通过顶点Data对象的name属性值，也就是顶点的标签号，设置其所在的聚类子图的标签号。得到顶点归属的聚类子图标签
    assert np.sum(labels < 0) < 1
    return labels  # 返回的labels是每个顶点索引对应的聚类子图的编号标签


def make_labels(gtmat):
    return gtmat.view(-1)


# 验证数据集
def validate(loader, net, crit, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accs = AverageMeter()
    precisions = AverageMeter()
    recalls = AverageMeter()

    net.eval()
    end = time.time()
    edges = list()
    scores = list()
    for i, graph_data in enumerate(loader):
        data_time.update(time.time() - end)

        feat = graph_data.x.cuda()
        adj = graph_data.edge_index.cuda()
        cid = graph_data.center_idx.cuda()
        h1id = graph_data.one_hop_idcs.cuda()
        gtmat = graph_data.y.cuda()

        batch = (graph_data.batch[-1] + 1).cuda()
        pred = net(feat, adj, h1id, batch)

        labels = make_labels(gtmat).long()
        loss = crit(pred, labels)
        pred = F.softmax(pred, dim=1)
        p, r, acc = accuracy(pred, labels)

        losses.update(loss.item(), feat.size(0))
        accs.update(acc.item(), feat.size(0))
        precisions.update(p, feat.size(0))
        recalls.update(r, feat.size(0))

        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            print('[{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {losses.val:.3f} ({losses.avg:.3f})\n'
                  'Accuracy {accs.val:.3f} ({accs.avg:.3f})\t'
                  'Precison {precisions.val:.3f} ({precisions.avg:.3f})\t'
                  'Recall {recalls.val:.3f} ({recalls.avg:.3f})'.format(
                i, len(loader), batch_time=batch_time,
                data_time=data_time, losses=losses, accs=accs,
                precisions=precisions, recalls=recalls))
        node_list = graph_data.unique_nodes_list.view(batch, -1)
        node_list = node_list.long().squeeze().numpy()

        h1id = h1id.view(batch, -1)
        cid = cid.view(batch, -1)
        # 一个批次内的cid中心点id，和 node_list，中心点和第一二条邻居节点 list集合
        for b in range(batch):
            #  cidb是一个批次内第几个knn图中心点的索引
            cidb = cid[b].int().item()
            # nl是一个批次内第几个knn图  中心点和第一二条邻居  节点集合
            nl = node_list[b]
            # 遍历32个批次内对应的第一跳邻居的索引
            # j是20个邻居所在的数组索引，n是对应的值
            for j, n in enumerate(h1id[b]):
                n = n.item()

                # edges是一个边的list集合  nl[cidb]是中心节点  nl[n]是中心节点的第一跳邻居的节点索引
                edges.append([nl[cidb], nl[n]])

                # pred是一个640*2的tensor  是对一个批次的32 * 20个邻居点与中心点是否链接的可能性的判断，640行每一行行和为1，对应着 [0]否/[1]是
                # b当前批次*第一跳邻居个数20 +j(j是20个邻居所在的数组索引)
                # 所以这个是返回了预测值当中，和中心点相连的顶点的边的预测可能性
                scores.append(pred[b * args.k_at_hop[0] + j, 1].item())
    # 这个edges和scores存储了与 中心点相连的邻居点 和 这些边所对应的 链接可能性的概率值
    edges = np.asarray(edges)
    scores = np.asarray(scores)
    return edges, scores


def accuracy(pred, label):
    pred = torch.argmax(pred, dim=1).long()
    acc = torch.mean((pred == label).float())
    pred = to_numpy(pred)
    label = to_numpy(label)
    p = precision_score(label, pred)
    r = recall_score(label, pred)
    return p, r, acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # misc
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--workers', default=1, type=int)
    parser.add_argument('--print_freq', default=40, type=int)

    # Optimization args
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=20)

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--k_at_hop', type=int, nargs='+', default=[20, 5])
    parser.add_argument('--active_connection', type=int, default=5)

    # Validation args
    parser.add_argument('--val_feat_path', type=str, metavar='PATH',
                        default=osp.join(working_dir, './data/1024.fea.npy'))
    parser.add_argument('--val_knn_graph_path', type=str, metavar='PATH',
                        default=osp.join(working_dir, './data/knn.graph.1024.bf.npy'))
    parser.add_argument('--val_label_path', type=str, metavar='PATH',
                        default=osp.join(working_dir, './data/1024.labels.npy'))
    # parser.add_argument('--val_feat_path', type=str, metavar='PATH',
    #                     default=osp.join(working_dir, './data/msmt17/test/test_feat.npy'))
    # parser.add_argument('--val_knn_graph_path', type=str, metavar='PATH',
    #                     default=osp.join(working_dir, './data/msmt17/transformed/msmt17_knns_nbrs_test.npy'))
    # parser.add_argument('--val_label_path', type=str, metavar='PATH',
    #                     default=osp.join(working_dir, './data/msmt17/test/test_label.npy'))
    # parser.add_argument('--val_feat_path', type=str, metavar='PATH',
    #                     default=osp.join(working_dir, './data/deepfashion/deepfashion_test_features.npy'))
    # parser.add_argument('--val_knn_graph_path', type=str, metavar='PATH',
    #                     default=osp.join(working_dir, './data/deepfashion/deepfashion_test/deepfashion_test_knn.npy'))
    # parser.add_argument('--val_label_path', type=str, metavar='PATH',
    #                     default=osp.join(working_dir, './data/deepfashion/deepfashion_test_labels.npy'))
    # parser.add_argument('--val_feat_path', type=str, metavar='PATH',
    #                     default=osp.join(working_dir, './ms1m/transformed/prat5_test_features.npy'))
    # parser.add_argument('--val_knn_graph_path', type=str, metavar='PATH',
    #                     default=osp.join(working_dir, './ms1m/transformed/prat5_test_knns.npy'))
    # parser.add_argument('--val_label_path', type=str, metavar='PATH',
    #                     default=osp.join(working_dir, './ms1m/transformed/prat5_test_labels.npy'))
    parser.add_argument('--train', type=bool, default=False)
    parser.add_argument(
        "--filters-1",
        type=int,
        default=512,
        help="Filters (neurons) in 1st convolution. Default is 64.",
    )

    parser.add_argument(
        "--filters-2",
        type=int,
        default=512,
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
        default=256,
        help="Filters (neurons) in 3rd convolution. Default is 16.",
    )
    # parser.add_argument(
    #     "--filters-5",
    #     type=int,
    #     default=128,
    #     help="Filters (neurons) in 3rd convolution. Default is 16.",
    # )
    parser.add_argument(
        "--dropout", type=float, default=0, help="Dropout probability. Default is 0."
    )
    # Test argsA
    parser.add_argument('--checkpoint', type=str, metavar='PATH', default='./logs/最重要-CASIA-100-10-4GIN-dropout0.3/epoch_4.ckpt')
    args = parser.parse_args()
    main(args)
