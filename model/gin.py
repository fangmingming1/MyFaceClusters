import numpy
import torch.nn
import torch
from torch_geometric.nn import GINConv, global_add_pool
import torch.nn.functional as F

class GIN(torch.nn.Module):

    def __init__(self, args):
        super(GIN, self).__init__()
        self.args = args
        self.training = args.train
        nn1 = torch.nn.Sequential(
            torch.nn.Linear(512, self.args.filters_1),
            torch.nn.ReLU(),
            torch.nn.Linear(self.args.filters_1, self.args.filters_1),
            torch.nn.BatchNorm1d(self.args.filters_1),
        )

        nn2 = torch.nn.Sequential(
            torch.nn.Linear(self.args.filters_1, self.args.filters_2),
            torch.nn.ReLU(),
            torch.nn.Linear(self.args.filters_2, self.args.filters_2),
            torch.nn.BatchNorm1d(self.args.filters_2),
        )

        nn3 = torch.nn.Sequential(
            torch.nn.Linear(self.args.filters_2, self.args.filters_3),
            torch.nn.ReLU(),
            torch.nn.Linear(self.args.filters_3, self.args.filters_3),
            torch.nn.BatchNorm1d(self.args.filters_3),
        )

        nn4 = torch.nn.Sequential(
            torch.nn.Linear(self.args.filters_3, self.args.filters_4),
            torch.nn.ReLU(),
            torch.nn.Linear(self.args.filters_4, self.args.filters_4),
            torch.nn.BatchNorm1d(self.args.filters_4),
        )
        # nn5 = torch.nn.Sequential(
        #     torch.nn.Linear(self.args.filters_4, self.args.filters_5),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(self.args.filters_5, self.args.filters_5),
        #     torch.nn.BatchNorm1d(self.args.filters_5),
        # )
        self.convolution_1 = GINConv(nn1, train_eps=True)
        self.convolution_2 = GINConv(nn2, train_eps=True)
        self.convolution_3 = GINConv(nn3, train_eps=True)
        self.convolution_4 = GINConv(nn4, train_eps=True)
        # self.convolution_5 = GINConv(nn5, train_eps=True)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(256, 256),
            torch.nn.PReLU(256),
            torch.nn.Linear(256, 2))


    def ginconvpass(self,edge,features):
        features = self.convolution_1(features, edge)
        features = F.relu(features)
        features = F.dropout(features, p=self.args.dropout, training=self.training)
        features = self.convolution_2(features, edge)
        features = F.relu(features)
        features = F.dropout(features, p=self.args.dropout, training=self.training)
        features = self.convolution_3(features, edge)
        features = F.relu(features)
        features = F.dropout(features, p=self.args.dropout, training=self.training)
        features = self.convolution_4(features, edge)
        # features = F.relu(features)
        # features = F.dropout(features, p=self.args.dropout, training=self.training)
        # features = self.convolution_5(features, edge)

        return features

    def forward(self, feat, A, h1id, batch):

        #batchvetor = []
        # 获得一跳邻居的个数
        one_hop_idcs = h1id.view(batch, self.args.k_at_hop[0])

        k1 = one_hop_idcs.size(-1)
        score = self.ginconvpass(edge=A, features=feat)
        score = score.view(batch,-1,self.args.filters_4)
        # for i in range(batch):
        #     for j in range(k1):
        #         batchvetor.append(i)
        # batchvetor = torch.tensor(batchvetor).cuda()
        dout = score.size(-1)
        # 初始化一个全零的维度是 B 批次 * k1 行 * dout 列 的边特征值矩阵     也就是计算特征的时候，考虑[200,10]两跳邻居，反向传播的时候，只考虑第一跳的200邻居
        edge_feat = torch.zeros(batch, k1, dout).cuda()

        for b in range(batch):
            # 从训练好的特征值矩阵x（B批次 * max_num_nodes行 * dout列） 中取出第b批次，第one_hop_idcs[b]行（具体要看这个list里面对应的值），dout列的 特征值进行更新
            edge_feat[b, :, :] = score[b, one_hop_idcs[b]]
        # 创建一个0到batch的
        edge_feat = edge_feat.view(-1, dout)
        pred = self.classifier(edge_feat)

        return pred


