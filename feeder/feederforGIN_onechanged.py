import numpy as np
import random
import torch
import torch.utils.data as torchdata
from torch_geometric.data import Data,Dataset
import time
'''
  首先对IJB-B数据进行特征提取，将特征保存为NxD维.npy文件，其中每一行是一个样本的D维特征。
  然后，将标签保存为一个Nx1维的.npy文件，每一行都是一个表示身份的整数。
  最后，生成 KNN 图（通过蛮力或 ANN）。KNN图要保存为一个Nx(K+1)维的.npy文件，每一行的第一个元素是节点索引，后面的K个元素是其KNN节点的索引。
'''
class GINFeederTWO(Dataset):
    def __init__(self, arg, feat_path, knn_graph_path, label_path, seed=1,
                 k_at_hop=[200, 5], active_connection=5, train=True):
        super().__init__()
        np.random.seed(seed)
        random.seed(seed)
        self.features = np.load(feat_path)
        # print('k_at_hop',k_at_hop)
        # print('k_at_hop[0]',k_at_hop[0])
        # 加载完的knn_graph是N*201的ndarray数组。[:,:k_at_hop[0]+1]这个意思是取 所有行，前 0到k_at_hop[0]+1 列 包括k_at_hop[0]+1
        # [: , k_at_hop[0]+1 : ] 则是取 所有行，k_at_hop[0]+1 到 最后一列 不包括k_at_hop[0]+1
        self.knn_graph = np.load(knn_graph_path)[:, :k_at_hop[0] + 1]
        self.labels = np.load(label_path)
        self.num_samples = len(self.features)
        # depth为跳跃的深度，这里也就2跳，第一跳200个邻居，第二跳5个邻居
        self.depth = len(k_at_hop)
        self.k_at_hop = k_at_hop
        self.active_connection = active_connection
        self.train = train
        self.arg = arg
        assert np.mean(k_at_hop) >= active_connection


    def len(self):
        return self.num_samples

    def get(self, index):
        '''
                return the vertex feature and the adjacent matrix A, together
                with the indices of the center node and its 1-hop nodes
                返回一个KNN子图 的 顶点特征和邻接矩阵A 中心节点和它的1跳节点的索引
                '''
        # hops[0] for 1-hop neighbors, hops[1] for 2-hop neighbors
        timing1 = time.time()
        hops = list()
        # 获得中心节点的索引
        center_node = index
        # print('index: ',index)
        # 将knn—子图 依据 中心节点索引，加入到hops中
        # 在512训练集下knn_graph是一个（18171,201）的二维数组，18171是人脸节点的个数，201是1（中心节点索引）+200（KNN图节点索引）
        hops.append(set(self.knn_graph[center_node][1:]))
        # print('hops ', hops) 此时hops是一个的list里面有一个1*201的set

        # Actually we dont need the loop since the depth is fixed here,
        # But we still remain the code for further revision
        # 实际上我们不需要循环，因为深度在这里是固定的，
        # 但我们仍然保留了进一步修订的代码
        for d in range(1, self.depth):  # 因为depth为2 所以这个d一直都是1
            hops.append(set())
            # list的负值逆序访问，-1是倒数第一个，-2是倒数第二个，在经过hops.append(set())之后，此时hops里有两个set集合，
            # hops[0]第一个set集合存放的是1-hop neighbors，hops[1]第二个set集合存放的是 2-hop neighbors
            for h in hops[-2]:
                # 所以这个for循环也就是循环遍历hops[0]，也就是第一邻居集合，然后再将第二跳邻居集合加入进去。
                # hopsp[-1]里面存的是一个set集合，set.update()方法是更新集合元素
                # k_at_hop[d] + 1 = 5 + 1 = 6
                # 所以 self.knn_graph[h][1:self.k_at_hop[d]+1] 得出的是 每个枢轴节点 的1-hop邻居的前5个邻居
                # k_at_hop=[200,10]所以知道了这个参数的意思，第一跳200个邻居，第二跳10个邻居
                hops[-1].update(set(self.knn_graph[h][1:self.k_at_hop[d] + 1]))

        '''
        用于列表的嵌套中
        先遍历c，再遍历b，把其中的元素a存到列表中
        c = [[7,8,9],[1,2,3],[4,5,6]]          等价于    c = [[7,8,9],[1,2,3],[4,5,6]]
        l = [a for b in c for a in b]                   k = []
        print(l)                                        for b in c:
                                                          for a in b:
        # [7, 8, 9, 1, 2, 3, 4, 5, 6]                        k.append(a)
        '''
        # 所以这个hops_set是将hops[0]第一跳邻居索引,hops[1]第二跳邻居索引,扁平化，水平拼接成一个一维list
        #               a      b       c      a     b
        hops_set = set([h for hop in hops for h in hop])

        # print('hops_set', len(hops_set))
        # print('[center_node,]',[center_node,])
        # set是一个无序的不重复元素序列 这个操作是将中心点加入到hops_set中，如果有的话，就不更新
        hops_set.update([center_node, ])
        # print('hops_set after update', len(hops_set))

        # 根据hops_set获得当前中心节点的第1跳，2跳 邻居，肯定是不重复的   有可能第一跳邻居的邻居是同一个节点，此时只需要保留一个即可
        unique_nodes_list = list(hops_set)
        # 获得 不重复节点的映射，将list中的每个1,2跳节点索引当做key   list中的序号当做映射值value
        unique_nodes_map = {j: i for i, j in enumerate(unique_nodes_list)}
        # print(type(unique_nodes_map))    dict  字典类型

        # 在unique_nodes_map里  根据中心节点索引，获得对应的list中的序号，得到center_idx
        center_idx = torch.Tensor([unique_nodes_map[center_node], ]).type(torch.long)
        #    print('center_idx', center_idx)
        # print('center_idx',center_idx.shape)

        # 根据hops[0]里第一跳邻居索引，从unique_nodes_map里获得对应list中的序号，得到one_hop_idcs
        one_hop_idcs = torch.Tensor([unique_nodes_map[i] for i in hops[0]]).type(torch.long)
        #    print('one_hop_idcs', one_hop_idcs.shape)

        # 获的枢轴子图 中心点 特征
        center_feat = torch.Tensor(self.features[center_node]).type(torch.float)
        #    print('center_feat', center_feat.shape)

        # unique_nodes_list是包含了1,2跳邻居索引的list，所以feat是1,2跳邻居的特征值
        feat = torch.Tensor(self.features[unique_nodes_list]).type(torch.float)
        #    print('feat', feat.shape)

        # 使用1-hop邻居特征 减去 枢轴子图中心点特征 进行归一化
        feat = feat - center_feat

        # 获得最大的节点数目，第一跳的邻居节点数*（第二跳邻居节点数+1）+1   比如在512训练集下 200*11+1=2201  最后一个加一应该是中心节点
        max_num_nodes = self.k_at_hop[0] * (self.k_at_hop[1] + 1) + 1

        # 通过unique_nodes_list获得节点的数目
        num_nodes = len(unique_nodes_list)

        # 创建一个全0的邻接矩阵A
        A = torch.zeros(num_nodes, num_nodes)

        _, fdim = feat.shape
        #  print('_',_)    _输出的是feat的行数，也就是特征的个数
        #  print('fdim',fdim)  fdim输出的是feat的列数，也就是特征的维度
        #  print('feat.shape',feat.shape)
        # 构建一个全零的max_num_nodes - num_nodes * fdim维的数组，在x轴方向上拼接feat，因为此时的feat是不重复节点的feat
        feat = torch.cat([feat, torch.zeros(max_num_nodes - num_nodes, fdim)], dim=0)
        #  print('torch.zeros(max_num_nodes - num_nodes, fdim)',torch.zeros(max_num_nodes - num_nodes, fdim).shape)
        #  print('feat',feat.shape)
        A_complex = set()
        # 这个for循环就是为了获得更新后邻接矩阵A
        # timing2 = time.time()
        # print("阶段一的时间为{}".format(timing2-timing1))
        for node in unique_nodes_list:
            # 获得 node的 邻居 下标 1到self.active_connection
            neighbors = self.knn_graph[node, 1:self.active_connection + 1]
            # print('neighbors.shape ',neighbors.shape)  neighbors.shape  (10,)
            # 循环遍历所有邻居， 如果node 的邻居中n在unique_nodes_list 里面，则要更新邻接矩阵
            for n in neighbors:
                if n in unique_nodes_list:
                    # # 记住此时unique_nodes_map是一个字典  key是节点索引，value是对应list的数组索引
                    # # 而A又是len(unique_nodes_list)*len(unique_nodes_list)的全零矩阵
                    # A[unique_nodes_map[node], unique_nodes_map[n]] = 1
                    # A[unique_nodes_map[n], unique_nodes_map[node]] = 1
                    # 构造元组
                    edge = (unique_nodes_map[n], unique_nodes_map[node]) if unique_nodes_map[n] > unique_nodes_map[node] else (unique_nodes_map[node], unique_nodes_map[n])
                    A_complex.add(edge)
                    # 利用set集合的唯一性去重，两点之间，只需要一条边就够了
        A_complex = list(A_complex)
        A_complex = torch.tensor(A_complex).T
        # timing3 = time.time()
        # print("阶段二的时间为{}".format(timing3-timing2))
        #   # 对A的按照y轴方向进行求和 这样D中每行的值为A矩阵每个节点的度
        #   D = A.sum(1, keepdim=True)
        # #  print(D.shape)    tensor（1027,1）
        #   # 用A除D    这样除完之后，邻接矩阵的每一行求和都是1
        #   A = A.div(D)
        # #  print(A.shape)
        #   # 创建一个大矩阵A_
        #   A_ = torch.zeros(max_num_nodes,max_num_nodes)
        #   # 将A_的前num_nodes行，前num_nodes列更新为A
        #   A_[:num_nodes,:num_nodes] = A
        # #  print(A_.shape)

        # 创建标签labels
        # numpy.asarray(a, dtype=None, order=None) 作用是将输入转换为数组
        # 参数：
        #     a：输入数据，可以转换为数组的任何形式。这包括列表，元组列表，元组，元组，列表元组和ndarray。
        #     dtype: 默认情况下，从输入数据中推断出数据类型
        #     order: 是使用行优先（C风格）还是列优先（Fortran风格）内存表示形式。
        # print(unique_nodes_list)
        # print(np.asarray(unique_nodes_list))
        # print(self.labels)
        labels = self.labels[np.asarray(unique_nodes_list)]
        # print(labels) # 1027*1

        # 将数组labels转化为tensor格式
        labels = torch.from_numpy(labels).type(torch.long)
        # edge_labels = labels.expand(num_nodes,num_nodes).eq(
        #        labels.expand(num_nodes,num_nodes).t())

        # 获得1跳邻居属于的聚类图标签   one_hop_idcs是unique_nodes_list里面 一跳邻居 所对应的list下标，而labels又是通过unique_nodes_list的节点索引，得到的标签值，所以labels的下标和unique_nodes_list的下标是对应的，那么相同的索引下，unique_nodes_list里面的节点索引和labels里面的标签值也是对应匹配的
        one_hop_labels = labels[one_hop_idcs]
        # print(one_hop_idcs)
        # print(one_hop_labels)
        # 获得中心节点的属于的聚类图标签
        center_label = labels[center_idx]
        # print(center_idx)
        # print(center_label)

        # 获得边的标签   那么如此可见，此时labels里面装的是每个点所对应的真实的聚类图编号， 通过比较center_label和one_hop_labels的标签值是否相同来判断是否属于同一个聚类图
        edge_labels = (center_label == one_hop_labels).long()
        # print(edge_labels)
        if self.train:
            # return (feat, A_, center_idx, one_hop_idcs), edge_labels
            graph_data = Data(x=feat, edge_index=A_complex, y=edge_labels, center_idx=center_idx,
                              one_hop_idcs=one_hop_idcs, edge_label=edge_labels)
            # timing4 = time.time()
            # print("阶段三的时间为{}".format(timing4-timing3))
            return graph_data

            # Testing
        unique_nodes_list = torch.Tensor(unique_nodes_list)
        unique_nodes_list = torch.cat(
            [unique_nodes_list, torch.zeros(max_num_nodes - num_nodes)], dim=0)
        graph_data = Data(x=feat, edge_index=A_complex, y=edge_labels, center_idx=center_idx,
                          one_hop_idcs=one_hop_idcs, unique_nodes_list=unique_nodes_list, edge_labels=edge_labels)
        return graph_data