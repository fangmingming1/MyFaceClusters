from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import numpy as np
import time

class Data(object):
    def __init__(self, name):
        self.__name = name
        self.__links = set()

    @property
    def name(self):
        return self.__name

    # @property最大的好处就是在类中把一个方法变成属性调用，起到既能检查属性，还能用属性的方式来访问该属性的作用。
    @property
    def links(self):
        return set(self.__links)

    def add_link(self, other, score):
        self.__links.add(other)
        other.__links.add(self)


# 链接组件
def connected_components(nodes, score_dict, th):
    '''
    conventional connected components searching
    '''
    result = []
    nodes = set(nodes) # 集合（set）是一个无序的不重复元素序列。
    while nodes:
        n = nodes.pop()  # set.pop() 方法用于随机移除一个元素。
        group = {n}
        queue = [n]
        while queue:
            # pop() 函数用于移除列表中的一个元素（默认最后一个元素），并且返回该元素的值。 pop(0)移除第一个元素并且返回
            n = queue.pop(0)
            if th is not None:
                neighbors = {l for l in n.links if score_dict[tuple(sorted([n.name, l.name]))] >= th}
            else:
                neighbors = n.links
            neighbors.difference_update(group)
            nodes.difference_update(neighbors)
            group.update(neighbors)
            queue.extend(neighbors)
            result.append(group)

    return result


# 连接组件约束
def connected_components_constraint(nodes, max_sz, score_dict=None, th=None):
    '''
    only use edges whose scores are above `th`
    只使用分数高于“th”的边
    if a component is larger than `max_sz`, all the nodes in this component are added into `remain` and returned for next iteration.
    如果一个组件大于' max_sz '，该组件中的所有节点都将被添加到' remain '中，并返回给下一次迭代。
    '''
    result = []
    remain = set()

    #将顶点vertex集合转成set
    nodes = set(nodes)

    # 循环遍历里面每一个Data对象
    while nodes:
        # 随机的从set里面移除一个Data对象
        n = nodes.pop()
        # 将n加入到一个set中，group用于接收返回最终的聚类图
        group = {n}
        # 将n的links（set集合）里面的n的邻居对象Data，加入到一个list中
        queue = [n]
        valid = True
        # 循环遍历队列，遍历所有的邻居    以树的层次遍历的方式
        while queue:
            # 使队头元素出队
            n = queue.pop(0)

            # th 是设置的分数阈值
            if th is not None:
                # tuple是元组
                # 如果score_dict里，当前节点n和他的链接节点的边的分数大于阈值th，那么将l(n的邻居节点Data对象)加入到neighbors里
                neighbors = {l for l in n.links if score_dict[tuple(sorted([n.name, l.name]))] >= th}
            else:
                # 就将节点n对象里的links集合（links是set集合，里面有节点n的所有邻居节点的Data对象）加入到neighbors
                neighbors = n.links

            # difference_update() 方法用于移除两个集合中都存在的元素。
            # difference_update() 方法与 difference() 方法的区别在于 difference() 方法返回一个移除相同元素的新集合，
            # 而 difference_update() 方法是直接在原来的集合中移除元素，没有返回值。
            # 所以说这行代码就是将已经加入到group的顶点Data对象从neighbors中移除，防止出现重复的元素
            neighbors.difference_update(group)
            # 将已经确定好 是当前节点n的邻居 从nodes（所有的节点中去掉），因为我们的任务是要将这些点构成聚类图，已经确定好的就删掉
            nodes.difference_update(neighbors)
            # update() 方法用于修改当前集合，可以添加新的元素或集合到当前集合中，如果添加的元素在集合中已存在，则该元素只会出现一次，重复的会忽略。 将已经确定好的邻居更新到聚类图的节点group里
            group.update(neighbors)
            # extend() 函数用于在列表末尾一次性追加另一个序列中的多个值（用新列表扩展原来的列表）。这个也就相当于层次遍历，从一个节点延伸的各个邻居可以当做树，采用层次遍历的方法，来实现遍历从一个节点出发的所有节点。然后再遍历nodes中剩下的的节点
            queue.extend(neighbors)

            #
            if len(group) > max_sz or len(remain.intersection(neighbors)) > 0:
                # if this group is larger than `max_sz`, add the nodes into `remain`
                # 如果这个组大于' max_sz '，将节点添加到' remain '中
                valid = False
                remain.update(group)
                break
        if valid: # if this group is smaller than or equal to `max_sz`, finalize it.
                  # 如果这个组小于或等于' max_sz '，结束它。
            result.append(group)
    return result, remain

# 图传播初始化
def graph_propagation_naive(edges, score, th):

    edges = np.sort(edges, axis=1)
    
    # construct graph
    score_dict = {} # score lookup table
    for i,e in enumerate(edges):
        score_dict[e[0], e[1]] = score[i]

    nodes = np.sort(np.unique(edges.flatten()))
    mapping = -1 * np.ones((nodes.max()+1), dtype=np.int)
    mapping[nodes] = np.arange(nodes.shape[0])
    link_idx = mapping[edges]
    vertex = [Data(n) for n in nodes]
    for l, s in zip(link_idx, score):
        vertex[l[0]].add_link(vertex[l[1]], s)

    # first iteration
    comps = connected_components(vertex, score_dict,th)

    return comps

# 图传播
def bfs_clustering(edges, score, max_sz, step=0.1, beg_th=0.9, pool=None):

    # 将得到的边集，按y轴维度来进行排序，也就是将每一行从小到大排序
    edges = np.sort(edges, axis=1)

    # 从分数当中找到最小值出来
    # th = beg_th
    th = score.min()
    # construct graph

    score_dict = {}  # score lookup table 分数查询表 是一个set集合

    if pool is None:
        # enumerate多用于在for循环中得到计数，利用它可以同时获得索引和值，即需要 index 和 value 值的时候可以使用enumerate；
        # 这个是直接用新的分数，覆盖旧的分数
        for i, e in enumerate(edges):
            score_dict[e[0], e[1]] = score[i]
    elif pool == 'avg':
        # i 是edges的每一行的索引，e是每一行对应的两个连接顶点的索引
        for i, e in enumerate(edges):
            # if score_dict.has_key((e[0],e[1])):
            # 如果两个链接的顶点在分数查询表里
            if (e[0], e[1]) in score_dict:
                # 则需要将之前的分数和现在的分数求个平均
                score_dict[e[0], e[1]] = 0.5*(score_dict[e[0], e[1]] + score[i])
            else:
                # 如果不在的话，将两个顶点的链接可能性增加到score_dict 如key， 和value   { (0, 14472) : 0.804916501045227}
                score_dict[e[0], e[1]] = score[i]

    elif pool == 'max':
        # 从新的和旧的分数选择一个最大的放入score_dict
        for i, e in enumerate(edges):
            # if score_dict.has_key((e[0],e[1])):
            if (e[0], e[1]) in score_dict:
                score_dict[e[0], e[1]] = max(score_dict[e[0], e[1]], score[i])
            else:
                score_dict[e[0], e[1]] = score[i]
    else:
        raise ValueError('Pooling operation not supported')

    # flatten将高维数据转化为一维数据，np.unique( )的用法 该函数是去除数组中的重复数字,并进行排序之后输出。
    nodes = np.sort(np.unique(edges.flatten()))

    # np.ones()函数返回给定形状和数据类型的新数组，其中元素的值设置为1。此函数与numpy zeros()函数非常相似。
    mapping = -1 * np.ones((nodes.max()+1), dtype=np.int)

    # 一个参数时，参数值为终点值，起点取默认值0，步长取默认值1。
    # nodes是18171的数组，也就是将mapping按照索引赋值   nodes.shape[0]是行坐标
    mapping[nodes] = np.arange(nodes.shape[0])

    # link_idx
    link_idx = mapping[edges]

    # vertex是一个顶点列表，里面存的是一个顶点对象Data，循环遍历nodes中18171个点，将每一个点的索引传入，并且初始化一个set与之对应
    # 所以说vertex是一个存放了18171个点的Data对象的list集合
    vertex = [Data(n) for n in nodes]

    # zip()函数用于将两个列表相对应位置的元素打包成元组。
    # 这个zip循环的意义是，将与某个顶点相连的所有顶点 加入到这个顶点的对象data中，
    # 这样我们就可以通过这个点的data对象的links方法，返回与之相连的所有的顶点的data对象，再通过name方法就可以返回这些data对象所对应的顶点的name（标签）
    for l, s in zip(link_idx, score):
        # vertex[l[0]]是两个相连的点的 第一个点索引，
        vertex[l[0]].add_link(vertex[l[1]], s)

    # first iteration 先不设置边的阈值，和传入边的分数，看是否能将所有顶点聚类成最终的图
    comps, remain = connected_components_constraint(vertex, max_sz)

    # second iteration  copms是一个列表，comps[:] 相当于重新创建一个列表对象，并且把值赋给 component，这个是用来接收迭代时生成的聚类图（顶点Data对象的set集合）
    components = comps[:]
    while remain:
        th = th + (1 - th) * step  # th的缓慢递增，通过不断提高th的值，来使remain里剩下的节点越来越少，使得其构成聚类图
        comps, remain = connected_components_constraint(remain, max_sz, score_dict, th)
        components.extend(comps)
    return components

def graph_propagation_soft(edges, score, max_sz, step=0.1, **kwargs):

    edges = np.sort(edges, axis=1)
    th = score.min()

    # construct graph
    score_dict = {} # score lookup table
    for i,e in enumerate(edges):
        score_dict[e[0], e[1]] = score[i]

    nodes = np.sort(np.unique(edges.flatten()))
    mapping = -1 * np.ones((nodes.max()+1), dtype=np.int)
    mapping[nodes] = np.arange(nodes.shape[0])
    link_idx = mapping[edges]
    vertex = [Data(n) for n in nodes]
    for l, s in zip(link_idx, score):
        vertex[l[0]].add_link(vertex[l[1]], s)

    # first iteration
    comps, remain = connected_components_constraint(vertex, max_sz)
    first_vertex_idx = np.array([mapping[n.name] for c in comps for n in c])
    fusion_vertex_idx = np.setdiff1d(np.arange(nodes.shape[0]), first_vertex_idx, assume_unique=True)
    # iteration
    components = comps[:]
    while remain:
        th = th + (1 - th) * step
        comps, remain = connected_components_constraint(remain, max_sz, score_dict, th)
        components.extend(comps)
    label_dict = {}
    for i,c in enumerate(components):
        for n in c:
            label_dict[n.name] = i
    print('Propagation ...')
    prop_vertex = [vertex[idx] for idx in fusion_vertex_idx]
    label, label_fusion = diffusion(prop_vertex, label_dict, score_dict, **kwargs)
    return label, label_fusion

def diffusion(vertex, label, score_dict, max_depth=5, weight_decay=0.6, normalize=True):
    class BFSNode():
        def __init__(self, node, depth, value):
            self.node = node
            self.depth = depth
            self.value = value
            
    label_fusion = {}
    for name in label.keys():
        label_fusion[name] = {label[name]: 1.0}
    prog = 0
    prog_step = len(vertex) // 20
    start = time.time()
    for root in vertex:
        if prog % prog_step == 0:
            print("progress: {} / {}, elapsed time: {}".format(prog, len(vertex), time.time() - start))
        prog += 1
        #queue = {[root, 0, 1.0]}
        queue = {BFSNode(root, 0, 1.0)}
        visited = [root.name]
        root_label = label[root.name]
        while queue:
            curr = queue.pop()
            if curr.depth >= max_depth: # pruning
                continue
            neighbors = curr.node.links
            tmp_value = []
            tmp_neighbor = []
            for n in neighbors:
                if n.name not in visited:
                    sub_value = score_dict[tuple(sorted([curr.node.name, n.name]))] * weight_decay * curr.value
                    tmp_value.append(sub_value)
                    tmp_neighbor.append(n)
                    if root_label not in label_fusion[n.name].keys():
                        label_fusion[n.name][root_label] = sub_value
                    else:
                        label_fusion[n.name][root_label] += sub_value
                    visited.append(n.name)
                    #queue.add([n, curr.depth+1, sub_value])
            sortidx = np.argsort(tmp_value)[::-1]
            for si in sortidx:
                queue.add(BFSNode(tmp_neighbor[si], curr.depth+1, tmp_value[si]))
    if normalize:
        for name in label_fusion.keys():
            summ = sum(label_fusion[name].values())
            for k in label_fusion[name].keys():
                label_fusion[name][k] /= summ
    return label, label_fusion
