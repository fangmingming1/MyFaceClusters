###################################################################
# File Name: utils.py
# Author: Zhongdao Wang
# mail: wcd17@mails.tsinghua.edu.cn
# Created Time: Tue 28 Aug 2018 04:57:29 PM CST
###################################################################

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import numpy as np
from scipy.sparse import coo_matrix
from sklearn.metrics.cluster import (contingency_matrix,
                                     normalized_mutual_info_score)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def norm(X):
    for ix,x in enumerate(X):
        X[ix]/=np.linalg.norm(x)
    return X

def plot_embedding(X,Y):
    x_min, x_max = np.min(X,0), np.max(X,0)
    X = (X - x_min) / (x_max - x_min)
    plt.figure(figsize=(10,10))
    for i in range(X.shape[0]):
        plt.text(X[i,0],X[i,1], str(Y[i]),
                color=plt.cm.Set1(Y[i]/10.),
                fontdict={'weight':'bold','size':12})
    plt.savefig('a.jpg')


EPS = np.finfo(float).eps


def contingency_matrix1(ref_labels, sys_labels):
    """Return contingency matrix between ``ref_labels`` and ``sys_labels``.
       返回' ref_labels '和' sys_labels '之间的权变矩阵
    """
    # 关于np.unique的具体使用可以看 http://manongjc.com/detail/31-mcuktrnmbkasigz.html
    # numpy.unique() 函数查找数组的唯一元素并将这些唯一元素作为排序数组返回。
    # return_inverse:bool(可选)
    # 如果此参数设置为 True，返回的index索引数组 是输入数组中元素，在输出unique数组中的位置索引
    ref_classes, ref_class_inds = np.unique(ref_labels, return_inverse=True)
    sys_classes, sys_class_inds = np.unique(sys_labels, return_inverse=True)
    n_frames = ref_labels.size   # 获得参考聚类的数目
    # Following works because coo_matrix sums duplicate entries. Is roughly
    # twice as fast as np.histogram2d. 下面的工作是因为coo_matrix对重复项求和。大约是np.histogram2d的两倍快。
    cmatrix = coo_matrix(  # https://blog.csdn.net/haoji007/article/details/105696394/ 详细的解释可以看这里 ，构建稀疏矩阵，类似于数据结构三元表的方式
        (np.ones(n_frames), (ref_class_inds, sys_class_inds)),
        shape=(ref_classes.size, sys_classes.size),
        dtype=np.int)  # coo_matrix（（data，（row，col）），shape=（n，m））
    cmatrix = cmatrix.toarray()
    return cmatrix, ref_classes, sys_classes


def bcubed(ref_labels, sys_labels, cm=None):
    """Return B-cubed precision, recall, and F1.

    The B-cubed precision of an item is the proportion of items with its
    system label that share its reference label (Bagga and Baldwin, 1998).
    Similarly, the B-cubed recall of an item is the proportion of items
    with its reference label that share its system label. The overall B-cubed
    precision and recall, then, are the means of the precision and recall for
    each item.

    Parameters
    ----------
    ref_labels : ndarray, (n_frames,)
        Reference labels. 参考标签

    sys_labels : ndarray, (n_frames,)
        System labels. 实际标签

    cm : ndarray, (n_ref_classes, n_sys_classes)
        Contingency matrix between reference and system labelings. If None,
        will be computed automatically from ``ref_labels`` and ``sys_labels``.
        Otherwise, the given value will be used and ``ref_labels`` and
        ``sys_labels`` ignored.
        cm是参考标签和系统标签之间的权变矩阵。
        如果为None，将从' ' ref_labels ' '和' sys_labels ' '自动计算，否则，将使用给定的值
        ，' ' ref_labels ' '和' sys_labels ' '被忽略。
        (Default: None)

    Returns
    -------
    precision : float
        B-cubed precision.

    recall : float
        B-cubed recall.

    f1 : float
        B-cubed F1.

    References
    ----------
    Bagga, A. and Baldwin, B. (1998). "Algorithms for scoring coreference
    chains." Proceedings of LREC 1998.
    """
    if cm is None:
        cm, _, _ = contingency_matrix(ref_labels, sys_labels)
    cm = cm.astype('float64')
    cm_norm = cm / cm.sum()  # 对cm进行归一化
    precision = np.sum(cm_norm * (cm / cm.sum(axis=0)))  #
    recall = np.sum(cm_norm * (cm / np.expand_dims(cm.sum(axis=1), 1)))
    f1 = 2*(precision*recall)/(precision + recall)
    return precision, recall, f1

def fowlkes_mallows_score(pred_labels, gt_labels, sparse=True):
    ''' The original function is from `sklearn.metrics.fowlkes_mallows_score`.
        We output the pairwise precision, pairwise recall and F-measure,
        instead of calculating the geometry mean of precision and recall.
    '''
    n_samples, = gt_labels.shape

    c = contingency_matrix(gt_labels, pred_labels, sparse=sparse)
    tk = np.dot(c.data, c.data) - n_samples
    pk = np.sum(np.asarray(c.sum(axis=0)).ravel()**2) - n_samples
    qk = np.sum(np.asarray(c.sum(axis=1)).ravel()**2) - n_samples

    avg_pre = tk / pk
    avg_rec = tk / qk
    fscore = _compute_fscore(avg_pre, avg_rec)

    return avg_pre, avg_rec, fscore

def _compute_fscore(pre, rec):
    return 2. * pre * rec / (pre + rec)

def bcubed1(gt_labels, pred_labels):

    gt_lb2idxs = _get_lb2idxs(gt_labels)
    pred_lb2idxs = _get_lb2idxs(pred_labels)

    num_lbs = len(gt_lb2idxs)
    pre = np.zeros(num_lbs)
    rec = np.zeros(num_lbs)
    gt_num = np.zeros(num_lbs)

    for i, gt_idxs in enumerate(gt_lb2idxs.values()):
        all_pred_lbs = np.unique(pred_labels[gt_idxs])
        gt_num[i] = len(gt_idxs)
        for pred_lb in all_pred_lbs:
            pred_idxs = pred_lb2idxs[pred_lb]
            n = 1. * np.intersect1d(gt_idxs, pred_idxs).size
            pre[i] += n**2 / len(pred_idxs)
            rec[i] += n**2 / gt_num[i]

    gt_num = gt_num.sum()
    avg_pre = pre.sum() / gt_num
    avg_rec = rec.sum() / gt_num
    fscore = _compute_fscore(avg_pre, avg_rec)

    return avg_pre, avg_rec, fscore

def _get_lb2idxs(labels):
    lb2idxs = {}
    for idx, lb in enumerate(labels):
        if lb not in lb2idxs:
            lb2idxs[lb] = []
        lb2idxs[lb].append(idx)
    return lb2idxs