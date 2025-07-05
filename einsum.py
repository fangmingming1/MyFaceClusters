import torch
import numpy as np
# a = torch.tensor(([[1,2],[3,4]]))
# b = torch.tensor(([[4,3],[2,1]]))
# c = torch.einsum("ik,kj->ij", a, b)
# print(a)
# print(b)
# print(c)
#
#
#
# e={3,4}
# d=[3,7]
# e.update(d)
# print(e)
#
# f =np.array([[1,2,3,4,5,6],[7,8,9,10,11,12]])
# print(f)
# g =np.array([1,2])
# print(g)
#
#
# _ , fdim =f.shape
# print(_)
# print(fdim)
#
#
# A=torch.zeros(10,10)
# print(A)
# D = A.sum(1, keepdim=False)
# print(D)
#
# A=A.view(-1)
# print(A)
# print(A.shape)

# knn_graph_dict = list()
# knn_graph_dict.append(dict())
# knn_graph_dict.append(dict())
# knn_graph_dict[-1][1] = []
# knn_graph_dict[-1][0] = []
# print(knn_graph_dict)

# a=np.array([[1,2],[3,4],[5,6]],dtype=int)
# print(a)
#
# for i,h in enumerate(a) :
#     print(i,h)

# idcs=[False,False,False,False,True]
# idcs=np.array(idcs)
# print(idcs)
# print(idcs.sum())
#
# pred=np.array((1,1,1,1,1,1,1,1))
# single_idcs = np.zeros_like(pred)
# print('single_idcs',single_idcs.shape)
#
# a = np.where(idcs)
# print('a is ', a)
# single_idcs[a]=1
# print('single_idcs :',single_idcs)

syslabels = np.array([9,8,7,6,5,4,3,2,1,1,1,2,2])
sysclasses,sysindex = np.unique(syslabels,return_inverse=True)
print('syslabels : ',syslabels)
print('sysclasses : ', sysclasses)
print('sysindex : ',sysindex)  # 返回的index是输入数组中元素，在输出unique数组中的位置索引
# syslabels :  [9 8 7 6 5 4 3 2 1 1 1 2 2]
# sysclasses :  [1 2 3 4 5 6 7 8 9]
# sysindex :  [8 7 6 5 4 3 2 1 0 0 0 1 1]