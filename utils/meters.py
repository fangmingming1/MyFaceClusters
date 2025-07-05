from __future__ import absolute_import


class AverageMeter(object):
    """Computes and stores the average and current value"""
    # AverageMeter()函数目的是计算一个epoch的平均损失，即将每个样本的损失取一下平均。
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
