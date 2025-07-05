import os
import time
import json
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

if __name__ == '__main__':

    # 配置中文字体，这里使用宋体
    rcParams['font.family'] = 'Times New Roman, SimSun'
    # 配置字体大小
    rcParams['font.size'] = 16
    # 配置正常显示负号
    rcParams['axes.unicode_minus'] = False

    # 理论上线
    # x = [5, 10, 20, 40, 80, 160]
    # F_scores = [87.4, 91.1, 92.8, 94.6, 95.9, 97.0]
    # NMI = [96.0, 96.9, 97.5, 98.1, 98.6, 99.0]
    # plt.figure(figsize=(8, 5))
    # plt.plot(x, F_scores, marker='s', label='Bcubed F-score', markersize=6)
    # plt.plot(x, NMI, marker='o', label='NMI', markersize=6)
    # plt.xlabel('K值')
    # plt.ylabel('NMI & Bcubed F-score（%）')
    # plt.legend(['Bcubed F-score', 'NMI'])
    #
    # plt.grid(True, linestyle="--", alpha=0.5, axis="both")
    # plt.savefig('理论上限图.jpg', dpi=1500,bbox_inches="tight")

    # 固定h1为20
    # h2 = [3, 4, 5, 6, 7]
    # F_pre_20 = [96.19, 95.91, 95.86, 95.81, 95.46]
    # F_recall_20 = [72.58, 72.95, 72.84, 72.86, 72.84]
    # plt.plot(h2, F_recall_20, marker='s', label='recall', markersize=6, color='#FE8010')
    # plt.plot(h2, F_recall_20, marker='s',label='Recall', markersize=6)




    # 整体fscores
    # h2 = [2, 3, 4, 5, 6, 7]
    # F_scores_5 = [77.03, 77.30, 77.13, 76.81, 76.81, 77.32]
    # F_scores_10 = [81.72, 81.57, 81.32, 81.41, 81.32, 81.48]
    # F_scores_20 = [82.31, 82.73, 82.66, 82.78, 82.77, 82.63]
    # F_scores_40 = [82.74, 83.19, 82.54, 82.68, 82.68, 83.00]
    # F_scores_80 = [82.58, 82.77, 82.80, 82.69, 82.77, 82.54]
    # plt.plot(h2, F_scores_5, marker='o', label='h1 = 5', markersize=6)
    # plt.plot(h2, F_scores_10, marker='s', label='h1 = 10', markersize=6)
    # plt.plot(h2, F_scores_20, marker='^', label='h1 = 20', markersize=6)
    # plt.plot(h2, F_scores_40, marker='p', label='h1 = 40', markersize=6)
    # plt.plot(h2, F_scores_80, marker='d', label='h1 = 80', markersize=6)
    # # # 显示图形
    # # for a, b in zip(h2, F_scores_10):
    # #     plt.text(a, b, b, ha='center', va='bottom', fontsize=10)  # 设置数据标签位置及大小
    # plt.legend(['h1 = 5', 'h1 = 10', 'h1 = 20', 'h1 = 40', 'h1 = 80'])
    # plt.xlabel('h2')
    # plt.ylabel('Bcubed Fscore (%)')
    # plt.grid(True, linestyle="--", alpha=0.5, axis="both")
    # plt.savefig('LGIN-h1-h2消融实验图.jpg', dpi=1500,bbox_inches="tight")

    # # 固定h2为5
    h1 = [10, 20, 40, 60, 80]
    F_pre_5 = [96.53, 95.86, 95.04, 94.75, 93.44]
    F_recall_5 = [70.38, 72.84, 73.17, 73.33, 74.15]
    #F_Bcubed_5 = [81.24,82.78,82.68,82.68,82.69]
    plt.plot(h1, F_recall_5, marker='o',label='Recall', markersize=6)    # #FE8010
    plt.plot(h1, F_pre_5, marker='s',label='Precision', markersize=6)
    #plt.plot(h1, F_Bcubed_5, marker='s',label='Bcubed_Fscores', markersize=6)
    for a, b in zip(h1, F_recall_5):
        plt.text(a, b, b, ha='center', va='bottom', fontsize=13)  # 设置数据标签位置及大小
    for a, b in zip(h1, F_pre_5):
        plt.text(a, b, b, ha='center', va='bottom', fontsize=13)  # 设置数据标签位置及大小
    plt.legend(['Recall', 'Precision'])
    # 添加一些标签和标题
    plt.xlabel('h1')
    plt.ylabel('Recall & Precision (%)')
    plt.grid(True, linestyle="--", alpha=0.5, axis="both")
    plt.savefig('固定h2=5的联合变化率曲线.jpg', dpi=1500,bbox_inches="tight")

    # 显示图形
    # for a, b in zip(x, F_scores):
    #     plt.text(a, b, b, ha='center', va='bottom', fontsize=10)  # 设置数据标签位置及大小
    # for a, b in zip(x, NMI):
    #     plt.text(a, b, b, ha='center', va='bottom', fontsize=10)  # 设置数据标签位置及大小


    plt.show()

    print()



