import numpy as np
import pandas as pd
import seaborn
from matplotlib import pyplot as plt


def create_adjacency_matrix_cooccurance(data_label):
    # data_label = data_label.replace({0: 1, 2: 0})#0换成1，1不变。2换成0
    data_label=data_label.to_numpy()
    #创建一个正方形的全0矩阵
    cooccur_matrix = np.zeros((data_label.shape[1], data_label.shape[1]), dtype=float)
    #从y_train中取出标签
    for y in data_label:
        y = list(y)
        for i in range(len(y)):
            for j in range(len(y)):
                # data_label
                #仅对二标签有效，多标签需要修改此处
                if y[i] == 0 and y[j] == 0:
                    cooccur_matrix[i, j] += 1
    #有几个标签row_sums列表就有几个值。值表示该标签下所有的值之和
    row_sums = data_label.sum(axis=0)

    print(cooccur_matrix)
    #共线矩阵
    #[[0. 0. 0.]
    #[0. 2. 1.]
    #[0. 1. 1.]]

    for i in range(cooccur_matrix.shape[0]):
        for j in range(cooccur_matrix.shape[0]):
            #因为分母不能为0，做的特别处理
            if row_sums[i] != 0:
                cooccur_matrix[i][j] = cooccur_matrix[i, j] / row_sums[i]
            else:
                cooccur_matrix[i][j] = cooccur_matrix[i, j]
    print(cooccur_matrix)
    #row_sums=[0,2,1]。共线矩阵按行除以row_sums 中的值
    #正则化后
    #[[0.  0.  0. ]
    # [0.  1.  0.5]
    # [0.  1.  1. ]]

    return cooccur_matrix

if __name__ == '__main__':
    df=pd.read_csv("D:\Pythonnnn\弱监督的农业社会化销售服务评价\弱监督\my_model训练数据\初始.csv")
    group=['品质','味道','价格','分量','外观','物流','客服','粗粒度']
    df=df[group]
    a=create_adjacency_matrix_cooccurance(df)
    seaborn.heatmap(a, cmap="Blues")
    plt.xticks(np.arange(len(group)), labels=group,
              ha="center",FontProperties="STSong")
    plt.yticks(np.arange(len(group)), labels=group,FontProperties="STSong",va="center")
    plt.show()