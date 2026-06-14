# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sympy.physics.control.control_plots import matplotlib

matplotlib.rcParams['legend.handlelength'] = 0
i=0
label_names = ['品质','味道','价格','分量','外观','物流','客服','评论的粗粒度评价']

def get_fer_data(data_path="D:\Pythonnnn\弱监督的农业社会化销售服务评价\测试集降维\data_embed_npy{}_10.npy".format(i),
                 label_path="D:\Pythonnnn\弱监督的农业社会化销售服务评价\测试集降维\label_npu{}_10.npy".format(i)):
    """
	该函数读取上一步保存的两个npy文件，返回data和label数据
    Args:
        data_path:
        label_path:

    Returns:
        data: 样本特征数据，shape=(BS,embed)
        label: 样本标签数据，shape=(BS,)
        n_samples :样本个数
        n_features：样本的特征维度

    """
    data = np.load(data_path)
    label = np.load(label_path)
    n_samples, n_features = data.shape

    return data, label, n_samples, n_features


color_map = ['r', 'y', 'k']  # 7个类，准备7种颜色
# shape=['.','*','1']

def plot_embedding_2D(data, label):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)
    fig = plt.figure()
    for j in range(data.shape[0]):
        plt.plot(data[j, 0], data[j, 1], marker='o', markersize=5, color=color_map[label[j]])
        # plt.plot(data[j, 0], data[j, 1], marker=shape[label[j]], markersize=6,color="k")
    plt.xticks([])
    plt.yticks([])
    plt.rcParams['font.family'] = ['STSong']  # 关键是这句
    plt.legend(["class 0","class 1","class 2"],loc='best')
    plt.title("{}维度".format(label_names[i]),FontProperties="STSong",fontsize=13)
    # plt.title("{}".format(label_names[i]),FontProperties="STSong",fontsize=13)

    return fig


def main():
    data, label, n_samples, n_features = get_fer_data()  # 根据自己的路径合理更改

    print('Begining......')

    # 调用t-SNE对高维的data进行降维，得到的2维的result_2D，shape=(samples,2)
    tsne_2D = TSNE(n_components=2, init='pca', random_state=0)
    result_2D = tsne_2D.fit_transform(data)

    print('Finished......')
    fig1 = plot_embedding_2D(result_2D, label)  # 将二维数据用plt绘制出来
    # fig1.show()
    fig1.savefig("./zl/{}.jpg".format(i))


if __name__ == '__main__':
    main()

