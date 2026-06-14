"""
官方给出的示例代码。官方重写的fasttext方法。
并给出了cleanlab结合fasttext的示例。但不是用来找低置信样本的。cleanlab的COUNT大类都是用来【参数寻优】寻找模型的最佳参数
"""
# https://github.com/cleanlab/examples/blob/master/fasttext_amazon_reviews/fasttext_amazon_reviews.ipynb
"""
执行代码会报错UnicodeDecodeError: 'gbk' codec can't decode byte 0x83 in position 20: illegal multibyte sequence
问题原因：cleanlab的代码中，with open没有加编码方式
解决办法：定位到报错处，在打开文件的with open中加上【encoding='utf-8'】
"""
import json
import pandas as pd
from cleanlab.classification import CleanLearning
from cleanlab.models.fasttext import FastTextClassifier, data_loader
import cleanlab
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.model_selection import ParameterGrid
import os
from datetime import datetime as dt

#定义网格参数
# param_list = ParameterGrid(
#     {
#         "cv_n_folds": [3],
#         "lr": [0.01, 0.05, 0.1, 0.5, 1.0],
#         "ngram": [3],
#         "epochs": [1, 5, 10],
#         "dim": [100],
#     }
# )
#遍历所有参数的组合，空列表 scores，用于存储每个参数组合的模型性能评分
# scores = []
# for i, params in enumerate(param_list):
#     print(i,params)


# 0 {'cv_n_folds': 3, 'dim': 100, 'epochs': 1, 'lr': 0.01, 'ngram': 3}
# 1 {'cv_n_folds': 3, 'dim': 100, 'epochs': 1, 'lr': 0.05, 'ngram': 3}
# 2 {'cv_n_folds': 3, 'dim': 100, 'epochs': 1, 'lr': 0.1, 'ngram': 3}
# 3 {'cv_n_folds': 3, 'dim': 100, 'epochs': 1, 'lr': 0.5, 'ngram': 3}
# 4 {'cv_n_folds': 3, 'dim': 100, 'epochs': 1, 'lr': 1.0, 'ngram': 3}
# 5 {'cv_n_folds': 3, 'dim': 100, 'epochs': 5, 'lr': 0.01, 'ngram': 3}
# 6 {'cv_n_folds': 3, 'dim': 100, 'epochs': 5, 'lr': 0.05, 'ngram': 3}
# 7 {'cv_n_folds': 3, 'dim': 100, 'epochs': 5, 'lr': 0.1, 'ngram': 3}
# 8 {'cv_n_folds': 3, 'dim': 100, 'epochs': 5, 'lr': 0.5, 'ngram': 3}
# 9 {'cv_n_folds': 3, 'dim': 100, 'epochs': 5, 'lr': 1.0, 'ngram': 3}
# 10 {'cv_n_folds': 3, 'dim': 100, 'epochs': 10, 'lr': 0.01, 'ngram': 3}
# 11 {'cv_n_folds': 3, 'dim': 100, 'epochs': 10, 'lr': 0.05, 'ngram': 3}
# 12 {'cv_n_folds': 3, 'dim': 100, 'epochs': 10, 'lr': 0.1, 'ngram': 3}
# 13 {'cv_n_folds': 3, 'dim': 100, 'epochs': 10, 'lr': 0.5, 'ngram': 3}
# 14 {'cv_n_folds': 3, 'dim': 100, 'epochs': 10, 'lr': 1.0, 'ngram': 3}


df = pd.read_csv("D:\Pythonnnn\弱监督的农业社会化销售服务评价\弱监督\my_model训练数据\初始.csv",encoding='utf-8')

"""cleanlab的fasttext函数固定了输入格式。【__label__0 ？ 这种 货 也 发 ？ ！】
以下代码将数据从df形式转换为固定格式"""
# 定义保存文件的路径
output_file = '1.txt'

# 打开文件以写入数据
with open(output_file, 'w', encoding='utf-8') as wf:
    # 遍历 DataFrame 的行
    for index, row in df.iterrows():
        # 从 DataFrame 中提取文本和标签
        text = row['评论']
        label = row['品质']

        # 将数据按照指定格式写入文件
        wf.write("__label__{} {}\n".format(
            label,
            text.strip().replace("\n", " __newline__ "),
        ))





labels=df["品质"]
#创建FastText实例
ftc = FastTextClassifier(
    train_data_fn="1.txt",
    batch_size=124,
    labels=[0,1,2],
    kwargs_train_supervised={
        "epoch": 1,

        "lr": 0.1,
        "wordNgrams": 3,
        #存储特征向量的哈希表大小。较大的值通常可以提高模型的性能，但会增加内存消耗。
        "bucket": 2000,
        "dim": 100,
        "loss": "softmax",
    },
)

#评价这种参数下的FT模型评分多少
pyx = cleanlab.count.estimate_cv_predicted_probabilities(
    X=np.arange(len(labels)),
    labels=labels,
    clf=ftc,
    cv_n_folds=3,
    seed=11,
)
#0.49079754601226994
#用于评估分类器性能
print(accuracy_score(labels, np.argmax(pyx, axis=1)))
