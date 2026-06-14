"""
cleanlab的寻找错误标签代码只兼容sklearn，但是Fasttext不在skL中
skift库对FT模型进行了封装，使其使用方法和sk中的模型一样
"""

import pandas as pd
from cleanlab.models.fasttext import FastTextClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from cleanlab.classification import CleanLearning


if __name__ == '__main__':

    # df=pd.read_csv("D:\Pythonnnn\弱监督的农业社会化销售服务评价\弱监督\伪标签数据集.csv")
    df = pd.read_csv("D:\Pythonnnn\弱监督的农业社会化销售服务评价\弱监督\my_model训练数据\初始.csv")

    #词嵌入。train_texts是向量列表。SVM模型需要
    # vectorizer = TfidfVectorizer()
    # train_texts = vectorizer.fit_transform(df['评论'])
    # print(train_texts)
    #(2, 84)	0.7703915423424527
    # (2, 182)	0.637571071714532
    # (3, 132)	0.6486600276080734



    # from sklearn import svm
    # clf = svm.SVC(probability=True)

    from skift import FirstColFtClassifier
    clf = FirstColFtClassifier(lr=0.3, epoch=1)

    print("模型加载完毕，下面才是与cleanlab有关的代码")

    #设置交叉验证折数，5或10有着更好的效果
    cv_n_folds = 5  # for efficiency; values like 5 or 10 will generally work better
    #使用cleanlab库封装模型
    cl = CleanLearning(clf, cv_n_folds=cv_n_folds)
    #注意：cleanlab中的5折交叉验证是在训练集中的，从始至终，测试集数据就没有放入cleanlab里

    #这是SVM的代码。用的词向量
    # label_issues = cl.find_label_issues(X=train_texts, labels=df["品质"])
    #FT代码，使用双层括号的字词
    label_issues = cl.find_label_issues(X=df[['评论']], labels=df["品质"])

    # label_issues.to_excel("label_issues.xlsx")
    #label_quality是一个【0，1】的数，分数越低，越可能是错误标签

    #筛选出cleanlab识别为错误标签的样本
    identified_issues = label_issues[label_issues["is_label_issue"] == True]
    print(identified_issues)
    # identified_issues.to_excel("identified_issues.xlsx")

    #进行一个由第到高的排序，输出前n个的下标
    N=20
    lowest_quality_labels = label_issues["label_quality"].argsort()[:N].to_numpy()
    print(lowest_quality_labels)

    high_quality_labels = label_issues["label_quality"].argsort()[-N:].to_numpy()
    #对置信度最低的n个标签进行展示
    def print_as_df(index):
        return pd.DataFrame(
            {
                "text": df['评论'],
                "given_label": df['品质'],
                "predicted_label": label_issues["predicted_label"],
                #以此为准，这才是正确标签
                "label_quality": label_issues["label_quality"],
                "is_label_issue":label_issues["is_label_issue"]
            },
        ).iloc[index]
    result=print_as_df(lowest_quality_labels)
    # result.to_excel("result_low.xlsx")

    result = print_as_df(high_quality_labels)
    # result.to_excel("result_high.xlsx")


