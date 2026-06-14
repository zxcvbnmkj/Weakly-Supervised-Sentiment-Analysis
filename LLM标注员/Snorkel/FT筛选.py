#encoding:utf-8
import pandas as pd
from cleanlab.classification import CleanLearning
from skift import FirstColFtClassifier

if __name__ == '__main__':
    #测试代码，先使用量少的数据集进行测试，没问题再换成伪标签数据集
    df = pd.read_csv("D:\Pythonnnn\弱监督的农业社会化销售服务评价\弱监督\my_model训练数据\初始.csv")

    clf = FirstColFtClassifier(lr=0.3, epoch=1)

    print("模型加载完毕，下面才是与cleanlab有关的代码")

    cv_n_folds = 5  # for efficiency; values like 5 or 10 will generally work better
    cl = CleanLearning(clf, cv_n_folds=cv_n_folds)

    weidus = ['品质']
    # weidus = ['品质', '味道', '价格', '分量', '外观', '物流', '客服', '粗粒度']
    for weidu in weidus:
        label_issues = cl.find_label_issues(X=df[['评论']], labels=df[weidu])
        #问题样本
        identified_issues = label_issues[label_issues["is_label_issue"] == True]
        identified_issues = identified_issues["label_quality"].argsort().to_numpy()
        #展示问题样本
        def print_as_df(index):
            return pd.DataFrame(
                {
                    "text": df['评论'],
                    "given_label": df[weidu],
                    "predicted_label": label_issues["predicted_label"],
                    #以此为准，这才是正确标签
                    "label_quality": label_issues["label_quality"],
                    "is_label_issue":label_issues["is_label_issue"]
                },
            ).iloc[index]
        result=print_as_df(identified_issues)
        # result.to_excel(f"result_{weidu}.xlsx")



