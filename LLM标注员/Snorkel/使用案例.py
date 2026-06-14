# -*- coding: utf-8 -*-
# Snorkel
#详细教程：#resource: https://www.snorkel.org/use-cases/01-spam-tutorial#3-writing-more-labeling-functions
#几种标注方式：
#1，关键字搜索：在句子中查找特定单词
#2，模式匹配：寻找特定的语法模式
#3，第三方模型：使用预先训练的模型（通常是与手头任务不同的模型）
#4，远程监督：使用外部知识库
#5，众包标签：将每个众包工作者视为一个黑盒函数，将标签分配给数据子集
import pandas as pd
from snorkel.labeling import LabelingFunction, labeling_function, PandasLFApplier
from snorkel.labeling.model import LabelModel
from snorkel.preprocess import preprocessor
from textblob import TextBlob

df = pd.read_csv(r"D:\Pythonnnn\1resource\kaggle_datasets\abcnews-date-text_sample.csv")
df = df.rename(columns={'headline_text': 'text'})
df['text'] = df['text'].astype(str)


#正式使用snorkel
POSITIVE = 1
NEGATIVE = 0
#弃权
ABSTAIN = -1

#查找传入的文本中是否具有词典中定义的积极/消极关键词。有则返回对应标签，没有返回“弃权”
def keyword_lookup(x, keywords, label):
    if any(word in x.text.lower() for word in keywords):
        return label
    return ABSTAIN



#创建标签函数LabelingFunction
def make_keyword_lf(keywords, label):
    return LabelingFunction(
        name=f"keyword_{keywords[0]}",
        f=keyword_lookup,
        resources=dict(keywords=keywords, label=label))


#积极、消极关键字定义
#这两个列表可以进一步扩展
keyword_positive = make_keyword_lf(keywords=['boosts', 'great', 'develops', 'promising', 'ambitious', 'delighted', 'record', 'win', 'breakthrough', 'recover', 'achievement', 'peace', 'party', 'hope', 'flourish', 'respect', 'partnership', 'champion', 'positive', 'happy', 'bright', 'confident', 'encouraged', 'perfect', 'complete', 'assured' ],label=POSITIVE)
keyword_negative = make_keyword_lf(keywords=['war','solidiers', 'turmoil', 'injur','trouble', 'aggressive', 'killed', 'coup', 'evasion', 'strike', 'troops', 'dismisses', 'attacks', 'defeat', 'damage', 'dishonest', 'dead', 'fear', 'foul', 'fails', 'hostile', 'cuts', 'accusations', 'victims',  'death', 'unrest', 'fraud', 'dispute', 'destruction', 'battle', 'unhappy', 'bad', 'alarming', 'angry', 'anxious', 'dirty', 'pain', 'poison', 'unfair', 'unhealthy'],label=NEGATIVE)

# print(keyword_positive)
# print(keyword_negative)
#LabelingFunction keyword_boosts, Preprocessors: []
# LabelingFunction keyword_war, Preprocessors: []



#使用预训练分类器textlob 建立预处理函数来确定极性和主观性
#定义预处理函数
#memoize=True启用函数的缓存功能，即对于相同的输入参数，函数的输出结果将会被缓存起来，下次再使用相同的参数调用函数时，将直接返回缓存的结果，而不会重新计算。
@preprocessor(memoize=True)
def textblob_sentiment(x):
    scores = TextBlob(x.text)
    #情感极性
    x.polarity = scores.sentiment.polarity
    #主客观程度。主观性通常在0到1的范围内
    x.subjectivity = scores.sentiment.subjectivity
    return x

#标签函数1：先经过预处理函数，再经过标签函数1。TextBlob的预测极性大于0.6判断为积极否则弃权
#pre等于一个列表，说明可以有多个预处理函数
#find polarity
@labeling_function(pre=[textblob_sentiment])
def textblob_polarity(x):
    return POSITIVE if x.polarity > 0.6 else ABSTAIN

#标签函数2
@labeling_function(pre=[textblob_sentiment])
def textblob_subjectivity(x):
    return POSITIVE if x.subjectivity >= 0.5 else ABSTAIN



#组合所有标签功能。一共4个标签函数
lfs = [keyword_positive, keyword_negative, textblob_polarity, textblob_subjectivity ]

#创建实例
applier = PandasLFApplier(lfs=lfs)
#在数据帧上应用标签函数列表
L_snorkel = applier.apply(df=df)
#创建标签函数实例。cardinality表示是几分类（弃权不算）
label_model = LabelModel(cardinality=2, verbose=True)
label_model.fit(L_snorkel)
#为样本生成最终标签
df["label"] = label_model.predict(L=L_snorkel)


#标签不是0、1的话就去掉
df= df.loc[df.label.isin([0,1]), :]
#find the label counts  标记成功的标签
print(df['label'].value_counts())

print(df["label"])