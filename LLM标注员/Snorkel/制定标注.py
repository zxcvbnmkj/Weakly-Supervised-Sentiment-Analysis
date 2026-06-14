#coding:utf-8
import csv
import jieba
import pandas as pd
from snorkel.labeling import LabelingFunction, labeling_function, PandasLFApplier
from snorkel.labeling.model import LabelModel
from snorkel.preprocess import preprocessor
from pyltp import Parser
from pyltp import Postagger


#正式使用snorkel
POSITIVE = 1
NEGATIVE = 0
#未提到
UNMENTIONED=2
#弃权
ABSTAIN = -1



def keyword_lookup(x, keywords, label,t_list):
    #使用jieba进行分词，避免和之前的ltp分词结果一样
    jieba_words = jieba.lcut(x.评论)
    # 在分词的基础上，标注词性
    postags = postagger.postag(jieba_words)
    postags = ' '.join(postags).split()
    # 在分词、词性的基础上，进行句法依存分析
    arcs = parser.parse(jieba_words, postags)


    head = []
    relation = []

    for arc in arcs:
        head.append(arc[0])
        relation.append(arc[1])

    i = 0
    for word in jieba_words:
        if word in t_list and relation[i] in ['SBV', 'VOB', 'FOB'] and jieba_words[head[i] - 1] in keywords:
            return label

        if word in keywords and relation[i] in ['ATT', 'CMP'] and jieba_words[head[i] - 1] in t_list:
            return label

        if word in t_list and relation[i] in ['COO'] and jieba_words[head[i] - 1] in keywords:
            return label

        if word in keywords and relation[i] in ['COO'] and jieba_words[head[i] - 1] in t_list:
            return label

        i += 1

    i = relation.index("HED")
    if jieba_words[i] not in t_list and postags[i] in ['n', 'v', 'nz']:
        for j in range(len(jieba_words)):
            if jieba_words[head[j] - 1] == jieba_words[i] and relation[j] in ['VOB', 'CMP'] and postags[i] != 'v':
                return label

    """潜在方面意见（句法规则4），用核心词识别未登录情感词"""
    if jieba_words[i] not in keywords and postags[i] in ['a', 'u', 'd']:
        for j in range(len(jieba_words)):
            if jieba_words[head[j] - 1] == jieba_words[i] and relation[j] in ['SBV', 'ATT'] and postags[i] in ['v', 'n', 'j', 'nz']:
                return label

    """潜在方面意见（句法规则5），识别核心词周围的情感表达"""

    for j in range(len(jieba_words)):
        # 根据候选属性词确定情感词
        if jieba_words[head[j] - 1] == jieba_words[i] and relation[j] in ['SBV', 'VOB'] and postags[i] in ['v', 'n', 'j',
                                                                                               'nz'] and len(
            jieba_words[j]) > 1:


            for k in range(len(jieba_words)):
                # 属性词与情感词不能一样，所以and words[k]!=words[j]
                if jieba_words[head[k] - 1] == jieba_words[i] and jieba_words[k] in keywords and jieba_words[k] != jieba_words[j]:
                    return label

        if jieba_words[head[j] - 1] == jieba_words[i] and relation[j] == 'CMP' and postags[i] in ['a', 'n'] and len(jieba_words[j]) > 1:

            for k in range(len(jieba_words)):
                if jieba_words[head[k] - 1] == jieba_words[i] and jieba_words[k] in t_list and jieba_words[k] != jieba_words[j]:
                    return label
    return ABSTAIN


def make_keyword_lf(keywords, label,t_list):
    return LabelingFunction(
        name=f"keyword_{keywords[0]}",
        f=keyword_lookup,
        resources=dict(keywords=keywords, label=label,t_list=t_list))






@labeling_function()
def FastText_Label(x):
    if x.FT==0:
        return NEGATIVE
    elif x.FT==1:
        return POSITIVE
    elif x.FT==2:
        return UNMENTIONED
    else:
        return ABSTAIN
@labeling_function()
def Label_OF_Model(x):
    if x.MODEL==0:
        return NEGATIVE
    elif x.MODEL==1:
        return POSITIVE
    elif x.MODEL==2:
        return UNMENTIONED
    else:
        return ABSTAIN

if __name__ == '__main__':

    list_positive=[]
    list_negative = []

    # 处理口语词.csv
    with open('口语词.csv', mode='r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row['情感极性'] == '积极':
                list_positive.append(row['词语'])
            elif row['情感极性'] == '消极':
                list_negative.append(row['词语'])
    # 处理网络流行语.csv
    with open('网络流行语.csv', mode='r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row['情感极性'] == '积极':
                list_positive.append(row['词语'])
            elif row['情感极性'] == '消极':
                list_negative.append(row['词语'])

    postagger = Postagger(r"D:\Pythonnnn\1resource\pyltp_model\pos.model")
    parser = Parser(r"D:\Pythonnnn\1resource\pyltp_model\parser.model")

    a = "品质、果肉、质地、营养、柚子、水果、商品"
    b = "味道、口感、尝、吃起来、闻、甜度、吃、肉、水份、水分、酸度"
    c = "价格、性价比、价钱、定价、成本、售价、价位"
    d = "分量、个头、大小、斤、箱、袋、个数"
    e = "包装、质量、配色、卖相、包装袋、包装盒、外包装、外盒、外观、看起来、看、摸起来、色泽、外皮、颜色、光泽、果皮、纹路、外皮、皮"
    f = "物流、发货、服务态度、运送、送货、配送、取件、收货、送货上门"
    g = "客服、态度、商家、卖家、店家、体验、处理、下单、厂家、补救、客服人员、感受、网店、服务、售后"
    alist = a.split("、")
    blist = b.split("、")
    clist = c.split("、")
    dlist = d.split("、")
    elist = e.split("、")
    flist = f.split("、")
    glist = g.split("、")
    t_list=[alist,blist,clist,dlist,elist,flist,glist]

    # 创建标签函数实例。cardinality表示是几分类（弃权不算）
    label_model = LabelModel(cardinality=3, verbose=True)

    weidus = ['品质', '味道', '价格', '分量', '外观', '物流', '客服', '粗粒度']
    for count in range(len(weidus)):
        keyword_positive = make_keyword_lf(
            keywords=list_positive,
            label=POSITIVE, t_list=t_list[count])

        keyword_negative = make_keyword_lf(
            keywords=list_negative, label=NEGATIVE, t_list=t_list[count])

        # 标签函数组合
        lfs = [keyword_positive, keyword_negative, FastText_Label, Label_OF_Model]

        # 创建实例
        applier = PandasLFApplier(lfs=lfs)

        df = pd.read_csv(f"./{weidus[count]}维度.csv")
        # 在数据帧上应用标签函数列表
        L_snorkel = applier.apply(df=df)
        label_model.fit(L_snorkel)
        # 为样本生成最终标签
        df["Final_Label"] = label_model.predict(L=L_snorkel)

        df.to_csv(f"最终预测标签_{weidus[count]}.csv",index=False)
        # 标签不是0、1的话就去掉
        # df = df.loc[df.label.isin([0, 1,2]), :]
        # find the label counts  标记成功的标签
        # print(df['label'].value_counts())
        # print(df["Label"])