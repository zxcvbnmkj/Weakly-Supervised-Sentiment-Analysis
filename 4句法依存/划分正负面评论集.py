import jieba
import pandas as pd
from jieba.analyse import textrank, extract_tags

df=pd.read_csv(r"D:\Pythonnnn\弱监督的农业社会化销售服务评价\2预处理\分词后.csv")
# df2=df[df.评分==4]
df2=df[df.评分<=3]
pos_sens=df2["分词"].values.tolist()
#仅保留汉字
# for mytext in pos_sens:
#     mytext = re.sub("([^\u4e00-\u9fa5])", '', mytext)
    # if mytext=='':
    #     pos_sens.remove('')

pos_str=" ".join(pos_sens)
# print(pos_str)

# df2=df[df.value==1]
# neg_sens=df2["comments"].values.tolist()
# neg_str=" ".join(neg_sens)


#合并2个列表，content的长度是2
# content=[pos_sens,neg_sens]


print('使用text-rank')
# jieba.del_word("根本")
# jieba.load_userdict("频繁项挖掘dict.txt")
dic = textrank(pos_str, topK=50, withWeight=True,allowPOS=['a'])
print(dic)


# print('使用tf-idf')
# #加载自定义idf词频文件，而不使用jieba默认的idf文件
# jieba.analyse.set_idf_path(r"D:\Pythonnnn\基于词典的细粒度情感分析\3词典扩充\idf文件_带次数.txt")
tags = extract_tags(pos_str, topK=50,withWeight=True,allowPOS=('a'))
print(tags)
