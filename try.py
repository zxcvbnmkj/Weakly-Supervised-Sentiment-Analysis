import pickle

import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from torch import nn, tensor

# df=pd.read_excel(r"D:\Pythonnnn\弱监督的农业社会化销售服务评价\data\红心柚\领域词典.xlsx")

#按行遍历df
# #index表示当前是那一行，从0开始。row表示这一行的所有内容。
# #可以使用row["领域正面情感种子词 "]这样来输出一个格子的内容
# # for index, row in df.iterrows():
# #     # print(index,row)
# #     print(row["领域正面情感种子词"])
# """
# 0 评价维度                                               品质
# 领域属性词             品质、水分、色泽、外皮、颜色、果肉、质地、光泽、果皮、纹路、外皮、营养
# 领域正面情感种子词    漂亮、新鲜、放心、保证、薄、不错、细腻、丰富、完美、干净、柔软、光滑、厚实、紧实
# 领域负面情感种子词                     发霉、坏果、烂果、干瘪、硬硬、遗憾、超厚、腐烂
# """
# # num_list=[2,3,0]
# # if 1 not in num_list:
# #     print("aaa")
#
# from keras.src.preprocessing.text import Tokenizer
# from keras.src.utils import pad_sequences
# from torch import nn
# group = ['品质', '味道', '价格', '分量', '外观', '物流', '客服', '粗粒度']
# tokenizer = Tokenizer(num_words=20,oov_token='<UNK>')
# tokenizer.fit_on_texts("".join(group))
# label_ids = tokenizer.texts_to_sequences(group)
# print(tensor(label_ids))
# label_embedding = nn.Embedding(len(tokenizer.index_word)+1, 300)
#
# embedded_label = label_embedding(tensor(label_ids))
# print(embedded_label)
# with open("D:\Pythonnnn\弱监督的农业社会化销售服务评价\深度学习\dl_data\label_embbed.pkl","wb") as f:
#     pickle.dump(embedded_label,f)

# def chinese_word_cut(mytext):
#     list=mytext.split(" ")
#     return "".join(list)
# df=pd.read_csv(r"D:\Pythonnnn\弱监督的农业社会化销售服务评价\弱监督\test_ture\完整测试集.csv")
#
#
# df["评论"] = df.评论.apply(chinese_word_cut)
#
# df["评论"].to_csv("paddleNLP_test.txt",index=False)
#
# a=[]
# # a=["zhanwei"]
# b=[1,2,3]
# a.extend(b)
# print(a)
# # del a[0]
# # print(a)


def open_dict(Dict='hahah',path=r'D:\Pythonnnn\1resource\公共词典\情感词典\知网Hownet\修饰词典'):
    path = path + '\%s.txt' %Dict
    dictionary = open(path, 'r', encoding='utf-8-sig',errors='ignore')#encoding='utf-8-sig',检查是否有文件头，并去掉
    dict = []
    for word in dictionary:
        word=word.strip('\n')
        word=word.strip(' ')
        dict.append(word)
    return dict


open_dict()