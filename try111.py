# coding:utf-8
import pickle

with open(r"D:\Pythonnnn\弱监督的农业社会化销售服务评价\4句法依存\总情感词典pos.txt","rb") as f:
    posdict = pickle.load(f)
print(posdict)
# with open(r"D:\Pythonnnn\弱监督的农业社会化销售服务评价\4句法依存\总情感词典neg.txt","rb") as f:
#     negdict = pickle.load(f)