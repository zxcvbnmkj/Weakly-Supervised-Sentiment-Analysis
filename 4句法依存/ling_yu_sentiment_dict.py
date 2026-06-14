# -*- coding:utf-8 -*-
import pandas as pd
from harvesttext import HarvestText
from pyltp import Postagger


pos_seeds="很足、赞、棒、爱吃、酸甜可口、物美价廉、物超所值、酸酸甜甜、凑合、刚好、送达、收到、回购、符合、好评、光顾、吻合、五星好评、信赖、送货上门、建议买、爱、可以、正宗、正货、漂亮、行、好、不错、新鲜、放心、保证、薄、细腻、丰富、高端、物美价廉、完美、干净、柔软、光滑、厚实、紧实、好吃、甜、酸甜、美味、适中、细腻、甜甜、浓郁、细嫩、鲜嫩、爽口、不酸、光滑、划算、便宜、实惠、合适、优惠、值得、沉甸甸、大、足、够、充足、均匀、匀称、适宜、完好、结实、讲究、干净、严实、精致、紧、仔细、漂亮、完美、快、快速、给力、不错、方便、快捷、完整、满意、及时、礼貌、热心、细致、周到、耐心、友善、亲切、良心"
neg_seeds="发白、厚、差点、干、涩口、老、酸、苦、坏、变质、扔了、退货、拖、久、垃圾、上当、烂、丢、发错、不符、退、取关、骗人、索赔、麻烦、不退、补偿、没收到、压扁、慢慢、谨慎、差评、发硬、发霉、坏果、烂果、干瘪、硬硬、遗憾、超厚、腐烂、酸酸、难吃、酸味、微苦、伤疤、酸苦、酸涩、苦涩、寡淡、巨苦、上涨、贵、不值、抬价、小贵、小、少、不足、一般、损坏、费劲、简陋、变形、破了、慢、差、没收到、糟糕、摔烂、恶劣、虚假宣传、欺骗、投诉"
pos_list=pos_seeds.split("、")
neg_list=neg_seeds.split("、")

dataset_text=pd.read_csv(r"D:\Pythonnnn\弱监督的农业社会化销售服务评价\data\红心柚\分词后_无停用词_空格.csv")
data_list=dataset_text["分词"].values
#句子被空格分隔不要紧，不影响的（HarvestSO-PMI要求的是["句子","句子"]的形式，但是有空格也可以）


postagger = Postagger(r"D:\Pythonnnn\1resource\pyltp_model\pos.model")
ht = HarvestText()

sent_dict = ht.build_sent_dict(data_list, min_times=3, pos_seeds=pos_list, neg_seeds=neg_list,
                                   scale="+-1")
print(sent_dict)
pos_words=[]
neg_words=[]

pos_words2=[]
neg_words2=[]
for key in sent_dict:
    if sent_dict[key]>0.2:
        pos_words.append(key)
    #改成0.05吧
    elif sent_dict[key]<-0.2:
        neg_words.append(key)
#所有的正负词
# print(pos_words)
# print(neg_words)

postags = postagger.postag(pos_words)
for i in range(len(pos_words)):
    if postags[i] in ['a','v']:
        pos_words2.append(pos_words[i])


# postags = postagger.postag(neg_words)
# for word,postag in zip(neg_words,postags):
#     if postag in ['a','v']:
#         neg_words2.append(word)
#经过属性过滤后的
# print(pos_words2)
# print(neg_words2)


#把种子词也添加上。两个列表可以用+
pos_words2=list(set(pos_words2+pos_list))
#负面词就不过滤词性了
neg_words2=list(set(neg_words+neg_list))

p="\n".join(pos_words2)
n="\n".join(neg_words2)
with open(r"D:\Pythonnnn\弱监督的农业社会化销售服务评价\4句法依存\积极情感词典\积极领域情感词.txt", 'w', encoding="utf-8") as file:
    file.write(p)
with open(r"D:\Pythonnnn\弱监督的农业社会化销售服务评价\4句法依存\消极情感词典\消极领域情感词.txt", 'w', encoding="utf-8") as file:
    file.write(n)






