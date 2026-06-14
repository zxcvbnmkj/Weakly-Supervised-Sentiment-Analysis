# -*- coding: utf-8 -*-
# from paddlenlp import Taskflow
# schema = ["观点词", "情感倾向[正向,负向,未提及]"]
# aspects = ["房间", "位置", "价格"]
# senta = Taskflow("sentiment_analysis", model="uie-senta-mini", schema=schema, aspects=aspects,home_path=r"D:\Pythonnnn\1resource\paddleNLP")
# # senta = Taskflow("sentiment_analysis", model="uie-senta-base", schema=schema, aspects=aspects,home_path=r"D:\Pythonnnn\1resource\paddleNLP")
# print(senta("这家店的房间很大，店家服务也很热情，就是价格有点贵"))
import json

import pandas as pd

#
# with open('D:\Pythonnnn\弱监督的农业社会化销售服务评价\LLM标注员\sentiment_analysis2.json', 'r', encoding='utf-8') as f:
#     con = json.load(f)
#     print(con)
# for i in range(0, len(con)):
#     print(con[i]['情感倾向[正向,负向,矛盾]']['text'])



#粗粒度中有3个输出是空

a=[]
b=[]
c=[]
d=[]
e=[]
fff=[]
g=[]
h=[]
studentsList = []
with open('D:\Pythonnnn\弱监督的农业社会化销售服务评价\LLM标注员\sentiment_analysis.json', 'r', encoding='utf-8') as f:
    for jsonObj in f:
        studentDict = json.loads(jsonObj)
        studentsList.append(studentDict)
#为什么列表不能叫f。只有命名为f时会出错。换个名字就好
for student in studentsList:
    a.append(student["评价维度"][0]["relations"]["情感倾向[正向,负向,未提及]"][0]["text"])
    b.append(student["评价维度"][1]["relations"]["情感倾向[正向,负向,未提及]"][0]["text"])
    c.append(student["评价维度"][2]["relations"]["情感倾向[正向,负向,未提及]"][0]["text"])
    d.append(student["评价维度"][3]["relations"]["情感倾向[正向,负向,未提及]"][0]["text"])
    e.append(student["评价维度"][4]["relations"]["情感倾向[正向,负向,未提及]"][0]["text"])
    fff.append(student["评价维度"][0]["relations"]["情感倾向[正向,负向,未提及]"][0]["text"])
    g.append(student["评价维度"][6]["relations"]["情感倾向[正向,负向,未提及]"][0]["text"])


studentsList = []
with open('D:\Pythonnnn\弱监督的农业社会化销售服务评价\LLM标注员\sentiment_analysis2.json', 'r', encoding='utf-8') as f:
    for jsonObj in f:
        studentDict = json.loads(jsonObj)
        studentsList.append(studentDict)
for student in studentsList:
    if student["情感倾向[正向,负向,矛盾]"][0]["text"]=="正向":
        h.append(2)
    elif student["情感倾向[正向,负向,矛盾]"][0]["text"]=="矛盾":
        h.append(1)
    elif student["情感倾向[正向,负向,矛盾]"][0]["text"]=="负向":
        h.append(0)
    else:
        print("!!!!!")
df=pd.DataFrame(data={"品质":a,'味道':b,'价格':c,'分量':d,'外观':e,'物流':fff,'客服':g,'粗粒度':h})
#加上索引，为了让列数与测试集列数保持一致
df.to_csv("paddleNLP预测结果.csv")
