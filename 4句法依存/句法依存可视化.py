# -*- coding: utf-8 -*-
#LTP做句法依存时，不过滤标点符号，效果最好
import os
import jieba
from pyltp import Postagger, Parser, Segmentor

postagger = Postagger(r"D:\Pythonnnn\1resource\pyltp_model\pos.model")
parser = Parser(r"D:\Pythonnnn\1resource\pyltp_model\parser.model")

sent = '柚子味道很好，很香，果肉饱满，水分足，没有坏果，商家发货速度很快，包装严实完整，就是这边送货态度恶劣，商家也负责任处理解决。'

segmentor = Segmentor(r"D:\Pythonnnn\1resource\pyltp_model\cws.model",
                      lexicon_path=r"user_dict.txt")

words = segmentor.segment(sent)

# words=sent.split(" ")
# words = segmentor.segment("外观 不 太 好看 ， 有 很多 黑 点 。 不 知道 是否 正常 。")

# 词性标注
postags = postagger.postag(words)

# 依存句法分析
arcs = parser.parse(words, postags)
# print(arcs)
# print(words)
# for i in arcs:
#     if i[1]=="WP":
#         arcs.remove(i)
# count=0
# for i in words:
#     if i=="，" or i=="。":
#         words.remove(i)
#         count+=1
# print(words)

rely_id = [arc[0] for arc in arcs]  # 提取依存父节点id
relation = [arc[1] for arc in arcs]  # 提取依存关系
heads = ['Root' if id == 0 else words[id-1] for id in rely_id]  # 匹配依存父节点词语

for i in range(len(words)):
    print(relation[i] + '(' + words[i] + ', ' + heads[i] + ')')

from graphviz import Digraph

g = Digraph('测试图片')

g.node(name='Root')
for word in words:
    g.node(name=word,fontname='Fangsong')

for i in range(len(words)):
    if relation[i] not in ['HED']:
        g.edge(words[i], heads[i], label=relation[i])
    else:
        if heads[i] == 'Root':
            g.edge(words[i], 'Root', label=relation[i])
        else:
            g.edge(heads[i], 'Root', label=relation[i])

g.view()

"""
arcs的值是：
[(2, 'ATT'), (4, 'SBV'), (4, 'ADV'), (0, 'HED'), (4, 'WP'), (7, 'ADV'), (4, 'COO'), (7, 'WP'), (10, 'SBV'), (7, 'COO'), (7, 'WP'), (13, 'SBV'), (7, 'COO'), (7, 'WP'), (7, 'COO'), (15, 'VOB'), (15, 'WP'), (19, 'SBV'), (20, 'ATT'), (21, 'SBV'), (28, 'ADV'), (21, 'WP'), (28, 'SBV'), (23, 'VOB'), (24, 'COO'), (23, 'WP'), (28, 'ADV'), (7, 'COO'), (31, 'ATT'), (31, 'ATT'), (32, 'SBV'), (28, 'VOB'), (28, 'WP'), (36, 'SBV'), (36, 'ADV'), (28, 'COO'), (36, 'VOB'), (36, 'COO'), (38, 'COO'), (4, 'WP')]

分词时分出了多少个词语，句法依存时就会有几个 子节点
如words长度是40，因此arcs长度也是40，arcs元组中的第一个是父节点id，第二个是关系类型。隐藏的信息是子节点编号

words:
['柚子', '味道', '很', '好', '，', '很', '香', '，', '果肉', '饱满', '，', '水分', '足', '，', '没有', '坏果', '，', '商家', '发货', '速度', '很快', '，', '包装', '严实', '完整', '，', '就', '是', '这边', '送货', '态度', '恶劣', '，', '商家', '也', '负', '责任', '处理', '解决', '。']
1           2     3
ps.编号0的是ROOT节点。子节点、父节点的词语编号都是从1开始。



最终结果（竖着念，就是原文）
ATT(柚子, 味道)
SBV(味道, 好)
ADV(很, 好)
HED(好, Root)【好-->Root】
WP(，, 好)
ADV(很, 香)
COO(香, 好)
WP(，, 香)
SBV(果肉, 饱满)
COO(饱满, 香)
WP(，, 香)
SBV(水分, 足)
COO(足, 香)
WP(，, 香)
COO(没有, 香)
VOB(坏果, 没有)
WP(，, 没有)
SBV(商家, 发货)
ATT(发货, 速度)
SBV(速度, 很快)
ADV(很快, 是)
WP(，, 很快)
SBV(包装, 是)
VOB(严实, 包装)
COO(完整, 严实)
WP(，, 包装)
ADV(就, 是)
COO(是, 香)
ATT(这边, 态度)
ATT(送货, 态度)
SBV(态度, 恶劣)
VOB(恶劣, 是)
WP(，, 是)
SBV(商家, 负)
ADV(也, 负)
COO(负, 是)
VOB(责任, 负)
COO(处理, 负)
COO(解决, 处理)
WP(。, 好)
"""