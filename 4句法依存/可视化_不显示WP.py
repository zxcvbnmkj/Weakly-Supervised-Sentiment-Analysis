# -*- coding: utf-8 -*-
#LTP做句法依存时，不过滤标点符号，效果最好

from pyltp import Postagger, Parser, Segmentor

postagger = Postagger(r"D:\Pythonnnn\1resource\pyltp_model\pos.model")
parser = Parser(r"D:\Pythonnnn\1resource\pyltp_model\parser.model")

sent = '柚子味道非常好，很香，果肉饱满，水分足，没有坏果，商家发货速度十分快，包装严实完整，就是这边送货态度恶劣，客服也负责任处理解决。'

segmentor = Segmentor(r"D:\Pythonnnn\1resource\pyltp_model\cws.model",
                      lexicon_path=r"user_dict.txt")


words = segmentor.segment(sent)
print(" ".join(words))
#
# # 词性标注
# postags = postagger.postag(words)
#
# # 依存句法分析
# arcs = parser.parse(words, postags)
#
#
# rely_id = [arc[0] for arc in arcs]  # 提取依存父节点id
# relation = [arc[1] for arc in arcs]  # 提取依存关系
# heads = ['Root' if id == 0 else words[id-1] for id in rely_id]  # 匹配依存父节点词语
#
# for i in range(len(words)):
#     print(relation[i] + '(' + words[i] + ', ' + heads[i] + ')')
#
# from graphviz import Digraph
#
# g = Digraph('测试图片')
#
# g.node(name='Root')
# for word in words:
#     if word!="，" and word!="。":
#         g.node(name=word,fontname='Fangsong')
#
# #edge是用来画边的。g.edge的参数12是箭头两侧节点，参数3是箭头上的标注
# for i in range(len(words)):
#     if relation[i] not in ['HED','WP']:
#         g.edge(words[i], heads[i], label=relation[i])
#     elif relation[i]=="WP":
#         print("跳过")
#     else:
#         #当关系是HED时
#         if heads[i] == 'Root':
#             g.edge(words[i], 'Root', label=relation[i])
#         else:
#             g.edge(heads[i], 'Root', label=relation[i])
#
# g.view()
#
