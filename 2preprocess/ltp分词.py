import numpy as np
from pyltp import Segmentor
from zhconv import zhconv
import pandas as pd
import re
"""
当前这版是，用于句法依存的，无停用词，不去除标点符号的，预处理
"""

# df=pd.read_csv(r"../data/红心柚/红心柚.csv")
# df=pd.read_csv(r"D:\Pythonnnn\弱监督实验\原始数据\红心柚\新增差评.csv")
df=pd.read_excel("../农资_化肥数据集/农资_化肥.xlsx")
print("初始长度",len(df))


#抽取出需要的那三列
# group=['评论','评分','评论时间']
group=['评论']

df=df[group]


#去重、去空
df=df.drop_duplicates(subset="评论",inplace=False)
#只要有一个空值就删除，axis=0表示删除行
df.dropna(axis=0, how='any', subset=None, inplace=True)

#去除某种类型的评论
# 找到所有包含特定值的行
rows_to_delete = df[df['评论'] == '此用户未填写评价内容']
# 删除包含特定值的行
df = df.drop(rows_to_delete.index)


rows_to_delete = df[df['评论'] == "此用户未及时填写评价内容"]
# 删除包含特定值的行
df = df.drop(rows_to_delete.index)


segmentor = Segmentor(r"D:\Pythonnnn\1resource\pyltp_model\cws.model",
                      lexicon_path=r"user_dict.txt")


#加载自定义停用词列表。该词典有利于主题建模时去除无关的频繁字
# stopword_file = open(r"./停用词.txt",encoding ='utf-8')
# stop_list = []
# for line in stopword_file:
#     #sub的作用是替换，把参数1替换为参数2。参数3是文本
#     line = re.sub(u'\\n|\\r', '', line)
#     stop_list.append(line)


#读取哈工大停用词表
# with open("D:\Pythonnnn\基于词典的细粒度情感分析\数据文件\停用词典\哈工大停用词表.txt", encoding="utf-8") as hgd_stopwordsfile:
#     str = hgd_stopwordsfile.read()
# hgd_stopwords_list = str.split("\n")

#使用集合去重再合并
# stop_list = list(set(stop_list+hgd_stopwords_list))


def chinese_word_cut(mytext):

    #繁体转换为简体
    mytext=zhconv.convert(mytext, 'zh-hans')

    #英文大写转换为小写
    mytext=mytext.lower()

    #过滤掉特殊字符、标点
    #只保留数字、小写英文字母、汉字
    # mytext = re.sub("([^\u0030-\u0039\u0061-\u007a\u4e00-\u9fa5])", '', mytext)
    #只保留小写英文、汉字。过滤掉数字。也会过滤掉表情、颜文字
    # mytext = re.sub("([^\u0061-\u007a\u4e00-\u9fa5])", '', mytext)
    #保留标点
    mytext = re.sub("([^\u0061-\u007a\u4e00-\u9fa5\u3000-\u303F\u4e00-\u9fa5\uff00-\uffef])", '', mytext)


    word_list = []

    seg_list = segmentor.segment(mytext)

    # while "不" in seg_list:
    #     # seg_list是一个列表，通过值求索引
    #     index = seg_list.index("不")
    #     if index == len(seg_list) - 1:
    #         break
    #     # 因为是[a,b)所以其实是index和index+1被拼接了
    #     seg_list[index: index + 2] = ["".join(seg_list[index: index + 2])]  # 列表切片赋值的酷炫写法
    # while "别" in seg_list:
    #     # seg_list是一个列表，通过值求索引
    #     index = seg_list.index("别")
    #     if index == len(seg_list) - 1:
    #         break
    #     # 因为是[a,b)所以其实是index和index+1被拼接了
    #     seg_list[index: index + 2] = ["".join(seg_list[index: index + 2])]  # 列表切片赋值的酷炫写法
    # while "没有" in seg_list:
    #     # seg_list是一个列表，通过值求索引
    #     index = seg_list.index("没有")
    #     if index == len(seg_list) - 1:
    #         break
    #     # 因为是[a,b)所以其实是index和index+1被拼接了
    #     seg_list[index: index + 2] = ["".join(seg_list[index: index + 2])]



    # while "很" in seg_list:
    #     # seg_list是一个列表，通过值求索引
    #     index = seg_list.index("很")
    #     if index == len(seg_list) - 1:
    #         break
    #     # 因为是[a,b)所以其实是index和index+1被拼接了
    #     seg_list[index: index + 2] = ["".join(seg_list[index: index + 2])]
    # while "特别" in seg_list:
    #     # seg_list是一个列表，通过值求索引
    #     index = seg_list.index("特别")
    #     if index == len(seg_list) - 1:
    #         break
    #     # 因为是[a,b)所以其实是index和index+1被拼接了
    #     seg_list[index: index + 2] = ["".join(seg_list[index: index + 2])]
    # while "非常" in seg_list:
    #     # seg_list是一个列表，通过值求索引
    #     index = seg_list.index("非常")
    #     if index == len(seg_list) - 1:
    #         break
    #     # 因为是[a,b)所以其实是index和index+1被拼接了
    #     seg_list[index: index + 2] = ["".join(seg_list[index: index + 2])]



    # for seg_word in seg_list:
    #     #find是一个bool元素，表示“是否在停用词中找到当前词语”。等于1表示当前词是停用词里面的，应该被删除。
    #     find = False
    #     for stop_word in stop_list:
    #         if stop_word == seg_word:
    #                 find = True
    #                 break
    #
    #     if find == False:
    #         word_list.append(seg_word)


    #把列表中的词转换为字符串，并使用空格作为分隔符
    # return (" ").join(word_list)
    return (" ").join(seg_list)

    #即使我们需要取出的是['这是','一句','话','。']这样的列表但
    #千万不可写成这样，否则excel中存的一行是['这是','一句','话','。']
    #看起来是列表形式，但取出来后这其实是一串字符串，而不是列表。需要把那种形式的字符串转换为列表更麻烦，还不如直接存为空格分隔的字符串
    #然后再split(" ")
    #return word_list


def connect(str):
    str=str.replace(" ",'')
    return str.replace('\n', '')

"""
第一列“过滤后”只是原始文本去处理停用词。他就是把分词这一列的空格取消了
因为textRank需要未经过分词的文本数据集
"""
df2 = pd.DataFrame(columns=['分词'])
# df2["评分"]=df["评分"]
# df2["评论时间"]=df["评论时间"]
#****df后面的是列名，需要改的
#新加一列：content_cutted
df2["分词"] = df.评论.apply(chinese_word_cut)
#要用到它的时候再去掉空格
# df["过滤后"] = df2.分词.apply(connect)
# df2["评价"]=df["评价"]
# df2["评分"]=df["评分"]

#只要有任意空值就删除那一行(any)
#只有所有列都为空，才删除那行(all)

#之前一直删不了空行是因为，空行中存在着''，需要把它替换成np.nan
df2.replace("",np.nan,inplace=True)
df2.dropna(axis=0, how='any', subset=None, inplace=True)
print("处理后长度",len(df2))
# df2.to_csv(r"../data/红心柚/分词后_无停用词_空格.csv",index=False)
df2.to_csv(r"../农资_化肥数据集/预处理后.csv",index=False)

#主题建模把表示否定的词合并了，且停用词不一样，把特别、非常等程度词去掉