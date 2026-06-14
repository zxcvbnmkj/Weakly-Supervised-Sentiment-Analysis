#同义词中只保留相似度大于0.6的

import pandas as pd
import synonyms
#选中几个格子。参数1的[]中写行的编号（从1开始），参数2的[]中写列名
#下面语句会打印主题这一列的前4个格子
# print(df.loc[[0,1,2,3],['主题']])


df=pd.read_excel("./主题模型的领域词典.xlsx")
def add_attr_words(texts):
    # print(texts)
    # print(type(texts))
    # #取出属性词
    # for i in range(7):
    synonyms_words_i=[]
    #得到的变量类型是DataFrame，而不是列表
    # sub_df=df.loc[[i]]
    # print(sub_df)
    #把选中的df中的一格，放入列表中
    #['酒店、家庭、孩子、一家、公寓、房间、']
    #如果直接写sub_df.values，则会有2层[]
    #[['酒店、家庭、孩子、一家、公寓、房间、']]
    #sub_df['领域核心属性词'].values得到一个numpy类型的返回值，需要把它变成list。必须加那个括号
    # cell_list=(sub_df['领域核心属性词'].values).tolist()
    # print(cell_list)

    #把里面的词取出来，按、拆分，再存入新列表中
    # list=cell_list[0].split("、")
    words_list = texts.split("、")
    # print(list)
    for j in words_list:
        #print(j, synonyms.nearby(j))
        #近义词方法返回一个元组：（【近义词列表】，【得分列表】）
        sys_turple=synonyms.nearby(j)
        y=0
        for x in sys_turple[1]:
            if x>=0.6:
                synonyms_words_i.append(sys_turple[0][y])
            y+=1
    # print(synonyms_words_i)

    #把核心属性词与扩充属性词合并，去重（扩充属性词中也有重复的）
    synonyms_words_i = list(set(words_list + synonyms_words_i))
    return " ".join(synonyms_words_i)


df["扩充后属性词典"] = df.attr_words.apply(add_attr_words)
df.to_excel("属性词扩充后.xlsx")





