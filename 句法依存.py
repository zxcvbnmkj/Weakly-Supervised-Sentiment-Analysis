"""
1，LTP句法 依存可视化网站
http://ltp.ai/demo.html
2，LTP说明文档
http://www.ltp-cloud.com/intro#dp_how


1,在score函数里面，如果修饰词是非常、特别，则count+1.5分，稍微则加0.8分
2，最后子维度综合得分为0的，视为积极吧
3，判断某个词是否属于属性词典时，不要简单的看他是否直接在词典中。可以通过word2vec判断相似度是否大于0.8
"""

# -*- coding:utf-8 -*-
import pickle
from functools import reduce
import pandas as pd

#修饰词典之后必须加一个\，但是加在path后面会报错！！只能写在%s之前了
def open_dict(Dict=None,path=r'D:\Pythonnnn\1resource\公共词典\情感词典\知网Hownet\修饰词典'):
    path = path + '\%s.txt' %Dict
    dictionary = open(path, 'r', encoding='utf-8-sig',errors='ignore')#encoding='utf-8-sig',检查是否有文件头，并去掉
    dict = []
    for word in dictionary:
        word=word.strip('\n')
        word=word.strip(' ')
        dict.append(word)
    return dict

#载入总情感词典
with open(r"D:\Pythonnnn\弱监督的农业社会化销售服务评价\4句法依存\总情感词典pos.txt","rb") as f:
    posdict = pickle.load(f)
with open(r"D:\Pythonnnn\弱监督的农业社会化销售服务评价\4句法依存\总情感词典neg.txt","rb") as f:
    negdict = pickle.load(f)
#有机会把它加到领域词典中去
# p_add=["行","酸酸甜甜","爱吃","送货上门","建议买"]
# posdict.extend(p_add)
# print(posdict)

# n_add=["","","拖","久","发白","发硬","老","酸","厚","发错"]
# negdict.extend(n_add)
# print(len(posdict))
# print(len(negdict))
# 12282
# 15378


#六个修饰词典
#inverse逆转的
#1，否定词典
inversedict=open_dict(Dict='inversedict')


#以下来自于HowNet程度词典
#下面要加分
#极为正面的程度词
mostdict = open_dict(Dict='mostdict')
#超、非常
# overdict = open_dict(Dict='overdict')
#很正面的程度词
verydict= open_dict(Dict='verydict')
#比较正面的程度词
moredict = open_dict(Dict='moredict')

#从这开始减分
#略微正面的程度词
ishdict = open_dict(Dict='ishdict')

#insufficient不足的。（知网中是：5，欠）
# 负面程度词
insufficientdict = open_dict(Dict='insufficientdict')

#q_list是情感词典的全部(包括正/负面)
q_list=posdict+negdict
#全部修饰词典
x_list=inversedict+mostdict+verydict+moredict+ishdict+insufficientdict

# df_domain=pd.read_excel(r"D:\Pythonnnn\弱监督的农业社会化销售服务评价\data\红心柚\领域词典.xlsx")


#句法分析
from pyltp import Parser, Segmentor  # 导入库Parser
from pyltp import Postagger#导入Postagger库

#获取已经预处理的数据集，所以不用再分词了
postagger = Postagger(r"D:\Pythonnnn\1resource\pyltp_model\pos.model")
parser = Parser(r"D:\Pythonnnn\1resource\pyltp_model\parser.model")
# segmentor = Segmentor(r"D:\Pythonnnn\1resource\pyltp_model\cws.model",
#                           lexicon_path=r"./2preprocess/user_dict.txt")

def get_detail(sentence):

    # words=segmentor.segment(sentence)
    words=sentence.split(" ")
    #在分词的基础上，标注词性
    postags = postagger.postag(words)#这里的words是分词后的结果
    postags=' '.join(postags).split()

    #在分词、词性的基础上，进行句法依存分析
    arcs = parser.parse(words,postags)

    #把依存关系的父节点和弧类型分开存入2个不同的列表中
    head=[]
    relation=[]

    for arc in arcs:
        head.append(arc[0])
        relation.append(arc[1])
    return head,relation,   words,postags




#这个函数用来从句法依存中抽取修饰词
#a表示抽取到的修饰词，x_list是前面定义好的：全部修饰词典
"""抽取否定/程度词（句法规则7）"""
def xiushi(word,words,relation,head):#找到情感词的修饰词
    j=0
    a=[]
    #遍历句子分词列表
    for i in words:
        #①当前分词位于修饰词典中②依存类型属于。。③父节点为当前传入的词语word
        if i in x_list and relation[j] in ['CMP','ADV'] and words[head[j]-1]==word:
            a.append(i)
        j+=1
    return a

#列表字典去重
def list_dict_duplicate_removal(list_dict):
    """list中dict重复的话，去重"""
    run_function = lambda x, y: x if y in x else x + [y]
    return reduce(run_function, [[], ] + list_dict)


#提取情感三元组
def get_list(sentence):

    #用来存三元组
    d_list=[]    
    head,relation,words,postags=get_detail(sentence)

    i=0
    for word in words:

        #i是当前“分词”的索引-1  【当前分词是子节点】
        #word就是当前分词
        #words[head[i]-1是父节点对应的词
        # print(i,word,relation[i],words[head[i]-1])
        """
        0 他 SBV 叫
        1 叫 HED 。
        2 汤姆 DBL 叫
        3 去 ADV 拿
        4 拿 VOB 叫
        5 外衣 VOB 拿
        6 。 WP 叫
        """

        #t_list是完整的属性词典、q_list是情感词典的全部。
        #word表示当前分词、i表示该分词在列表中的索引（同时也是依存类型、所依赖父节点的列表索引，通过i可以找到它们）
        #relation[i]表示当前分词的弧类型、head[i]表示该分词的父节点编号，因为父节点在编号时多了一个root，所以这个编号和分词列表的下标不一致
        #只有把父节点编号-1才能得到父词在分词列表中的下标，然后找到父节点对应的词

        #当前分词【子节点】是属性词（处于属性词典中）、依存类型属于'SBV','VOB','FOB'、父节点属于情感词典中
        #SBV是主谓关系。我（子节点）-->吃（父节点）
        #ps."口感好"、”价格便宜“也是主谓关系（形容词【便宜/好】依赖于名词）
        #VOB是动宾关系。吃（宾语、子节点）-->饭
        #FOB是前置宾语（本质也是动宾关系）。”什么书都读，什么菜都吃“。书依赖于读、菜依赖于吃

        #以上三种关系中，句子的核心词”head“都是一个主要动词：”吃“、”读“。其他主语、名词都依赖于这个动词
        #核心动词只依赖于root
        #root是根节点，他只会指向一个节点（独生子）

        #？？？SBV关系这里是不是写反了
        if word in t_list and relation[i] in ['SBV','VOB','FOB'] and words[head[i]-1] in q_list:
            d={'属性词':word,'情感词':words[head[i]-1],'修饰词':xiushi(words[head[i]-1],words,relation,head)}
            print(f"句法规则1.1：{d}")
            d_list.append(d)

        #当前分词位于情感词典中、关系类型属于'ATT','CMP'、父节点属于属性词典中
        #ATT是形容词和名词的关系（定中关系）。”蓝月亮“、”红苹果“、”便宜的价格“。形容词会依赖于名词
        #CMP是动词和补语。”服务态度好“、”吃完“。补语（好、完）会依赖于动词/服务态度等名词
        """显示方面意见（句法规则1）"""
        #word in list？？？这里是不是有问题
        if word in q_list and relation[i] in ['ATT','CMP'] and words[head[i]-1] in t_list:
            d={'属性词':words[head[i]-1],'情感词':word,'修饰词':xiushi(word,words,relation,head)}
            print(f"句法规则1.2：{d}")
            d_list.append(d)

        # print(word,relation[i],words[head[i]-1])
        #并列关系。”猫和老鼠“、”喜羊羊与灰太狼“。如果一个位于属性词内、一个位于情感词内的话，就提取
        if word in t_list and relation[i] in ['COO'] and words[head[i]-1] in q_list:
            d={'属性词':word,'情感词':words[head[i]-1],'修饰词':xiushi(words[head[i]-1],words,relation,head)}
            print(f"句法规则1.3：{d}")
            d_list.append(d)

        if word in q_list and relation[i] in ['COO'] and words[head[i]-1] in t_list:
            d={'属性词':words[head[i]-1],'情感词':word,'修饰词':xiushi(word,words,relation,head)}
            print(f"句法规则1.4：{d}")
            d_list.append(d)


        i+=1


    """隐示方面意见（句法规则2），应对省略主语的情况"""
    #把刚刚得到的显式三元组中的情感词放到q-word中。
    #作用是，抽取隐式依赖时，把显示依赖排除掉
    q_word = []
    for d in d_list:
        q_word.append(d['情感词'])

    #重新迭代了该句子
    for word in words:
        if word not in q_word and word in q_list1:
            d={'属性词':'品质','情感词':word,'修饰词':xiushi(word,words,relation,head)}
            d_list.append(d)
        if word not in q_word and word in q_list2:
            d={'属性词':'味道','情感词':word,'修饰词':xiushi(word,words,relation,head)}
            d_list.append(d)
        if word not in q_word and word in q_list3:
            d={'属性词':'价格','情感词':word,'修饰词':xiushi(word,words,relation,head)}
            d_list.append(d)
        if word not in q_word and word in q_list4:
            d={'属性词':'分量','情感词':word,'修饰词':xiushi(word,words,relation,head)}
            d_list.append(d)
        if word not in q_word and word in q_list5:
            d={'属性词':'外观','情感词':word,'修饰词':xiushi(word,words,relation,head)}
            d_list.append(d)
        if word not in q_word and word in q_list6:
            d={'属性词':'物流','情感词':word,'修饰词':xiushi(word,words,relation,head)}
            d_list.append(d)
        if word not in q_word and word in q_list7:
            d={'属性词':'客服','情感词':word,'修饰词':xiushi(word,words,relation,head)}
            d_list.append(d)


    """潜在方面意见（句法规则3），用核心词识别未登录属性词"""
    """显示、隐式都可以直接得出该三元组属于哪个属性维度。但是潜在三元组不行，所以通过word2vec相似度匹配"""
    # i=0
    # for word,postag in zip(words,postags):
    #     #如果
    #     if word not in t_list and relation[i]=='HED' and postag in ['n','v','nz']:
    #返回核心词的下标
    i=relation.index("HED")
    if words[i] not in t_list and postags[i] in ['n','v','nz']:
        print(f"{words[i]}被列为了候选属性词")
        for j in range(len(words)):
            if words[head[j]-1]==words[i] and relation[j] in ['VOB','CMP'] and postags[i] !='v':
                d = {'属性词':words[i], '情感词': words[j], '修饰词': xiushi(word, words, relation, head)}
                print(f"句法规则3：{d}")
                d_list.append(d)

    """潜在方面意见（句法规则4），用核心词识别未登录情感词"""
    if words[i] not in q_list and postags[i] in ['a', 'u', 'd']:
        print(f"{words[i]}被列为了候选情感词")
        for j in range(len(words)):
            if words[head[j]-1]==words[i] and relation[j] in ['SBV','ATT'] and postags[i] in ['v','n','j','nz']:
                d = {'属性词':words[head[j]-1], '情感词': words[i], '修饰词': xiushi(word, words, relation, head)}
                print(f"句法规则4.1：{d}")
                d_list.append(d)
            else:
                d = {'属性词': 'null', '情感词': words[i], '修饰词': xiushi(word, words, relation, head)}
                print(f"句法规则4.2：{d}")
                d_list.append(d)

    """潜在方面意见（句法规则5），识别核心词周围的情感表达"""

    for j in range(len(words)):
        #根据候选属性词确定情感词
        if words[head[j] - 1] == words[i] and relation[j] in ['SBV', 'VOB'] and postags[i] in ['v', 'n', 'j',
                                                                                    'nz'] and len(words[j])>1:

            print(words[j],"作为了候选属性词")
            for k in range(len(words)):
                #属性词与情感词不能一样，所以and words[k]!=words[j]
                if words[head[k] - 1] == words[i] and words[k] in q_list and words[k]!=words[j]:
                    d = {'属性词': words[j], '情感词': words[k], '修饰词': xiushi(word, words, relation,head)}
                    print(f"句法规则5.1：{d}")
                    d_list.append(d)

        if words[head[j] - 1] == words[i] and relation[j]=='CMP' and postags[i] in ['a', 'n'] and len(words[j]) > 1:
            print(words[j],"作为了候选情感词")
            for k in range(len(words)):
                if words[head[k] - 1] == words[i] and words[k] in t_list and words[k]!=words[j]:
                    d = {'属性词': words[k], '情感词': words[j], '修饰词': xiushi(word, words, relation,
                                                                                             head)}
                    print(f"句法规则5.2：{d}")
                    d_list.append(d)
                else:
                    d = {'属性词':'null', '情感词': words[j], '修饰词': xiushi(word, words, relation,
                                                                                             head)}
                    print(f"句法规则5.3：{d}")
                    d_list.append(d)

    #去重
    # print(len(d_list))
    d_list = list_dict_duplicate_removal(d_list)
    # print(len(d_list))
    return (d_list)

# d_list=get_list("这家店 的 花生 嚼着 很香")
#d_list=get_list("干净 卫生 性价比 很 高 适合 旅游 出差 点 赞 干净 卫生 性价比 很 高 适合 旅游 出差 点 赞 真心 好 推荐 住")
# print(d_list)
#d_list=get_list("非常宽敞 特别干净 面积真的很大 装修风格很简单但真的很好看 挺商务的 最近住过的大连酒店这里最喜欢了 住过港汇好多次了哈哈")
# print(d_list)
# [{'属性词': '性价比', '情感词': '高', '修饰词': ['很']}, {'属性词': '酒店', '情感词': '高', '修饰词': ['很']}, {'属性词': '交通', '情感词': '方便', '修饰词': ['很']}, {'属性词': '环境', '情感词': '舒服', '修饰词': ['很']}, {'属性词': '服务', '情感词': '热情', '修饰词': ['非常']}]
# [{'属性词': '酒店', '情感词': '喜欢', '修饰词': ['最']}, {'属性词': '环境', '情感词': '宽敞', '修饰词': []}, {'属性词': '环境', '情感词': '干净', '修饰词': ['特别']}]


"""上面都是在抽取三元组
下面开始计算句子得分了"""

#倾向分析
#定义判断奇偶的函数
def judgeodd(num):
    if num%2==0:
        return 'even'
    else:
        return 'odd'

#a_list是第一个子维度的属性词典
def score(alist,d_list):
    #得分，初始为0
    totalcount=0
    count=0
    a=0#这句话中，命中的属性词。即情感三元组中的属性词属于“属性词词典”则a+1
    for d in d_list:#d是第一个情感三元组
        if d['属性词'] in alist:
            a+=1
            #该元组的属性词属于维度n的属性词典，且情感词位于积极词典。
            #否定修饰词个数为计数则-1,偶数则+1
            if d['情感词'] in posdict:
                count=1
            elif d['情感词'] in negdict:
                count = -1
            else:
                print("打分时出现了情感词典中不存在的词", d['情感词'])
                continue

            c = 0 #情感词前否定词的个数
            for w in d['修饰词']:
                #否定词典
                if w in inversedict:
                    c += 1
                if w in mostdict:
                    count*=2
                if w in verydict:
                    count*=1.6
                if w in moredict:
                    count*=1.2
                if w in ishdict:
                    count*=0.8
                if w in ishdict:
                    count *= 0.5
            #否定词个数为奇数，得分翻转
            if judgeodd(c) == 'odd':
                count=-1*count


            totalcount+=count
    #如果所有三元组的属性都不在子维度n的属性词典中，则a=0，totalcount=20
    if a==0:
        return 20
    else:
        return totalcount

#bug:属性词命中了，但是情感词未命中，导致返回发totalcount1=0，最后被划为负面
def s_sentence(d_list):
    #六个评价子维度的默认得分（选一个不可能达到的分数）
    score_dict={'品质':20,
                '味道':20,
                '价格':20,
                '分量':20,
                '外观':20,
                '物流':20,
                '客服':20,}
    #计算得分
    totalcount1=score(alist,d_list)
    if totalcount1==20:
        score_dict['品质']=2
    elif totalcount1>=0:
        score_dict['品质']=1
    elif totalcount1<0:
        score_dict['品质']=0
        
    totalcount2=score(blist,d_list)
    if totalcount2==20:
        score_dict['味道']=2
    elif totalcount2>=0:
        score_dict['味道']=1
    elif totalcount2<0:
        score_dict['味道']=0
    
    totalcount3=score(clist,d_list)
    if totalcount3==20:
        score_dict['价格']=2
    elif totalcount3>=0:
        score_dict['价格']=1
    elif totalcount3<0:
        score_dict['价格']=0

    totalcount4=score(dlist,d_list)
    if totalcount4==20:
        score_dict['分量']=2
    elif totalcount4>=0:
        score_dict['分量']=1
    elif totalcount4<0:
        score_dict['分量']=0

    totalcount5=score(elist,d_list)
    if totalcount5==20:
        score_dict['外观']=2
    elif totalcount5>=0:
        score_dict['外观']=1
    elif totalcount5<0:
        score_dict['外观']=0

    totalcount6=score(flist,d_list)
    if totalcount6==20:
        score_dict['物流']=2
    elif totalcount6>=0:
        score_dict['物流']=1
    elif totalcount6<0:
        score_dict['物流']=0

    totalcount7 = score(glist, d_list)
    if totalcount7 == 20:
        score_dict['客服'] = 2
    elif totalcount7 >= 0:
        score_dict['客服'] = 1
    elif totalcount7 < 0:
        score_dict['客服'] = 0

    return score_dict



a="品质、果肉、质地、营养、柚子、水果、商品"
b="味道、口感、尝、吃起来、闻、甜度、吃、肉、水份、水分、酸度"
c="价格、性价比、价钱、定价、成本、售价、价位"
d="分量、个头、大小、斤、箱、袋、个数"
e="包装、质量、配色、卖相、包装袋、包装盒、外包装、外盒、外观、看起来、看、摸起来、色泽、外皮、颜色、光泽、果皮、纹路、外皮、皮"
f="物流、发货、服务态度、运送、送货、配送、取件、收货、送货上门"
g="客服、态度、商家、卖家、店家、体验、处理、下单、厂家、补救、客服人员、感受、网店、服务、售后"
alist=a.split("、")
blist=b.split("、")
clist=c.split("、")
dlist=d.split("、")
elist=e.split("、")
flist=f.split("、")
glist=g.split("、")
#t_list是上述list之和（total）
#属性词典
t_list=alist+blist+clist+dlist+elist+flist+glist
#隐式观点分析。从情感词典中抽取出对应评价维度的词
q1="新鲜、放心、保证、不错、丰富、完美、干净、发霉、坏果、烂果、干瘪、遗憾、腐烂、好、行、发白、发硬、很足、差评、可以、正宗、正货、烂、棒、爱、变质、赞、喜欢、坏、谨慎、建议买、垃圾、五星好评、扔了、凑合、上当、退货、臭"
q2="好吃、甜、酸甜、美味、适中、不错、细腻、甜甜、浓郁、细嫩、鲜嫩、爽口、不酸、光滑、酸酸、难吃、酸味、微苦、酸苦、酸涩、苦涩、寡淡、巨苦、好、老、苦、酸、差点、厚、很足、可以、干、刚好、酸甜可口、涩口、酸酸甜甜、爱吃"
q3="划算、便宜、实惠、合适、优惠、值得、不错、上涨、贵、不值、抬价、小贵、物美价廉、物超所值"
q4="沉甸甸、大、足、适中、够、充足、合适、均匀、匀称、适宜、小、少、不足、一般、小个、大个"
q5="柔软、硬硬、光滑、超厚、发白、发硬、厚实、紧实、薄、细腻、皮薄、完好、结实、讲究、干净、严实、精致、紧、干、仔细、不错、皮厚、漂亮、完美、损坏、费劲、简陋、变形、破了、伤疤、高端、物美价廉"
q6="快、快速、给力、不错、方便、快捷、完整、慢、差、没收到、糟糕、摔烂、拖、久、可以、收到、送达、垃圾、慢慢、没收到、送货上门、压扁、丢"
q7="满意、及时、礼貌、热心、细致、周到、耐心、友善、亲切、良心、恶劣、虚假宣传、欺骗、投诉、补偿、回购、好评、发错、符合、不符、骗人、退、不退、吻合、光顾、取关、五星好评、麻烦、索赔、信赖、缺斤少两"
q_list1=q1.split("、")
q_list2=q2.split("、")
q_list3=q3.split("、")
q_list4=q4.split("、")
q_list5=q5.split("、")
q_list6=q6.split("、")
q_list7=q7.split("、")

# 1，三元组提取测试
# if __name__ == '__main__':
#     syz=get_list("柚子 味道 非常 好 ， 很 香 ， 果肉 饱满 ， 水分 足 ， 没有 坏果 ， 商家 发货 速度 十分 快 ， 包装 严实 完整 ， 就 是 这边 送货 态度 恶劣 ， 客服 也 负 责任 处理 解决 。")
#     print(syz)
#     print(s_sentence(syz))






#展示部分内容
# if __name__ == '__main__':
#     content = []
#     score_list = []
#     san_yuan_zu_list = []
#
#     df = pd.read_csv(r"D:\Pythonnnn\弱监督的农业社会化销售服务评价\data\红心柚\分词后_无停用词.csv")
#     content = df["分词"].values
#
#     for sentence in content[0:5]:
#         # d_list是情感三元组，score_dict是7个维度上的得分
#         d_list = get_list(sentence)
#         score_dict = s_sentence(d_list)
#
#         # 把[{'属性词': '酒店', '情感词': '喜欢', '修饰词': ['最']}, {'属性词': '环境', '情感词': '宽敞', '修饰词': []}]
#         # 把d_list这种列表变成元组列表
#         # 在转换一条评论之前，一定要记得把元组列表清空
#         d_turple_list = []
#         # i是一个字典，d_list是内容为字典的列表
#         for i in d_list:
#             # 属性词、情感词只会有一个，修饰词却会有多个。修饰词是一个列表，所以把它转换为空格分隔的字符串
#             str = " ".join(i["修饰词"])
#             d_turple = (i['属性词'], str, i["情感词"])
#             d_turple_list.append(d_turple)
#
#         san_yuan_zu_list.append(d_turple_list)
#         score_list.append(score_dict)
#
#     # 存为字典。value是长列表
#     final_dict = {"comments": content[0:5], 'san_yuan_zu': san_yuan_zu_list,
#                   'value': score_list}
#
#     df = pd.DataFrame(final_dict)
#     df.to_csv(r"D:\Pythonnnn\弱监督的农业社会化销售服务评价\data\红心柚\finel_sample.csv", index=False)


if __name__ == '__main__':
    content = []
    score_list = []
    san_yuan_zu_list = []

    # df = pd.read_csv(r"D:\Pythonnnn\弱监督的农业社会化销售服务评价\data\红心柚\分词后_无停用词_空格.csv")
    df = pd.read_csv(r"D:\Pythonnnn\弱监督实验\原始数据\红心柚\新增差评分词后.csv")

    content = df["分词"].values

    for sentence in content:
        # d_list是情感三元组，score_dict是7个维度上的得分
        d_list = get_list(sentence)
        score_dict = s_sentence(d_list)

        # 把[{'属性词': '酒店', '情感词': '喜欢', '修饰词': ['最']}, {'属性词': '环境', '情感词': '宽敞', '修饰词': []}]
        # 把d_list这种列表变成元组列表
        # 在转换一条评论之前，一定要记得把元组列表清空
        d_turple_list = []
        # i是一个字典，d_list是内容为字典的列表
        for i in d_list:
            # 属性词、情感词只会有一个，修饰词却会有多个。修饰词是一个列表，所以把它转换为空格分隔的字符串
            str = " ".join(i["修饰词"])
            d_turple = (i['属性词'], str, i["情感词"])
            d_turple_list.append(d_turple)

        san_yuan_zu_list.append(d_turple_list)
        score_list.append(score_dict)

    # 存为字典。value是长列表
    final_dict = {"comments": content, 'san_yuan_zu': san_yuan_zu_list,
                  'value': score_list}

    df = pd.DataFrame(final_dict)
    #增加了词语之后的就是finaal2
    df.to_csv(r"D:\Pythonnnn\弱监督实验\原始数据\红心柚\新增差评依存得分.csv", index=False)
