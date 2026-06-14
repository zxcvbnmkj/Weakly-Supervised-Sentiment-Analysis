"""执行此文件用来创建语料的词频。
最后结果类似于vocab.txt"""
import json
from collections import Counter
import jieba
import pandas as pd
from pyltp import Segmentor
from tqdm import tqdm
from config import *


segmentor = Segmentor(r"D:\Pythonnnn\1resource\pyltp_model\cws.model",
                      lexicon_path=r"D:\Pythonnnn\弱监督的农业社会化销售服务评价\2preprocess\user_dict.txt")
def build_wordmap(contents):
    word_freq = Counter()
    #tqdm库用来显示进度条
    for sentence in tqdm(contents):
        #strip()去掉字符串末尾的空格
        seg_list = segmentor.segment(sentence.strip())
        # Update word frequency
        #只遍历了第一句话时，词频仅有第一句话中的。遍历到第二句时，会更新词频
        word_freq.update(list(seg_list))

    # Create word map
    #如果词频大于规定的最小词频，就放入words中
    words = [w for w in word_freq.keys() if word_freq[w] > min_word_freq]
    print(words)
    df=pd.DataFrame(data={"word":words})
    df.to_csv(r"D:\1filezy\R\主题词云图\所有词语.csv",index=False)
    #把所有编号后移4格
    # word_map = {k: v + 4 for v, k in enumerate(words)}
    # word_map['<pad>'] = 0
    # word_map['<start>'] = 1
    # word_map['<end>'] = 2
    # word_map['<unk>'] = 3
    # print('len(word_map): ' + str(len(word_map)))
    #
    # with open('./dl_data/1_WORDMAP.json', 'w') as file:
    #     #indent指定缩进量
    #     json.dump(word_map, file, indent=4)


if __name__ == '__main__':
    set=pd.read_csv("D:\\google_drive\\weekly_learn\\dl_data\\训练集.csv")
    build_wordmap(set['评论'])
