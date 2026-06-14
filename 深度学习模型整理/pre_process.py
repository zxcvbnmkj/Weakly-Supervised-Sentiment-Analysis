"""执行此文件用来创建语料的词频。
最后结果类似于vocab.txt"""
import sys
sys.path.append('/content/drive/Othercomputers/我的笔记本电脑/google_drive/weekly_learn')
import json
from collections import Counter
import jieba
import pandas as pd
from pyltp import Segmentor
from tqdm import tqdm
from config import *


segmentor = Segmentor(r"cws.model")
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
    #把所有编号后移4格
    word_map = {k: v + 4 for v, k in enumerate(words)}
    word_map['<pad>'] = 0
    word_map['<start>'] = 1
    word_map['<end>'] = 2
    word_map['<unk>'] = 3
    print('len(word_map): ' + str(len(word_map)))

    with open('./dl_data/asap_WORDMAP.json', 'w') as file:
        #indent指定缩进量
        json.dump(word_map, file, indent=4)


if __name__ == '__main__':
    df1=pd.read_csv("/content/drive/Othercomputers/我的笔记本电脑/google_drive/weekly_learn/dl_data/ASAP训练集.csv")
    df2 = pd.read_csv("/content/drive/Othercomputers/我的笔记本电脑/google_drive/weekly_learn/dl_data/ASAP训练集.csv")
    df=pd.concat([df1,df2])
    print(len(df))
    build_wordmap(df['review'])
