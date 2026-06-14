
#这个数据集里面没有定义或import  label_name等变量，却不会报错
#是因为from utils import *这里，utils里面import了config.py
import itertools
import jieba
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import sys
sys.path.append('D:\google_drive\weekly_learn')
from utils_asap import *

print(label_names)

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='uint8')[y]


# Meaning	    Positive	Neutral	    Negative	Not mentioned
# Old labels    1	        0	        -1	        -2
# New labels    3           2           1           0
#使标签从0起始
# def map_sentimental_type(value):
#     return value + 2


def parse_user_reviews(user_reviews):
    samples = []
    #遍历每一条评论（包括标签）
    for i in range(len(user_reviews)):
        #第i条评论
        content = user_reviews['review'][i]
        #num_labels是定义在config中的，表示具体的分类数
        label_tensor = np.empty((num_labels,), dtype=np.int32)
        #label_names是预先定义的20个列的名字
        for idx, name in enumerate(label_names):
            sentimental_type = user_reviews[name][i]
            # label_tensor[:, idx] = to_categorical(y, num_classes)
            # CrossEntropyLoss does not expect a one-hot encoded vector as the target, but class indices.
            label_tensor[idx] = sentimental_type
        #处理之后存入列表的字典中。label存为一串数字
        samples.append({'content': content, 'label_tensor': label_tensor})
    # print(samples)
    return samples


def zeroPadding(l, fillvalue=PAD_token):
    return list(itertools.zip_longest(*l, fillvalue=fillvalue))


# Returns padded input sequence tensor and lengths
def inputVar(indexes_batch):
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    padVar = torch.LongTensor(padList)
    return padVar, lengths


# Returns all items for a given batch of pairs返回给定批次对的所有项
def batch2TrainData(pair_batch):
    #按长度排序。由长到短
    pair_batch.sort(key=lambda x: len(x[0]), reverse=True)
    input_batch, output_batch = [], []
    for pair in pair_batch:
        input_batch.append(pair[0])
        output_batch.append(pair[1])
    inp, lengths = inputVar(input_batch)
    #标签从array类型转换为了tensor类型
    output = torch.LongTensor(output_batch)
    #padding后的评论ids，评论的实际长度，标签（tensor类型）
    #他们三都是列表，长度都为chunk_size（batch_size）。也就是一个样本的三个要素分别打包了
    return inp, lengths, output


class SaDataset(Dataset):
    #split传入要处理的数据集
    def __init__(self, filename, voc):
        #修改：把split改为直接传入地址
        #self.split = split
        self.voc = voc
        # assert self.split in {'train', 'valid'}

        # if split == 'train':
        #     filename = os.path.join(train_folder, train_filename)
        # elif split == 'valid':
        #     filename = os.path.join(valid_folder, valid_filename)
        # else:
        #     filename = os.path.join(test_a_folder, test_a_filename)

        user_reviews = pd.read_csv(filename)
        #units中有一个同名函数parse_user_reviews。但是优先使用本文件中的函数
        self.samples = parse_user_reviews(user_reviews)
        #//是地板除，表示先做除法再向下取整
        self.num_chunks = len(self.samples) // chunk_size

    def __getitem__(self, i):
        pair_batch = []

        for i_chunk in range(chunk_size):
            idx = i * chunk_size + i_chunk
            content = self.samples[idx]['content']
            content = content.strip()
            seg_list = jieba.cut(content)
            #把一句话转换为词频文件中的id。加上UNK\END两种表示方式
            input_indexes = encode_text(self.voc.word2index, list(seg_list))
            label_tensor = self.samples[idx]['label_tensor']
            #把一个chunk的样本放在一个列表中
            pair_batch.append((input_indexes, label_tensor))

        return batch2TrainData(pair_batch)

    def __len__(self):
        return self.num_chunks
