# -*- coding: utf-8 -*-
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

warnings.filterwarnings("ignore")
from 深度学习.DiceLoss import MultiDSCLoss
import torch
from torch import optim
from config import device, label_names, print_every
from data_gen import SaDataset
from utils import accuracy, Lang
import logging

softmax = torch.nn.Softmax(dim=1)
a=[]
b=[]
c=[]
d=[]
e=[]
fff=[]
g=[]
h=[]
def accuracy(scores, targets,idx, k=1):

    batch_size = targets.size(0)
    #tensor.topk方法。参数1表示返回最大的几个数。参数2表示在哪个维度返回，参数3表示是否按从大到小的顺序。参数4表示返回的k个数是否要排序
    #返回值1表示返回的k的值，返回值2则是k的索引
    #把输出穿过softmax层
    # scores = softmax(scores)
    #标签
    value, ind = scores.topk(k, 1, True, True)
    # print(value)
    # print(f"当前是标签{idx}，预测结果是：",ind)
    global a
    global b
    global c
    global d
    global e
    global fff
    global g
    global h
    if idx==0:
        a.extend(scores.tolist())
    if idx==1:
        b.extend(scores.tolist())
    if idx==2:
        c.extend(scores.tolist())
    if idx==3:
        d.extend(scores.tolist())
    if idx==4:
        e.extend(scores.tolist())
    if idx==5:
        fff.extend(scores.tolist())
    if idx==6:
        g.extend(scores.tolist())
    if idx==7:
        h.extend(scores.tolist())







    # print("真实标签是：", targets)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    print('命中个数: ',int(correct_total.item()),"该批次总个数",batch_size)

    p,r,f,s=precision_recall_fscore_support(targets, ind,average=None)
    print(f"p:{p}\nr:{r}\nf:{f}")

    p, r, f, s = precision_recall_fscore_support(targets, ind, average='macro')

    return correct_total.item() * (100.0 / batch_size),p,r,f

def create_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')

    # 创建一个handler，用于写入日志文件
    file_handler = logging.FileHandler(
        #日志文件的地址
        filename="./log/模型推理.log")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    return logger
#创建日志
logger = create_logger()


def valid(val_data, encoder,criterion):
    encoder.eval()  # eval mode (no dropout or batchnorm)

    valid_avg_acc=0
    valid_avg_p=0
    valid_avg_r=0
    valid_avg_f1=0



    with torch.no_grad():
        # Batches
        for i_batch, (input_variable, lengths, target_variable) in enumerate(val_data):
            acc_test = 0
            recall_test = 0
            precious_test = 0
            f1_test = 0


            input_variable = input_variable.to(device)
            lengths = lengths.to(device)
            target_variable = target_variable.to(device)

            outputs= encoder(input_variable, lengths)

            loss = 0

            for idx, _ in enumerate(label_names):
                loss += criterion(outputs[:, :, idx], target_variable[:, idx]) / len(label_names)
                # acc += accuracy(outputs[:, :, idx], target_variable[:, idx],idx) / len(label_names)
                a2, p, r, f = accuracy(outputs[:, :, idx], target_variable[:, idx], idx)

                acc_test += a2 / len(label_names)
                recall_test += r / len(label_names)
                precious_test += p / len(label_names)
                f1_test += f / len(label_names)

            valid_avg_acc+=acc_test/len(val_data)
            valid_avg_p+=precious_test/len(val_data)
            valid_avg_r+=recall_test/len(val_data)
            valid_avg_f1+=f1_test/len(val_data)


            if i_batch % print_every == 0:
                logger.info('Validation（批次数/总批数）: [{0}/{1}]\t'.format(i_batch, len(val_data)))
                logger.info(r"本批次的测试集准确率、macro-p\r\f是{}、{}、{}、{}".format(acc_test,precious_test, recall_test, f1_test))

    logger.info("整个验证集的准确率、p、r、f1是{}、{}、{}、{}".format(valid_avg_acc,valid_avg_p,valid_avg_r,valid_avg_f1))

    return valid_avg_acc


def main():
    #获取词频文件
    voc = Lang('./dl_data/1_WORDMAP.json')

    #纯伪
    # val_data = SaDataset(r'D:\Pythonnnn\弱监督的农业社会化销售服务评价\伪标签测试集2.csv', voc)
    #伪+真
    # val_data = SaDataset(r'D:\Pythonnnn\弱监督的农业社会化销售服务评价\伪标签200个测试集.csv', voc)
    val_data = SaDataset(r'D:\google_drive\weekly_learn\dl_data\测试集.csv', voc)

    #创建模型实例。参数1是input_size，它其实就是vocab.txt的长度。它决定了embedding层的行
    # encoder = EncoderRNN(voc.n_words, hidden_size, encoder_n_layers, dropout)

    #要加载模型的地址
    checkpoint = r'D:\google_drive\weekly_learn\save_models\6\BEST_checkpoint.tar'
    # checkpoint = r'C:\Users\张丽\Desktop\google_drive\深度学习\models\models\checkpoint_11_93.999.tar'
    checkpoint = torch.load(checkpoint)
    encoder = checkpoint['encoder']

    encoder = encoder.to(device)
    encoder.eval()

    # criterion = nn.CrossEntropyLoss().to(device)
    criterion=MultiDSCLoss(alpha=0.8)

    val_acc = valid(val_data, encoder,criterion)

    global a
    global b
    global c
    global d
    global e
    global fff
    global g
    global h

    with open("../ROC曲线/品质.pkl", "wb") as f:
        pickle.dump(a, f)
    with open("../ROC曲线/味道.pkl", "wb") as f:
        pickle.dump(b, f)
    with open("../ROC曲线/价格.pkl", "wb") as f:
        pickle.dump(c, f)
    with open("../ROC曲线/分量.pkl", "wb") as f:
        pickle.dump(d, f)
    with open("../ROC曲线/外观.pkl", "wb") as f:
        pickle.dump(e, f)
    with open("../ROC曲线/物流.pkl", "wb") as f:
        pickle.dump(fff, f)
    with open("../ROC曲线/客服.pkl", "wb") as f:
        pickle.dump(g, f)
    with open("../ROC曲线/粗粒度.pkl", "wb") as f:
        pickle.dump(h, f)

    #ravel把多维列表拉伸为一维列表
    # a = list(np.ravel(a))
    # b = list(np.ravel(b))
    # c = list(np.ravel(c))
    # d = list(np.ravel(d))
    # e = list(np.ravel(e))
    # fff = list(np.ravel(fff))
    # g = list(np.ravel(g))
    # h = list(np.ravel(h))

    # df = pd.DataFrame(data={"品质": a,'味道':b,'价格':c,'分量':d,'外观':e,'物流':fff,'客服':g,'粗粒度':h})
    # print(len(df))
    # # 加上索引，为了让列数与测试集列数保持一致
    # df.to_csv("农业模型预测标签_概率.csv",index=False)

if __name__ == '__main__':
    main()
