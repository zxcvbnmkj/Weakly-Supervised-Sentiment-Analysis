# -*- coding: utf-8 -*-
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

warnings.filterwarnings("ignore")
sys.path.append('D:\Pythonnnn\弱监督的农业社会化销售服务评价\深度学习')
# from 深度学习.DiceLoss import MultiDSCLoss
import torch
from 深度学习.configASAP import device, label_names, print_every
print(label_names)
from 深度学习.data_gen_asap import SaDataset
from 深度学习.utils import accuracy, Lang
import logging

#CE模型不能使用DSC函数的推理代码
a=[]
b=[]
c=[]
d=[]
e=[]
fff=[]
g=[]
h=[]
i9=[]
i10=[]
i11=[]
i12=[]
i13=[]
i14=[]
i15=[]
i16=[]
i17=[]
i18=[]

def accuracy(scores, targets,idx, k=1):

    batch_size = targets.size(0)
    #tensor.topk方法。参数1表示返回最大的几个数。参数2表示在哪个维度返回，参数3表示是否按从大到小的顺序。参数4表示返回的k个数是否要排序
    #返回值1表示返回的k的值，返回值2则是k的索引
    #标签
    value, ind = scores.topk(k, 1, True, True)
    #概率
    # ind, value = scores.topk(k, 1, True, True)
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
    global i9
    global i10
    global i11
    global i12
    global i13
    global i14
    global i15
    global i16
    global i17
    global i18
    if idx==0:
        a.append(ind.tolist())
    if idx==1:
        b.append(ind.tolist())
    if idx==2:
        c.append(ind.tolist())
    if idx==3:
        d.append(ind.tolist())
    if idx==4:
        e.append(ind.tolist())
    if idx==5:
        fff.append(ind.tolist())
    if idx==6:
        g.append(ind.tolist())
    if idx==7:
        h.append(ind.tolist())
    if idx==8:
        i9.append(ind.tolist())
    if idx==9:
        i10.append(ind.tolist())
    if idx==10:
        i11.append(ind.tolist())
    if idx==11:
        i12.append(ind.tolist())
    if idx==12:
        i13.append(ind.tolist())
    if idx==13:
        i14.append(ind.tolist())
    if idx==14:
        i15.append(ind.tolist())
    if idx==15:
        i16.append(ind.tolist())
    if idx==16:
        i17.append(ind.tolist())
    if idx==17:
        i18.append(ind.tolist())







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
        filename="../深度学习/log/模型推理.log")
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
                # loss += criterion(outputs[:, :, idx], target_variable[:, idx]) / len(label_names)
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
    voc = Lang(r'D:\google_drive\weekly_learn\dl_data\asap_WORDMAP.json')

    # val_data = SaDataset(r'D:\Pythonnnn\弱监督的农业社会化销售服务评价\弱监督\test_ture\新测试集_200个.csv', voc)
    val_data = SaDataset(r'D:\google_drive\weekly_learn\dl_data\ASAP测试集.csv', voc)

    #创建模型实例。参数1是input_size，它其实就是vocab.txt的长度。它决定了embedding层的行
    # encoder = EncoderRNN(voc.n_words, hidden_size, encoder_n_layers, dropout)

    #要加载模型的地址
    checkpoint = r'D:\google_drive\weekly_learn\save_models\11\BEST_checkpoint.tar'
    # checkpoint = r'C:\Users\张丽\Desktop\google_drive\深度学习\models\models\checkpoint_11_93.999.tar'
    checkpoint = torch.load(checkpoint)
    encoder = checkpoint['encoder']
    print(device)
    encoder = encoder.to(device)
    encoder.eval()

    criterion = torch.nn.CrossEntropyLoss().to(device)
    # criterion=MultiDSCLoss(alpha=0.8)

    val_acc = valid(val_data, encoder,criterion)

    global a
    global b
    global c
    global d
    global e
    global fff
    global g
    global h
    global i9
    global i10
    global i11
    global i12
    global i13
    global i14
    global i15
    global i16
    global i17
    global i18
    a = list(np.ravel(a))
    b = list(np.ravel(b))
    c = list(np.ravel(c))
    d = list(np.ravel(d))
    e = list(np.ravel(e))
    fff = list(np.ravel(fff))
    g = list(np.ravel(g))
    h = list(np.ravel(h))
    i9 = list(np.ravel(i9))
    i10 = list(np.ravel(i10))
    i11 = list(np.ravel(i11))
    i12 = list(np.ravel(i12))
    i13 = list(np.ravel(i13))
    i14 = list(np.ravel(i14))
    i15 = list(np.ravel(i15))
    i16= list(np.ravel(i16))
    i17= list(np.ravel(i17))
    i18= list(np.ravel(i18))

    df = pd.DataFrame(data={"品质": a,'味道':b,'价格':c,'分量':d,'外观':e,'物流':fff,'客服':g,'粗粒度':h})
    print(len(df))
    # 加上索引，为了让列数与测试集列数保持一致
    df.to_csv("农业模型预测标签ASAP.csv",index=False)

if __name__ == '__main__':
    main()
