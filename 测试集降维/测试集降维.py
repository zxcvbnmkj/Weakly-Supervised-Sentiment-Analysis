# -*- coding: utf-8 -*-
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
sys.path.append('D:\Pythonnnn\弱监督的农业社会化销售服务评价\深度学习')
from 深度学习.utils import Lang,accuracy

warnings.filterwarnings("ignore")
from 深度学习.DiceLoss import MultiDSCLoss
import torch
from torch import optim
from 深度学习.config import device, label_names, print_every
from 深度学习.data_gen import SaDataset
# from 深度学习.utils import accuracy, Lang
import logging


data_embed_collect0=[]
data_embed_collect1=[]
data_embed_collect2=[]
data_embed_collect3=[]
data_embed_collect4=[]
data_embed_collect5=[]
data_embed_collect6=[]
data_embed_collect7=[]

label_collect0=[]
label_collect1=[]
label_collect2=[]
label_collect3=[]
label_collect4=[]
label_collect5=[]
label_collect6=[]
label_collect7=[]
def accuracy(scores, targets,idx, k=1):

    batch_size = targets.size(0)
    #tensor.topk方法。参数1表示返回最大的几个数。参数2表示在哪个维度返回，参数3表示是否按从大到小的顺序。参数4表示返回的k个数是否要排序
    #返回值1表示返回的k的值，返回值2则是k的索引
    value, ind = scores.topk(k, 1, True, True)
    # print(value)
    # print(f"当前是标签{idx}，预测结果是：",ind)



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


def valid(val_data, encoder):
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

            data_embed_collect0.append(outputs[:, :, 0])
            data_embed_collect1.append(outputs[:, :, 1])
            data_embed_collect2.append(outputs[:, :, 2])
            data_embed_collect3.append(outputs[:, :, 3])
            data_embed_collect4.append(outputs[:, :, 4])
            data_embed_collect5.append(outputs[:, :, 5])
            data_embed_collect6.append(outputs[:, :, 6])
            data_embed_collect7.append(outputs[:, :, 7])

            label_collect0.append(target_variable[:, 0])
            label_collect1.append(target_variable[:, 1])
            label_collect2.append(target_variable[:, 2])
            label_collect3.append(target_variable[:, 3])
            label_collect4.append(target_variable[:, 4])
            label_collect5.append(target_variable[:, 5])
            label_collect6.append(target_variable[:, 6])
            label_collect7.append(target_variable[:, 7])



            for idx, _ in enumerate(label_names):
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
    data_embed_npy0 = torch.cat(data_embed_collect0, axis=0).detach().numpy()
    data_embed_npy1 = torch.cat(data_embed_collect1, axis=0).detach().numpy()
    data_embed_npy2 = torch.cat(data_embed_collect2, axis=0).detach().numpy()
    data_embed_npy3 = torch.cat(data_embed_collect3, axis=0).detach().numpy()
    data_embed_npy4 = torch.cat(data_embed_collect4, axis=0).detach().numpy()
    data_embed_npy5 = torch.cat(data_embed_collect5, axis=0).detach().numpy()
    data_embed_npy6 = torch.cat(data_embed_collect6, axis=0).detach().numpy()
    data_embed_npy7 = torch.cat(data_embed_collect7, axis=0).detach().numpy()

    label_npu0 = torch.cat(label_collect0, axis=0).detach().numpy()
    label_npu1 = torch.cat(label_collect1, axis=0).detach().numpy()
    label_npu2 = torch.cat(label_collect2, axis=0).detach().numpy()
    label_npu3 = torch.cat(label_collect3, axis=0).detach().numpy()
    label_npu4 = torch.cat(label_collect4, axis=0).detach().numpy()
    label_npu5 = torch.cat(label_collect5, axis=0).detach().numpy()
    label_npu6 = torch.cat(label_collect6, axis=0).detach().numpy()
    label_npu7 = torch.cat(label_collect7, axis=0).detach().numpy()

    np.save("./2/data_embed_npy0_10.npy", data_embed_npy0)
    np.save("./2/data_embed_npy1_10.npy", data_embed_npy1)
    np.save("./2/data_embed_npy2_10.npy", data_embed_npy2)
    np.save("./2/data_embed_npy3_10.npy", data_embed_npy3)
    np.save("./2/data_embed_npy4_10.npy", data_embed_npy4)
    np.save("./2/data_embed_npy5_10.npy", data_embed_npy5)
    np.save("./2/data_embed_npy6_10.npy", data_embed_npy6)
    np.save("./2/data_embed_npy7_10.npy", data_embed_npy7)

    np.save("./2/label_npu0_10.npy", label_npu0)
    np.save("./2/label_npu1_10.npy", label_npu1)
    np.save("./2/label_npu2_10.npy", label_npu2)
    np.save("./2/label_npu3_10.npy", label_npu3)
    np.save("./2/label_npu4_10.npy", label_npu4)
    np.save("./2/label_npu5_10.npy", label_npu5)
    np.save("./2/label_npu6_10.npy", label_npu6)
    np.save("./2/label_npu7_10.npy", label_npu7)
    print("标签存储完毕")
    return valid_avg_acc


def main():
    #获取词频文件
    voc = Lang('../深度学习/dl_data/1_WORDMAP.json')

    # val_data = SaDataset(r'D:\Pythonnnn\弱监督的农业社会化销售服务评价\弱监督\test_ture\新测试集_200个.csv', voc)
    # val_data = SaDataset(r'D:\Pythonnnn\弱监督的农业社会化销售服务评价\伪标签200个测试集.csv', voc)
    val_data = SaDataset(r'D:\google_drive\weekly_learn\dl_data\测试集.csv', voc)


    #创建模型实例。参数1是input_size，它其实就是vocab.txt的长度。它决定了embedding层的行
    # encoder = EncoderRNN(voc.n_words, hidden_size, encoder_n_layers, dropout)

    #要加载模型的地址
    checkpoint = r'D:\google_drive\weekly_learn\save_models\2\BEST_checkpoint.tar'
    # checkpoint = r'C:\Users\张丽\Desktop\google_drive\深度学习\models\models\checkpoint_11_93.999.tar'
    checkpoint = torch.load(checkpoint)
    encoder = checkpoint['encoder']

    encoder = encoder.to(device)
    encoder.eval()



    valid(val_data, encoder)

if __name__ == '__main__':
    main()
