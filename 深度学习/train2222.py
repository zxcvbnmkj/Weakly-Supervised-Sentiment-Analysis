"""
RNN网络、交叉熵损失+log_softmax
"""
# -*- coding: utf-8 -*-
import pickle
import warnings

import pandas as pd

from 深度学习.DiceLoss import MultiDSCLoss
from 深度学习.models import EncoderRNN
from 深度学习.标签共现矩阵 import create_adjacency_matrix_cooccurance

warnings.filterwarnings("ignore")
import numpy as np
import torch
from torch import nn
from torch import optim
from config import device, label_names, print_every, hidden_size, encoder_n_layers, dropout, learning_rate, start_epoch, \
    epochs
from data_gen import SaDataset
# from models import EncoderRNN
from utils import accuracy, Lang, timestamp, adjust_learning_rate, save_checkpoint

import logging
def create_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')

    # 创建一个handler，用于写入日志文件
    file_handler = logging.FileHandler(
        #日志文件的地址
        filename="./log/初次尝试.log")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    return logger
#创建日志
logger = create_logger()
data_embed_collect=[]
label_collect=[]
def train(epoch, train_data, encoder, optimizer,criterion):
    train_avg_acc = 0
    train_avg_p = 0
    train_avg_r = 0
    train_avg_f1 = 0


    # Ensure dropout layers are in train mode
    encoder.train()


    # Batches
    for i_batch, (input_variable, lengths, target_variable) in enumerate(train_data):
        # Zero gradients
        optimizer.zero_grad()

        # Set device options
        input_variable = input_variable.to(device)
        lengths = lengths.to(device)
        target_variable = target_variable.to(device)

        #正式输入模型
        outputs= encoder(input_variable, lengths)


        loss = 0
        acc = 0
        recall=0
        precious=0
        f1=0
        #梯度累加一般用来解决内存溢出问题。设置x个批次做一次梯度更新。则每个批次得到的loss就需要除以x
        #损失、均值是每种标签上的平均数
        for idx, _ in enumerate(label_names):
            #使用交叉熵损失，计算每个类别上的损失，再求均值.
            # outputs是[batch size, num_classes, num_labels]。即[batch size, 4, 20]。target_variable是batch_size,
            # num_calss，即bs,20
            #第一轮中计算出了**该批次上**、**第一类上**的损失值（idx初始为0）再除类别数.....把每一个类别的损失值（除了类别数的）相加。得到的就是20个类别上的平均损失
            #/len(labels)是梯度正则化
            a=criterion(outputs[:, :, idx], target_variable[:, idx]) / len(label_names)
            loss += a
            a2,p,r,f= accuracy(outputs[:, :, idx], target_variable[:, idx],idx)

            acc+=a2/ len(label_names)
            recall +=r/ len(label_names)
            precious +=p/ len(label_names)
            f1 +=f/ len(label_names)

        loss.backward()

        optimizer.step()

        # logger.info status
        if i_batch % print_every == 0:
            #val、avg（实际值、均值）
            logger.info('[{0}] 当前轮数、批次数、总批次: [{1}][{2}/{3}]\t'.format(timestamp(), epoch, i_batch, len(train_data)))
            logger.info(f"本批次的训练损失是{loss}")
            logger.info(r"本批次的训练集准确率、macro-p\r\f是{}、{}、{}、{}".format(acc, precious, recall, f1))

        train_avg_acc += acc / len(train_data)
        train_avg_p += precious / len(train_data)
        train_avg_r += recall / len(train_data)
        train_avg_f1 += f1 / len(train_data)
    logger.info("这一轮训练集的准确率、p、r、f1是{}、{}、{}、{}".format(train_avg_acc,train_avg_p,train_avg_r,train_avg_f1))



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
    #词频文件长度
    train_data = SaDataset('D:\Pythonnnn\弱监督实验\原始数据\红心柚\训练集.csv', voc)
    # train_data = SaDataset('D:\Pythonnnn\弱监督的农业社会化销售服务评价\弱监督\伪标签数据集.csv', voc)

    val_data = SaDataset(r'D:\Pythonnnn\弱监督实验\原始数据\红心柚\测试集.csv', voc)

    #创建模型实例。参数1是input_size，它其实就是vocab.txt的长度。它决定了embedding层的行
    encoder = EncoderRNN(voc.n_words, hidden_size, encoder_n_layers, dropout)

    encoder = encoder.to(device)

    optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    # criterion = nn.CrossEntropyLoss().to(device)
    #分别对应类别0、1、2
    criterion=MultiDSCLoss(alpha=0.55)

    best_acc = 0
    #动态学习率与早停法
    #epochs_since_improvement表示连续几个epoch没有改善
    epochs_since_improvement = 0

    # Epochs
    for epoch in range(start_epoch, epochs):
        #如果连续 8 个 epoch 没有改善，则衰减学习率，并在 20 个 epoch 后终止训练
        if epochs_since_improvement == 20:
            break
        if epochs_since_improvement > 0 and epochs_since_improvement % 8 == 0:
            #原学习率会被乘上0.8
            adjust_learning_rate(optimizer, 0.8)

        # One epoch's training
        train(epoch, train_data, encoder, optimizer,criterion)
        logger.info("\n\n以下是测试集\n\n")

        # One epoch's validation
        val_acc = valid(val_data, encoder,criterion)


        #更新验证集最佳准确率
        # is_best = val_acc > best_acc
        # best_acc = max(best_acc, val_acc)
        #
        # if not is_best:
        #     epochs_since_improvement += 1
        #     logger.info("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        # else:
        #     epochs_since_improvement = 0

        # Save checkpoint
        save_checkpoint(epoch, encoder, optimizer, val_acc, is_best)

        # Reshuffle samples
        #打乱样本
        np.random.shuffle(train_data.samples)
        np.random.shuffle(val_data.samples)
    #存储最后一轮的模型
    # is_best=True
    # save_checkpoint(epoch, encoder, optimizer, val_acc)


if __name__ == '__main__':
    main()
