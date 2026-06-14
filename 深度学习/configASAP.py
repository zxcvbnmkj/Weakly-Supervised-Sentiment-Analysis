# -*- coding: utf-8 -*-
import torch
print("这里是config asap")
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device ='cpu'

# Configure training/optimization
learning_rate = 0.01
min_word_freq = 10
print_every = 50

#他其实就相当于batch_size了
chunk_size = 64
#数据集有6个维度，20项指标
num_labels = 18
num_classes = 4  # number of sentimental types（正面、负面、中性、未提及）
#存在哪个目录下
save_folder = './models'

# Configure models
start_epoch = 0
epochs = 1
# hidden_size = 500
hidden_size = 200
encoder_n_layers = 2
dropout = 0.05
batch_first = False


label_names =["Location#Transportation",'Location#Downtown','Location#Easy_to_find','Service#Queue','Service#Hospitality','Service#Parking','Service#Timely','Price#Level','Price#Cost_effective','Price#Discount','Ambience#Decoration','Ambience#Noise','Ambience#Space','Ambience#Sanitary','Food#Portion','Food#Taste','Food#Appearance','Food#Recommend']


# Default word tokens
PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token
UNK_token = 3
