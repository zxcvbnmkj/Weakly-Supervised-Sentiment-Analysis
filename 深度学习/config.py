import torch

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device ='cpu'

# Configure training/optimization
learning_rate = 0.01
min_word_freq = 2
print_every = 1

#他其实就相当于batch_size了
chunk_size = 32
#数据集有6个维度，20项指标
num_labels = 8
num_classes = 3  # number of sentimental types（正面、负面、中性、未提及）
#存在哪个目录下
save_folder = './models'

# Configure models
start_epoch = 0
epochs = 20
# hidden_size = 500
hidden_size = 200
encoder_n_layers = 2
dropout = 0.05
batch_first = False


label_names = ['品质','味道','价格','分量','外观','物流','客服','粗粒度']


# Default word tokens
PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token
UNK_token = 3
