"""图注意力标签信息网络+文本处理网络"""
#因为这个代码是把隐藏层相加而非concat，所以隐藏层维度不会变化，还是300.而标签注意力却是600了。这里的解决办法是，标签注意力的hidden_size不再*2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from config import num_labels, num_classes, batch_first,device


#input_size是词典的大小4052，hiddden_size是300
class GATsru(nn.Module):
    def __init__(self, input_size, hidden_size,adjacency, n_layers=3, dropout=0,heads=4):
        super(GATsru, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        # 定义一个embedding层，是一个长方形。参数1指定有几行，参数2指定有几列
        # 用它来词嵌入。每一个ids，都会映射到embedding的一行上。
        # 传入参数时，input_size即词频文件的长度。也就是每一个ids都会有一个对应的专属词嵌入向量
        # 它是随机的，不像word2vec，相似的词词向量也相近
        self.embedding = nn.Embedding(input_size, hidden_size)

        # Initialize GRU; the input_size and hidden_size params are both set to 'hidden_size'
        #   because our input size is a word embedding with number of features == hidden_size
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers,
                          dropout=(0 if n_layers == 1 else dropout), bidirectional=True)
        self.fc = nn.Linear(hidden_size, num_labels * num_classes)


        # 用几个注意力头
        self.heads = heads
        # 两个线性层。
        # input_dim是输入的词向量的维度，相当于模型内部的hidden_dim。该模型中h_s等同于input_size了
        input_dim=300
        # self.transform_dim1 = nn.Linear(input_dim, hidden_size * 2, bias=False)
        # self.transform_dim2 = nn.Linear(hidden_size * 2, hidden_size * 2, bias=False)

        self.transform_dim1 = nn.Linear(input_dim, hidden_size, bias=False)
        self.transform_dim2 = nn.Linear(hidden_size, hidden_size, bias=False)

        self.transform_dimensions = [self.transform_dim1, self.transform_dim2]

        #words and labels dropout
        self.dropout = nn.Dropout(dropout)

        self.fc2 = nn.Linear(num_labels, num_labels * num_classes)
        slope = 0.01
        self.activation = nn.LeakyReLU(slope)

        self.adjacency = nn.Parameter(adjacency)
        # self.edge_weights = nn.Linear(hidden_size * 2 * 2, 1, bias=False)
        self.edge_weights = nn.Linear(hidden_size * 2, 1, bias=False)
        self.tanh = nn.Tanh()





    # input_seq：padding后的评论的ids
    # input_lengths每个评论的实际长度
    #添加了一个标签嵌入
    def forward(self, input_seq, input_lengths,label_embedding, hidden=None):
        # 两个注意力层
        for td in self.transform_dimensions:  # Two Multiheaded GAT layers
            outputs = []
            for head in range(self.heads):

                label_embed = td(label_embedding)

                # print(label_embedding.shape)#20，300
                # print(label_embed.shape)


                n, embed_size = label_embed.shape

                label_embed_combinations = label_embed.unsqueeze(1).expand(-1, n, -1)
                label_embed_combinations = torch.cat(
                    [label_embed_combinations, label_embed.unsqueeze(0).expand(n, -1, -1)], dim=2)
                e = self.activation(self.edge_weights(label_embed_combinations).squeeze(2))

                attention_coefficients = self.tanh(torch.mul(e, self.adjacency))

                new_h = torch.matmul(attention_coefficients.to(label_embed.dtype), label_embed)
                outputs.append(new_h)
            outputs = self.activation(torch.mean(torch.stack(outputs, dim=0), dim=0))

            label_embedding = outputs

            attention_features = self.dropout(label_embedding)
            # transpose表示转置
            attention_features = attention_features.transpose(0, 1)







        # input_seq = [sent len, batch size]
        # Convert word indexes to embeddings
        embedded = self.embedding(input_seq)
        # embedded = [sent len, batch size, hidden size]
        # Pack padded batch of sequences for RNN module
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths.cpu())
        # Forward pass through GRU
        outputs, hidden = self.gru(packed, hidden)
        # Unpack padding
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs)
        # Sum bidirectional GRU outputs
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]
        # MGnet是省略了seq_len，它是一个句子对应一个向量。这里是一个词对应一个向量
        # outputs = [sent len, batch size, hidden size]
        # outputs = outputs[-1]

        # idx:[batch_size,hiden_size]
        # 提取每个示例的最后一个时间步长的输出。input_lengths是一维列表，长度等于batch_size
        idx = (input_lengths - 1).view(-1, 1).expand(
            len(input_lengths), outputs.size(2))

        time_dimension = 1 if batch_first else 0
        # 增加一个维度。加在最高位还是最低维由batch_first决定。因为config中batch_first是false，所以在最高位加了一层
        idx = idx.unsqueeze(time_dimension)  # [1,batch_size,hidden_size]

        # gather是对张量取值的。第一个参数是规定按列取还是按行，第二个参数是对于每一行要取的数据的索引
        # Shape: (batch_size, rnn_hidden_dim)
        outputs = outputs.gather(
            time_dimension, Variable(idx)).squeeze(time_dimension)

        #32*200与200*8
        predicted_labels = torch.matmul(outputs, attention_features)

        #这个线性层的作用
        # outputs = [batch size, hidden size]==>outputs = [batch size, num_labels * num_classes]
        # outputs = self.fc(outputs)

        outputs = self.fc2(predicted_labels)




        # outputs = [batch size, num_labels * num_classes]
        outputs = outputs.view((-1, num_classes, num_labels))
        # outputs = [batch size, num_classes, num_labels]即[batch_size,4,20]

        # 它是softmax的改进
        # 1，对数运算时求导更容易，加快了反向传播的速度。2，解决Softmax可能存在的上溢和下溢的问题。
        # 因为是dim=1，所以是中间那个维度做了softmax操作
        outputs = F.log_softmax(outputs, dim=1)
        # print(outputs)
        # [25,4,20]
        # outputs = [batch size, num_classes, num_labels]

        return outputs
