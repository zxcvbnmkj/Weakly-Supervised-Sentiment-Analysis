import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from config import num_labels, num_classes, batch_first
import cuda_functional


class OurModel(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=2, dropout=0.05):
        super(OurModel, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        #定义一个embedding层，是一个长方形。参数1指定有几行，参数2指定有几列
        #用它来词嵌入。每一个ids，都会映射到embedding的一行上。
        #传入参数时，input_size即词频文件的长度。也就是每一个ids都会有一个对应的专属词嵌入向量
        #它是随机的，不像word2vec，相似的词词向量也相近
        self.embedding = nn.Embedding(input_size, hidden_size)

        # Initialize GRU; the input_size and hidden_size params are both set to 'hidden_size'
        #   because our input size is a word embedding with number of features == hidden_size

        self.gru = nn.GRU(hidden_size, hidden_size, n_layers,
                          dropout=dropout, bidirectional=True)
        # self.sru = cuda_functional.SRU(hidden_size, hidden_size, n_layers,
        #                                dropout=dropout, bidirectional=True)
        self.fc = nn.Linear(hidden_size, num_labels * num_classes)

    #input_seq：padding后的评论的ids
    #input_lengths每个评论的实际长度
    def forward(self, input_seq, input_lengths, hidden=None):
        # input_seq = [sent len, batch size]
        # Convert word indexes to embeddings
        embedded = self.embedding(input_seq)
        # embedded = [sent len, batch size, hidden size]
        # Pack padded batch of sequences for RNN module

        #数据增强层
        dropout_prob = 0.05
        if self.training:
            #torch.rand 函数用于生成一个张量，其中的元素是从区间 [0, 1) 中均匀分布的随机数。
            #创建掩码。生成和embedded大小一致的矩阵。当矩阵数字小于dropout_prob就被置为0，相当于掩码。
            mask = torch.rand(embedded.size()) > dropout_prob
            embedded = embedded * mask

        #因为torch升级，所以该函数的参数2必须在cpu上。使用gpu会报错。应当加一个.cpu()
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths.cpu())
        # Forward pass through GRU
        outputs, hidden = self.gru(packed, hidden)
        # Unpack padding
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs)
        # Sum bidirectional GRU outputs
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]
        # outputs = [sent len, batch size, hidden size]
        # outputs = outputs[-1]

        #idx:[batch_size,hiden_size]
        #提取每个示例的最后一个时间步长的输出。input_lengths是一维列表，长度等于batch_size
        idx = (input_lengths - 1).view(-1, 1).expand(
            len(input_lengths), outputs.size(2))


        time_dimension = 1 if batch_first else 0
        #增加一个维度。加在最高位还是最低维由batch_first决定。因为config中batch_first是false，所以在最高位加了一层
        idx = idx.unsqueeze(time_dimension)#[1,batch_size,hidden_size]

        #gather是对张量取值的。第一个参数是规定按列取还是按行，第二个参数是对于每一行要取的数据的索引
        # Shape: (batch_size, rnn_hidden_dim)
        outputs = outputs.gather(
            time_dimension, Variable(idx)).squeeze(time_dimension)


        # outputs = [batch size, hidden size]
        outputs = self.fc(outputs)

        # outputs = [batch size, num_labels * num_classes]
        outputs = outputs.view((-1, num_classes, num_labels))
        # outputs = [batch size, num_classes, num_labels]即[batch_size,4,20]

        #它是softmax的改进
        #1，对数运算时求导更容易，加快了反向传播的速度。2，解决Softmax可能存在的上溢和下溢的问题。
        #因为是dim=1，所以是中间那个维度做了softmax操作
        outputs = F.log_softmax(outputs, dim=1)
        # print(outputs)
        #[25,4,20]
        # outputs = [batch size, num_classes, num_labels]
        return outputs
