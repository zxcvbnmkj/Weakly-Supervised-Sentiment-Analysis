import datetime
import json
import os
import time
from config import *
from sklearn.metrics import precision_recall_fscore_support


def encode_text(word_map, c):
    return [word_map.get(word, word_map['<unk>']) for word in c] + [word_map['<end>']]


class Lang:
    def __init__(self, filename):
        word_map = json.load(open(filename, 'r'))
        self.word2index = word_map
        self.index2word = {v: k for k, v in word_map.items()}
        self.n_words = len(word_map)


class AverageMeter(object):
    """跟踪指标的最新、平均值、总和和计数"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# 指数加权平均值
class ExpoAverageMeter(object):
    # Exponential指数的 Weighted Average Meter米
    def __init__(self, beta=0.9):
        self.reset()

    def reset(self):
        self.beta = 0.9
        #value。也就是一个标签上的真实准确率
        self.val = 0
        self.avg = 0
        self.count = 0

    #传入的是这个批次的所有标签上的平均准确率
    #按这个更新方法，准确率由原来的值和这一次的值组成。占比是9：1
    # def update(self, val):
    #     self.val = val
    #     #0.9*avg+0.1*val???
    #     self.avg = self.beta * self.avg + (1 - self.beta) * self.val

    def update(self, val):
        self.avg = val
        #0.9*avg+0.1*val???
        # self.avg = self.beta * self.avg + (1 - self.beta) * self.val

def adjust_learning_rate(optimizer, shrink_factor):
    """
    Shrinks learning rate by a specified factor.
    :param optimizer: optimizer whose learning rate must be shrunk.
    :param shrink_factor: factor in interval (0, 1) to multiply learning rate with.
    """

    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))


def ensure_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


def save_checkpoint(epoch, encoder, optimizer, val_acc, is_best):
    #确保config中定义的目录存在
    ensure_folder(save_folder)

    state = {'encoder': encoder,
             'optimizer': optimizer}

    if is_best:
        filename = '{0}/checkpoint_{1}_{2:.3f}.tar'.format(save_folder, epoch, val_acc)
        torch.save(state, filename)

        # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
        torch.save(state, '{}/BEST_checkpoint.tar'.format(save_folder))


def encode_text(word_map, c):
    return [word_map.get(word, word_map['<unk>']) for word in c] + [word_map['<end>']]
from cleanlab.multilabel_classification.filter import find_label_issues


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



def timestamp():
    return datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
