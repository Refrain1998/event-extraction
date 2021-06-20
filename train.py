# 训练部分
#coding:utf-8
import json
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from torchcrf import CRF
from tqdm import tqdm
import torchtext
from torchtext.data import Field, TabularDataset
from torchtext.data import BucketIterator
# from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import f1_score
import os
from preproccess import read_by_lines
from model import BiLSTM_CRF,max_len

def x_tokenize(x):
    return [c for c in x.split(" ")]

def trigger_y_tokenize(y):
    return [int(trigger_labels_map[c]) for c in y.split(" ")]

def role_y_tokenize(y):
    return [int(role_labels_map[c]) for c in y.split(" ")]

def save_best_model(model, v, mode=1, note=""):
    global min_best, max_best
    # 保存模型
    if (mode == 1 and min_best > v) or (mode == 2 and max_best < v):
        if mode == 1:
            min_best = v
        if mode == 2:
            max_best = v
        state = {'net': model.state_dict()}
        save_path = os.path.join("./temp", 'best_model'+ str(note) +'_' + str(1) + '.model')
        print(str(mode)," Save.......")
        torch.save(state, save_path)
        
def train(model, train_iter, dev_iter, optimizer, Epoch, note=""):
    result = {}
    result["train_loss"], result["dev_loss"], result["dev_f1"], result["lr"] = [], [], [], []  ### 记录训练过程
    global min_best, max_best
    min_best = 999
    max_best = -999
    n = 0
    for epoch in range(Epoch):  
        print("** epoch:", epoch)
        train_iter.init_epoch()
        _loss_sum = 0
        step = 0
        for inputs in iter(dev_iter):
            model.zero_grad()                       ## 梯度清零
            _, loss = model(inputs)               
            loss.backward()                         ## 向后传播计算梯度
            optimizer.step()                        ## 根据梯度更新网络参数

            ## 目前本文档仅作为示例，为了加快训练速度, 每个epoch仅使用前10个batch
            n += 1
            if n == 10:
                n = 0
                break
            step += 1
            _loss_sum += loss.detach().item()

        ## 记录训练过程中的loss变化
        result["train_loss"].append(_loss_sum/step)
        print("train_loss:", _loss_sum/step)
        step = 0
        model.eval()
        dev_iter.init_epoch()
        result['lr'].append(optimizer.param_groups[0]['lr'])
        
        ## 目前本文档仅作为示例，为了加快训练速度, 仅使用一组batch作为验证集
        with torch.no_grad():
            emissions, loss = model(next(iter(trigger_dev_iter)))
            loss_v = loss.detach().item()
            result["dev_loss"].append(loss_v)
            ## 在训练过程中，选取在验证集上loss表现最低的模型。
            save_best_model(model, loss_v, 1, note)

            res = model.crf.decode(emissions)
            val_y = next(iter(dev_iter)).label
            f1_v = f1_score(np.array(val_y).reshape(-1), np.array(res).reshape(-1), average='macro')
            result["dev_f1"].append(f1_v)
            ## 在训练过程中，选取在验证集上f1表现最高的模型。
            save_best_model(model, f1_v, 2, note)
            print("dev_loss:", loss_v)
            print("dev_f1:", f1_v)
        model.train()
    return result


trigger_save_path = "temp/vocab_trigger_label_map.txt"
role_save_path = "temp/vocab_roles_label_map.txt"
## 生成trigger标签映射id的变量
trigger_labels_map, role_labels_map = {}, {}
for line in read_by_lines(trigger_save_path):
    arr = line.split("\t")
    trigger_labels_map[arr[0]] = int(arr[1])
    
for line in read_by_lines(role_save_path):
    arr = line.split("\t")
    role_labels_map[arr[0]] = int(arr[1])


TEXT = Field(sequential=True, tokenize=x_tokenize,
                     use_vocab=True, batch_first=True,
                     fix_length=max_len, 
                     eos_token=None, init_token=None,
                     include_lengths=True, pad_token=0)

LABEL = Field(sequential=True, tokenize=trigger_y_tokenize, use_vocab=False, batch_first=True, pad_token=130,fix_length=max_len)

R_LABEL = Field(sequential=True, tokenize=role_y_tokenize, use_vocab=False, batch_first=True, pad_token=130,fix_length=max_len)

ID = Field(sequential=False, use_vocab=True)

'''
参数说明
sequential 类型boolean, 作用：是否为序列，一般文本都为True，标签为False
tokenize 类型: function， 作用: 文本处理，默认为str.split(), 这里对x和y分别自定义了处理函数。
use_vocab： 类型: boolean， 作用：是否建立词典
batch_first：类型: boolean， 作用：为True则返回Batch维度为(batch_size， 文本长度), False 则相反
fix_length：类型: int, 作用：固定文本的长度，长则截断，短则padding，可认为是静态padding；为None则按每个Batch内的最大长度进行动态padding。
eos_token：类型：str, 作用: 句子结束字符
init_token：类型：str, 作用: 句子开始字符
include_lengths：类型: boolean， 作用：是否返回句子的原始长度，一般为True，方便RNN使用。
pad_token：padding的字符，默认为”“, 这里因为原始数据已经转成了int类型，所以使用0。注意这里的pad_token要和你的词典vocab里的“”的Id保持一致，否则会影响后面词向量的读取。
'''

# 读取文件生成数据集
trigger_fields = [('text', TEXT), ('label', LABEL)]

trigger_train, trigger_dev = TabularDataset.splits(
    path='datasets/', format='tsv',
    train='trigger_train.tsv',
    validation='trigger_dev.tsv',
    fields=trigger_fields)

role_fields = [('text', TEXT), ('label', R_LABEL)]
role_train, role_dev = TabularDataset.splits(
    path='datasets/', format='tsv',
    train='role_train.tsv',
    validation='role_dev.tsv',
    fields=role_fields)

test_fields =  [('id', ID), ('text', TEXT)]
test = TabularDataset.splits(
    path='./temp/', format='tsv',
    test='test.tsv',
    fields=test_fields)[0]

# 构建词表
TEXT.build_vocab(trigger_train)
ID.build_vocab(test)

# 根据dataset生成对应的迭代器
trigger_train_iter, trigger_dev_iter = BucketIterator.splits((trigger_train, trigger_dev),
                                             batch_sizes=(8, 8, 1),
                                             device = torch.device("cpu"),
                                             sort_key=lambda x: len(x.text), 
                                             sort_within_batch=True,
                                             repeat=False)

role_train_iter, role_dev_iter = BucketIterator.splits((role_train, role_dev),
                                             batch_sizes=(8, 8, 1),
                                             device = torch.device("cpu"),
                                             sort_key=lambda x: len(x.text),
                                             sort_within_batch=True,
                                             repeat=False)
test_iter = torchtext.data.Iterator(test, batch_size=1, train=False, sort=False)


# 定义模型结构
# 定义一个双向lstm神经网络的事件抽取模型
# 为了加快计算，这里embedding维度我们设为5，隐藏层的维度为4，一个lstm层
# 使用tanh激活函数得到文本特征，然后使用一个crf层作为序列分类器。
EMBEDDING_DIM = 5
HIDDEN_DIM = 4
vocab_size = len(TEXT.vocab.stoi)

# 实例化模型
# trigger模型用于提取文本的事件触发词
# role模型用于提取文本的事件论元
trigger_model = BiLSTM_CRF(vocab_size, trigger_labels_map, EMBEDDING_DIM, HIDDEN_DIM)
role_model = BiLSTM_CRF(vocab_size, role_labels_map, EMBEDDING_DIM, HIDDEN_DIM)
# 设置优化器
trigger_optimizer = optim.SGD(trigger_model.parameters(), lr=0.01, weight_decay=1e-4)
role_optimizer = optim.SGD(role_model.parameters(), lr=0.01, weight_decay=1e-4)

min_best = 9999
max_best = -9999

if __name__ == "__main__":
    # 训练模型
    trigger_result = train(trigger_model, trigger_train_iter, trigger_dev_iter, trigger_optimizer, 6, 'trigger')
    role_result = train(role_model, role_train_iter, role_dev_iter, role_optimizer, 6, 'role')

    # 模型评估
    trigger_model.eval()
    with torch.no_grad():
        next(iter(trigger_dev_iter))
        emissions, _ = trigger_model(next(iter(trigger_dev_iter)))
        res = trigger_model.crf.decode(emissions)
        print(res[0][:10])


    val_y = next(iter(trigger_dev_iter)).label
    print("验证集f1值：{}".format(f1_score(np.array(val_y).reshape(-1), np.array(res).reshape(-1), average='macro')))

    # 模型保存
    trigger_save_path = "./temp/trigger_model.pth"
    state = {'net': trigger_model.state_dict()}
    torch.save(state, trigger_save_path)

    role_save_path = "./temp/role_model.pth"
    state = {'net': role_model.state_dict()}
    torch.save(state, role_save_path)

