#coding:utf-8
import json
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from torchcrf import CRF
from tqdm import tqdm
import torchtext
from torchtext.data import Field, Example, TabularDataset
from torchtext.data import BucketIterator
# from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import f1_score
import os

#全局初始化配置参数。固定随机种子
seed=1

####
# 按行读取文件，返回为list类型
#
# 参数{string}: path 文件路径
# 参数{string}: t_code 读取文件编码形式
#
# 返回{list}: result  
####
def read_by_lines(path, t_code="utf-8"):
    """read the data by line"""
    result = list()
    with open(path, "r", encoding=t_code) as infile:
        for line in infile:
            result.append(line.strip())
    return result

####
# 将data数据写入文件
#
# 参数{string}: path 文件路径
# 参数{list}：data 要存储的数据
# 参数{string}: t_code 读取文件编码形式
####
def write_by_lines(path, data, t_code="utf-8"):
    """write the data"""
    with open(path, "w", encoding=t_code) as outfile:
        [outfile.write(d + "\n") for d in data]

## 生成触发词识别模型标签
schema_path = "datasets/event_schema.json"
trigger_save_path = "temp/vocab_trigger_label_map.txt"
event_types = set()
for line in read_by_lines(schema_path):
    d_json = json.loads(line)
    event_types.add(d_json["event_type"])
outputs = []
index = 0
for et in list(event_types):
    outputs.append(u"B-{}\t{}".format(et, index))
    index += 1
    outputs.append(u"I-{}\t{}".format(et, index))
    index += 1
outputs.append(u"O\t{}".format(index))
print(u"include event type {},  create label {}".format(
    len(event_types), len(outputs)))
write_by_lines(trigger_save_path, outputs)


## 论元角色识别模型标签
role_save_path = "temp/vocab_roles_label_map.txt"
index = 0
roles = set()
for line in read_by_lines(schema_path):
    d_json = json.loads(line)
    for role in d_json["role_list"]:
        roles.add(role["role"])
outputs = []
for r in list(roles):
    outputs.append(u"B-{}\t{}".format(r, index))
    index += 1
    outputs.append(u"I-{}\t{}".format(r, index))
    index += 1
outputs.append(u"O\t{}".format(index))
print(u"include roles {}，create label {}".format(len(roles), len(outputs)))
write_by_lines(role_save_path, outputs)

## 生成trigger标签映射id的变量
trigger_labels_map, role_labels_map = {}, {}
for line in read_by_lines(trigger_save_path):
    arr = line.split("\t")
    trigger_labels_map[arr[0]] = int(arr[1])
    
for line in read_by_lines(role_save_path):
    arr = line.split("\t")
    role_labels_map[arr[0]] = int(arr[1])

####
# 预处理文本，处理部分噪声数据
#
# 参数{string}: text
#
# 返回{list}: data
####
def clean_text(text):
    text = text.lower()
    text = text.replace("<br />", "")
    return [
                u"，" if t == u" " or t == u"\n" or t == u"\t" else t
                for t in list(text)
            ]

####
# 对初始化后的label赋值
# 
# 参数{list}： data
# 参数{int}： start
# 参数{int}： l
# 参数{string}： _type
#
# 返回{list} data
###
def label_data(data, start, l, _type, s=""):
    """label_data"""
    for i in range(start, start + l):
        suffix = u"B-" if i == start else u"I-"
        if i >= len(data):
            print(data, start, _type, s)
        data[i] = u"{}{}".format(suffix, _type)
    return data

####
# 切分原始数据集为trigger模型和role模型所需数据集合
#
# 参数{string}: path
#
# 返回{list, list}
####
def split_data(path):
    trigger_output = []
    role_output = []
    with open(path, 'r',encoding='utf-8') as f:
        for line in f:
            d_json = json.loads(line.strip())
            _id = d_json["id"]
            text_a = clean_text(d_json["text"])
            ## 初始化事件标签为“O”
            labels = [u"O"] * len(text_a)
            for event in d_json["event_list"]:        ## 遍历数据中的事件列表
                event_type = event["event_type"]       ## 获取事件类型
                start = event["trigger_start_index"]   ## 获取事件触发词的索引值
                trigger = event["trigger"]             ## 获取事件触发词
                labels = label_data(labels, start,
                                    len(trigger), event_type, text_a)     ## 获取文本对应标签
            trigger_output.append(u"{}\t{}".format(u' '.join(text_a),
                                           u' '.join(labels)))            ## 将text和label组合存入数组
            ## 初始化角色标签为“O”
            labels = [u"O"] * len(text_a)
            for event in d_json["event_list"]:  ## 遍历每一个事件
                labels = [u"O"] * len(text_a)
                for arg in event["arguments"]:   ## 遍历事件论元
                    role_type = arg["role"]      ## 获取论元类型
                    argument = arg["argument"]   ## 获取论元实体
                    start = arg["argument_start_index"]   ## 获取论元在文本中的索引值
                    labels = label_data(labels, start,
                                        len(argument), role_type) # 获取论元在文本中的对应标签
            role_output.append(u"{}\t{}".format(u' '.join(text_a),
                                               u' '.join(labels))) # 组合文本和标签，存入数组
    return trigger_output, role_output
    
train_data_path = "datasets/train.json"
dev_data_path = "datasets/dev.json"
trigger_train_data, role_train_data = split_data(train_data_path)
trigger_dev_data, role_dev_data = split_data(dev_data_path)

print(role_dev_data[:3])

write_by_lines("./temp/trigger_train.tsv", trigger_train_data)
write_by_lines("./temp/role_train.tsv", role_train_data)
write_by_lines("./temp/trigger_dev.tsv", trigger_dev_data)
write_by_lines("./temp/role_dev.tsv", role_dev_data)

res = []
with open('./datasets/test1.json', 'r',encoding='utf-8') as f:
    for line in f:
        d_json = json.loads(line.strip())
        _t = d_json['text']
        _id = d_json['id']
        res.append(_id + '\t' + " ".join(_t))
write_by_lines("./temp/test.tsv", res)

max_len = 256
def x_tokenize(x):
    return [c for c in x.split(" ")]

def trigger_y_tokenize(y):
    return [int(trigger_labels_map[c]) for c in y.split(" ")]

def role_y_tokenize(y):
    return [int(role_labels_map[c]) for c in y.split(" ")]


TEXT = Field(sequential=True, tokenize=x_tokenize,
                     use_vocab=True, batch_first=True,
                     fix_length=max_len, 
                     eos_token=None, init_token=None,
                     include_lengths=True, pad_token=0)

LABEL = Field(sequential=True, tokenize=trigger_y_tokenize, use_vocab=False, batch_first=True, pad_token=130,fix_length=max_len)

R_LABEL = Field(sequential=True, tokenize=role_y_tokenize, use_vocab=False, batch_first=True, pad_token=130,fix_length=max_len)

ID = Field(sequential=False, use_vocab=True)

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

EMBEDDING_DIM = 5
HIDDEN_DIM = 4
vocab_size = len(TEXT.vocab.stoi)

class BiLSTM_CRF(nn.Module):

    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True, batch_first=True)
        self.crf = CRF(self.tagset_size, batch_first=True)

        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

    def forward(self, inputs): 
        embeds = self.word_embeds(inputs.text[0])
        batch_size = embeds.shape[0]
        hidden = (torch.randn(2, batch_size, self.hidden_dim // 2),
                    torch.randn(2, batch_size, self.hidden_dim // 2))
        lstm_out, hidden = self.lstm(embeds, hidden)
        
        emissions = self.hidden2tag(lstm_out)
        
        mask = []
        if hasattr(inputs, "label"):
            for i in range(batch_size):
                one_num = int(inputs.text[1][i])
                if one_num > 256:
                    one_num = 256
                mask_it = [True]*one_num + [False]*(max_len - one_num)
                mask.append(mask_it)
            mask = torch.tensor(mask)
            loss = -self.crf(emissions, inputs.label, mask = mask, reduction="mean")

            return emissions, loss
        else:
            return emissions

trigger_model = BiLSTM_CRF(vocab_size, trigger_labels_map, EMBEDDING_DIM, HIDDEN_DIM)
role_model = BiLSTM_CRF(vocab_size, role_labels_map, EMBEDDING_DIM, HIDDEN_DIM)

trigger_optimizer = optim.SGD(trigger_model.parameters(), lr=0.01, weight_decay=1e-4)
role_optimizer = optim.SGD(role_model.parameters(), lr=0.01, weight_decay=1e-4)

min_best = 9999
max_best = -9999

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
                break;
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

trigger_result = train(trigger_model, trigger_train_iter, trigger_dev_iter, trigger_optimizer, 6, 'trigger')

role_result = train(role_model, role_train_iter, role_dev_iter, role_optimizer, 6, 'role')

model_sgd = BiLSTM_CRF(vocab_size, trigger_labels_map, EMBEDDING_DIM, HIDDEN_DIM)
model_Momentum = BiLSTM_CRF(vocab_size, trigger_labels_map, EMBEDDING_DIM, HIDDEN_DIM)
model_RMSprop = BiLSTM_CRF(vocab_size, trigger_labels_map, EMBEDDING_DIM, HIDDEN_DIM)
model_Adam = BiLSTM_CRF(vocab_size, trigger_labels_map, EMBEDDING_DIM, HIDDEN_DIM)
## 实例化 优化器
LR = 0.1
opt_SGD=torch.optim.SGD(model_sgd.parameters(),lr=LR)
opt_Momentum=torch.optim.SGD(model_Momentum.parameters(),lr=LR,momentum=0.8)
opt_RMSprop=torch.optim.RMSprop(model_RMSprop.parameters(),lr=LR,alpha=0.9)
opt_Adam=torch.optim.Adam(model_Adam.parameters(),lr=LR,betas=(0.9,0.99))

res_sgd = train(model_sgd, trigger_train_iter, trigger_dev_iter, opt_SGD, 6, 'trigger_sgd')
res_Momenum = train(model_Momentum, trigger_train_iter, trigger_dev_iter, opt_Momentum, 6, 'trigger_Momenum')
res_RMSprop = train(model_RMSprop, trigger_train_iter, trigger_dev_iter, opt_RMSprop, 6, 'trigger_RMSprop')
res_Adam= train(model_Adam, trigger_train_iter, trigger_dev_iter, opt_Adam, 6, 'trigger_Adam')

model_RMSprop_1 = BiLSTM_CRF(vocab_size, trigger_labels_map, EMBEDDING_DIM, HIDDEN_DIM)
model_RMSprop_2 = BiLSTM_CRF(vocab_size, trigger_labels_map, EMBEDDING_DIM, HIDDEN_DIM)
model_RMSprop_3 = BiLSTM_CRF(vocab_size, trigger_labels_map, EMBEDDING_DIM, HIDDEN_DIM)
model_RMSprop_4 = BiLSTM_CRF(vocab_size, trigger_labels_map, EMBEDDING_DIM, HIDDEN_DIM)
LR = 0.1
opt_RMSprop_1=torch.optim.RMSprop(model_RMSprop_1.parameters(),lr=LR,alpha=0.9)
LR = 0.15
opt_RMSprop_2=torch.optim.RMSprop(model_RMSprop_2.parameters(),lr=LR,alpha=0.9)
LR = 0.2
opt_RMSprop_3=torch.optim.RMSprop(model_RMSprop_3.parameters(),lr=LR,alpha=0.9)
LR = 0.25
opt_RMSprop_4=torch.optim.RMSprop(model_RMSprop_4.parameters(),lr=LR,alpha=0.9)

res_1 = train(model_RMSprop_1, trigger_train_iter, trigger_dev_iter, opt_RMSprop_1, 6, 'lr=0.01')
res_2 = train(model_RMSprop_2, trigger_train_iter, trigger_dev_iter, opt_RMSprop_2, 6, 'lr=0.05')
res_3 = train(model_RMSprop_3, trigger_train_iter, trigger_dev_iter, opt_RMSprop_3, 6, 'lr=0.10')
res_4= train(model_RMSprop_4, trigger_train_iter, trigger_dev_iter, opt_RMSprop_4, 6, 'lr=0.20')

trigger_model.eval()
with torch.no_grad():
    next(iter(trigger_dev_iter))
    emissions, _ = trigger_model(next(iter(trigger_dev_iter)))
    res = trigger_model.crf.decode(emissions)
    print(res[0][:10])


val_y = next(iter(trigger_dev_iter)).label
print("验证集f1值：{}".format(f1_score(np.array(val_y).reshape(-1), np.array(res).reshape(-1), average='macro')))

trigger_save_path = "./temp/trigger_model.bin"
state = {'net': trigger_model.state_dict()}
torch.save(state, trigger_save_path)

role_save_path = "./temp/role_model.bin"
state = {'net': role_model.state_dict()}
torch.save(state, role_save_path)

# best_model_path = os.path.join("./datasets/models", 'best_model'+ '_' + str(1) + '.model')
checkpoint = torch.load(trigger_save_path)
trigger_model.load_state_dict(checkpoint['net'], strict=False)

checkpoint = torch.load(role_save_path)
role_model.load_state_dict(checkpoint['net'], strict=False)

# trigger 模型结果
test_iter.init_epoch()
emissions = trigger_model(next(iter(test_iter)))
trigger_res = trigger_model.crf.decode(emissions)
# role 模型结果
test_iter.init_epoch()
text = next(iter(test_iter)).text
emissions = role_model(next(iter(test_iter)))
role_res = role_model.crf.decode(emissions)
# 原句子
sentence = "".join([TEXT.vocab.itos[i] for i in text[0][0][:int(text[1][0])]])

trigger_labels_map_res = {}
for i in trigger_labels_map:
    trigger_labels_map_res[trigger_labels_map[i]] = i

role_labels_map_res = {}
for i in role_labels_map:
    role_labels_map_res[role_labels_map[i]] = i

trigger_res_label = [trigger_labels_map_res[i] for i in trigger_res[0]]
role_res_label = [role_labels_map_res[i] for i in role_res[0]]

t_trigger = {"text": ["北", "京", "时", "间", "8", "月", "8", "日", "，", "2", "0", "1", "9", "年", "国", "际", "泳", "联", "游", "泳", "世", "界", "杯", "济", "南", "站", "女", "子", "5", "0", "米", "仰", "泳", "决", "赛", "，", "中", "国", "选", "手", "刘", "湘", "游", "出", "2", "7", "秒", "3", "5", "的", "好", "成", "绩", "，", "夺", "冠", "的", "同", "时", "还", "打", "破", "了", "赛", "会", "纪", "录", "。", "然", "而", "，", "这", "一", "次", "的", "夺", "冠", "却", "让", "泳", "迷", "对", "于", "刘", "湘", "在", "不", "久", "前", "的", "光", "州", "游", "泳", "世", "锦", "赛", "上", "放", "弃", "了", "该", "项", "目", "的", "决", "定", "更", "加", "耿", "耿", "于", "怀", "。"], "labels": ["O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "B-竞赛行为-夺冠", "I-竞赛行为-夺冠", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O"]}

t_role = {"text": ["北", "京", "时", "间", "8", "月", "8", "日", "，", "2", "0", "1", "9", "年", "国", "际", "泳", "联", "游", "泳", "世", "界", "杯", "济", "南", "站", "女", "子", "5", "0", "米", "仰", "泳", "决", "赛", "，", "中", "国", "选", "手", "刘", "湘", "游", "出", "2", "7", "秒", "3", "5", "的", "好", "成", "绩", "，", "夺", "冠", "的", "同", "时", "还", "打", "破", "了", "赛", "会", "纪", "录", "。", "然", "而", "，", "这", "一", "次", "的", "夺", "冠", "却", "让", "泳", "迷", "对", "于", "刘", "湘", "在", "不", "久", "前", "的", "光", "州", "游", "泳", "世", "锦", "赛", "上", "放", "弃", "了", "该", "项", "目", "的", "决", "定", "更", "加", "耿", "耿", "于", "怀", "。"], "labels": ["B-时间", "I-时间", "I-时间", "I-时间", "I-时间", "I-时间", "I-时间", "I-时间", "O", "B-夺冠赛事", "I-夺冠赛事", "I-夺冠赛事", "I-夺冠赛事", "I-夺冠赛事", "I-夺冠赛事", "I-夺冠赛事", "I-夺冠赛事", "I-夺冠赛事", "I-夺冠赛事", "I-夺冠赛事", "I-夺冠赛事", "I-夺冠赛事", "I-夺冠赛事", "I-夺冠赛事", "I-夺冠赛事", "I-夺冠赛事", "I-夺冠赛事", "I-夺冠赛事", "I-夺冠赛事", "I-夺冠赛事", "I-夺冠赛事", "I-夺冠赛事", "I-夺冠赛事", "I-夺冠赛事", "I-夺冠赛事", "O", "B-冠军", "I-冠军", "I-冠军", "I-冠军", "I-冠军", "I-冠军", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O"]}


###
# 根据labels对text提取结果
# 
# 参数{string}：text
# 参数{list}：labels
#
# 返回提取后的结果
##
def extract_result(text, labels):
    """extract_result"""
    ret, is_start, cur_type = [], False, None
    labels = labels[:len(text)]
    for i, label in enumerate(labels):
        if label != u"O":
            _type = label[2:]
            if label.startswith(u"B-"):
                is_start = True 
                cur_type = _type
                ret.append({"start": i, "text": [text[i]], "type": _type})
            elif _type != cur_type:
                """
                # 如果是没有B-开头的，则不要这部分数据
                cur_type = None
                is_start = False
                """
                cur_type = _type
                is_start = True
                ret.append({"start": i, "text": [text[i]], "type": _type})
            elif is_start:
                ret[-1]["text"].append(text[i])
            else:
                cur_type = None
                is_start = False
        else:
            cur_type = None
            is_start = False
    return ret



## 获取事件类型对应的事件论元
schema_datas = read_by_lines(schema_path)
schema = {}
for s in schema_datas:
    d_json = json.loads(s)
    schema[d_json["event_type"]] = [r["role"] for r in d_json["role_list"]]

## 将属于事件类型的事件论元选取出来作为最后结果
r_ret = extract_result(t_role["text"], t_role["labels"])
role_ret={}
for r in r_ret:
    role_type = r["type"]
    if role_type not in role_ret:
        role_ret[role_type] = []
    
    start = r['start']
    end = start + len(u"".join(r["text"]))
    role_ret[role_type].append("".join(t_trigger["text"])[start:end])

t_ret = extract_result(t_trigger["text"], t_trigger['labels'])
pred_event_types = list(set([t["type"] for t in t_ret]))
event_list = []
for event_type in pred_event_types:
    role_list = schema[event_type]
    arguments = []
    for role_type, ags in role_ret.items():
        if role_type not in role_list:
            continue
        for arg in ags:
            if len(arg) == 1:
                # 一点小trick
                continue
            arguments.append({"role": role_type, "argument": arg})
    event = {"event_type": event_type, "arguments": arguments}
    event_list.append(event)

print({ 
   "text": "".join(t_trigger["text"]),
    "event_list": event_list
}
)