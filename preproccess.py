# 数据处理部分
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

####预处理数据集

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
  

if __name__ == "__main__":
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
    # print(u"include event type {},  create label {}".format(len(event_types), len(outputs)))
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
    # print(u"include roles {}，create label {}".format(len(roles), len(outputs)))
    # include event type 65,  create label 131
    # include roles 121，create label 243
    write_by_lines(role_save_path, outputs)

    ## 生成trigger标签映射id的变量
    trigger_labels_map, role_labels_map = {}, {}
    for line in read_by_lines(trigger_save_path):
        arr = line.split("\t")
        trigger_labels_map[arr[0]] = int(arr[1])
        
    for line in read_by_lines(role_save_path):
        arr = line.split("\t")
        role_labels_map[arr[0]] = int(arr[1])

    train_data_path = "datasets/train.json"
    dev_data_path = "datasets/dev.json"
    trigger_train_data, role_train_data = split_data(train_data_path)
    trigger_dev_data, role_dev_data = split_data(dev_data_path)
    # 保存数据集
    write_by_lines("./temp/trigger_train.tsv", trigger_train_data)
    write_by_lines("./temp/role_train.tsv", role_train_data)
    write_by_lines("./temp/trigger_dev.tsv", trigger_dev_data)
    write_by_lines("./temp/role_dev.tsv", role_dev_data)
    print("训练集处理完毕")
    # 测试数据从json格式转换为tsv格式。
    res = []
    with open('./datasets/test1.json', 'r',encoding='utf-8') as f:
        for line in f:
            d_json = json.loads(line.strip())
            _t = d_json['text']
            _id = d_json['id']
            res.append(_id + '\t' + " ".join(_t))
    write_by_lines("./temp/test.tsv", res)
    print("测试集处理完毕")