# -*- coding:utf-8 -*-
# 测试部分
import json
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from torchcrf import CRF
from tqdm import tqdm
import torchtext
# from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import f1_score
import os
from train import trigger_model,role_model,test_iter,TEXT,trigger_labels_map, role_labels_map 
from preproccess import read_by_lines

trigger_save_path = "./temp/trigger_model.pth"
checkpoint = torch.load(trigger_save_path)
trigger_model.load_state_dict(checkpoint['net'], strict=False)

role_save_path = "./temp/role_model.pth"
checkpoint = torch.load(role_save_path)
role_model.load_state_dict(checkpoint['net'], strict=False)

# test_iter = torchtext.data.Iterator(test, batch_size=1, train=False, sort=False)
# trigger 模型结果
test_iter.init_epoch()
emissions = trigger_model(next(iter(test_iter)))
trigger_res = trigger_model.crf.decode(emissions)
# role 模型结果
test_iter.init_epoch()
text = next(iter(test_iter)).text
emissions = role_model(next(iter(test_iter)))
role_res = role_model.crf.decode(emissions)

trigger_labels_map_res = {}
for i in trigger_labels_map:
    trigger_labels_map_res[trigger_labels_map[i]] = i

role_labels_map_res = {}
for i in role_labels_map:
    role_labels_map_res[role_labels_map[i]] = i

trigger_res_label = [trigger_labels_map_res[i] for i in trigger_res[0]]
role_res_label = [role_labels_map_res[i] for i in role_res[0]]

t_trigger = {"text": ["北", "京", "时", "间", "8", "月", "8", "日", "，", "2", "0", "1", "9", "年", "国", "际", "泳", "联", "游", "泳", "世", "界", "杯", "济", "南", "站", "女", "子", "5", "0", "米", "仰", "泳", "决", "赛", "，", "中", "国", "选", "手", "刘", "湘", "游", "出", "2", "7", "秒", "3", "5", "的", "好", "成", "绩", "，", "夺", "冠", "的", "同", "时", "还", "打", "破", "了", "赛", "会", "纪", "录", "。", "然", "而", "，", "这", "一", "次", "的", "夺", "冠", "却", "让", "泳", "迷", "对", "于", "刘", "湘", "在", "不", "久", "前", "的", "光", "州", "游", "泳", "世", "锦", "赛", "上", "放", "弃", "了", "该", "项", "目", "的", "决", "定", "更", "加", "耿", "耿", "于", "怀", "。"], "labels": ["O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "B-竞赛行为-夺冠", "I-竞赛行为-夺冠", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O","O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O"]}

t_role = {"text": ["北", "京", "时", "间", "8", "月", "8", "日", "，", "2", "0", "1", "9", "年", "国", "际", "泳", "联", "游", "泳", "世", "界", "杯", "济", "南", "站", "女", "子", "5", "0", "米", "仰", "泳", "决", "赛", "，", "中", "国", "选", "手", "刘", "湘","游", "出", "2", "7", "秒", "3", "5", "的", "好", "成", "绩", "，", "夺", "冠", "的", "同", "时", "还", "打", "破", "了", "赛", "会", "纪", "录", "。", "然", "而", "，", "这", "一", "次", "的", "夺", "冠", "却", "让", "泳", "迷", "对", "于", "刘", "湘", "在", "不", "久", "前", "的", "光", "州", "游", "泳", "世", "锦", "赛", "上", "放", "弃", "了", "该", "项", "目", "的", "决", "定", "更", "加", "耿", "耿", "于", "怀", "。"], "labels": ["B-时间", "I-时间", "I-时间", "I-时间", "I-时间", "I-时间", "I-时间", "I-时间", "O", "B-夺冠赛事", "I-夺冠赛事", "I-夺冠赛事", "I-夺冠赛事", "I-夺冠赛事", "I-夺冠赛事", "I-夺冠赛事", "I-夺冠赛事", "I-夺冠赛事", "I-夺冠赛事", "I-夺冠赛事", "I-夺冠赛事", "I-夺冠赛事", "I-夺冠赛事", "I-夺冠赛事", "I-夺冠赛事", "I-夺冠赛事", "I-夺冠赛事", "I-夺冠赛事", "I-夺冠赛事", "I-夺冠赛事", "I-夺冠赛事", "I-夺冠赛事", "I-夺冠赛事", "I-夺冠赛事", "I-夺冠赛事", "O", "B-冠军", "I-冠军", "I-冠军", "I-冠军", "I-冠军", "I-冠军","O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O"]}


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


schema_path = "datasets/event_schema.json"
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
                continue
            arguments.append({"role": role_type, "argument": arg})
    event = {"event_type": event_type, "arguments": arguments}
    event_list.append(event)

print({ 
   "text": "".join(t_trigger["text"]),
    "event_list": event_list
}
)