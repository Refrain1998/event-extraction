# 对比试验部分
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
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import f1_score
from train import train,vocab_size,trigger_labels_map,role_labels_map,EMBEDDING_DIM,HIDDEN_DIM,trigger_train_iter,trigger_dev_iter
import os
from model import BiLSTM_CRF

# 对比实验一： 设置四种优化器SGD、Momentum、RMSProp、Adam更新网络参数，探究优化器对模型收敛的影响。
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

plt.figure()  # 初始化画布
N = [i for i in range(len(res_sgd['train_loss']))]  # 取横坐标的值
plt.plot(N, res_sgd['train_loss'], label='SGD_train_loss')  # 绘制训练loss的折线图
plt.scatter(N, res_sgd['train_loss'])  # 绘制训练loss的散点图

plt.plot(N, res_Momenum['train_loss'], label='Momenum_train_loss')  # 绘制训练loss的折线图
plt.scatter(N, res_Momenum['train_loss'])  # 绘制训练loss的散点图

plt.plot(N, res_RMSprop['train_loss'], label='RMSprop_train_loss')  # 绘制训练loss的折线图
plt.scatter(N, res_RMSprop['train_loss'])  # 绘制训练loss的散点图

plt.plot(N, res_Adam['train_loss'], label='Adam_train_loss')  # 绘制训练loss的折线图
plt.scatter(N, res_Adam['train_loss'])  # 绘制训练loss的散点图

plt.xlabel('Epoch #')  # 设置坐标轴名称
plt.ylabel('Loss')
plt.legend()
plt.savefig("./temp/train_loss_v1.jpg")
# plt.show() # 显示图片

# 对比实验二： 设置不同学习率，探究学习率对模型收敛的影响。
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

plt.figure()  # 初始化画布
N = [i for i in range(len(res_sgd['train_loss']))]  # 取横坐标的值
plt.plot(N, res_1['train_loss'], label='Momenum_lr_0.1')  # 绘制训练loss的折线图
plt.scatter(N, res_1['train_loss'])  # 绘制训练loss的散点图

plt.plot(N, res_2['train_loss'], label='Momenum_lr_0.15')  # 绘制训练loss的折线图
plt.scatter(N, res_2['train_loss'])  # 绘制训练loss的散点图

plt.plot(N, res_3['train_loss'], label='Momenum_lr_0.2')  # 绘制训练loss的折线图
plt.scatter(N, res_3['train_loss'])  # 绘制训练loss的散点图

plt.plot(N, res_4['train_loss'], label='Momenum_lr_0.25')  # 绘制训练loss的折线图
plt.scatter(N, res_4['train_loss'])  # 绘制训练loss的散点图

plt.xlabel('Epoch #')  # 设置坐标轴名称
plt.ylabel('Loss')
plt.legend()
plt.savefig("./temp/train_loss_v2.jpg")
# plt.show() # 显示图片