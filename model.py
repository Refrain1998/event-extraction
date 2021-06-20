# 模型代码
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
max_len = 256
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
