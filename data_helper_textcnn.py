# -*- coding: utf-8 -*-


import torch
import numpy as np
from torch.utils.data import Dataset
from utils import *
from config.textcnn_config import TextCNNConfig


config = TextCNNConfig()


class CnnDataSet(Dataset):
    def __init__(self, data_path):
        label2idx = config.label2idx
        word2idx = config.word2idx
        self.texts = []
        self.labels = []
        with open(data_path) as f:
            for line in f:
                line = json.loads(line)
                words = [word2idx.get(w, 1) for w in list(line["sentence"])[:config.base_config.max_seq_len]]
                self.labels.append(label2idx[line["label"]])
                tmp = [0] * config.base_config.max_seq_len
                tmp[:len(words)] = words
                self.texts.append(tmp)
        self.labels = np.array(self.labels)
        self.texts = np.array(self.texts)
        self.labels = torch.as_tensor(self.labels)
        self.texts = torch.as_tensor(self.texts)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        return self.texts[index], self.labels[index]







