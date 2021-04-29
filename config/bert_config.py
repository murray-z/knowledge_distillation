# -*- coding: utf-8 -*-


import torch
from utils import *
from config.base_config import BaseConfig


class BertConfig():
    def __init__(self):
        self.base_config = BaseConfig()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.label2idx = load_json(self.base_config.label2idx_path)
        self.class_num = len(self.label2idx)
        self.epochs = 4
        self.lr = 2e-5
        self.hidden_size = 768
        self.dropout = 0.1
        self.batch_size = 32
        self.model_path = "./model/bert.pth"
        self.log_path = "./log/train_bert.log"