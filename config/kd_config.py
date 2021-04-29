# -*- coding: utf-8 -*-

import torch
from config.base_config import BaseConfig
from config.bert_config import BertConfig
from config.textcnn_config import TextCNNConfig


class KDConfig():
    def __init__(self):
        self.base_config = BaseConfig()
        self.bert_config = BertConfig()
        self.textcnn_config = TextCNNConfig()
        self.batch_size = 128
        self.epochs = 10
        self.lr = 1e-3
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.T = 10            # 调整温度
        self.alpha = 0.9       # 调整soft_target loss 和 hard_target loss 比重
        self.model_path = "./model/textcnn_kd.pth"
        self.log_path = "./log/train_kd.log"