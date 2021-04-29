# -*- coding: utf-8 -*-


import torch
from utils import *
from config.base_config import BaseConfig



class TextCNNConfig(object):
    """配置参数"""
    def __init__(self):
        self.base_config = BaseConfig()
        self.label2idx = load_json(self.base_config.label2idx_path)
        self.word2idx = load_json(self.base_config.word2idx_path)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')   # 设备
        self.dropout = 0.5                                              # 随机失活
        self.num_classes = len(self.label2idx)                          # 类别数
        self.n_vocab = self.base_config.vocab_size                      # 词表大小
        self.num_epochs = 10                                            # epoch数
        self.batch_size = 128                                           # mini-batch大小
        self.learning_rate = 1e-3                                       # 学习率
        self.embed_size = 300                                           # embed_size
        self.filter_sizes = (2, 3, 4)                                   # 卷积核尺寸
        self.num_filters = 128                                          # 卷积核数量(channels数)
        self.embedding_pretrained = None                                # 预训练 word embedding
        self.model_path = "./model/textcnn.pth"                         # 模型保存路径
        self.log_path = "./log/train_textcnn.log"                       # 训练log