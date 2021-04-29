# -*- coding: utf-8 -*-



class BaseConfig():
    def __init__(self):
        self.train_data_path = "./data/train.json"
        self.test_data_path = "./data/test.json"
        self.dev_data_path = "./data/dev.json"
        self.label2idx_path = "./data/label2idx.json"
        self.word2idx_path = "./data/word2idx.json"
        self.vocab_size = 5000
        self.max_seq_len = 100