# -*- coding: utf-8 -*-


import json
import torch
from torch.utils.data import Dataset
from config.kd_config import KDConfig
from data_helper_bert import encode_fn

config = KDConfig()


class KDdataset(Dataset):
    def __init__(self, data_path):
        label2idx = config.textcnn_config.label2idx
        word2idx = config.textcnn_config.word2idx
        texts = []
        self.labels = []
        with open(data_path) as f:
            for line in f:
                line = json.loads(line)
                text = line["sentence"]
                label = line["label"]
                texts.append(list(text))
                self.labels.append(label2idx[label])

        # 转成bert需要格式
        self.input_ids, self.token_type_ids, self.attention_mask = encode_fn(texts)

        # 转成textcnn需要格式
        self.cnn_ids = []
        for text in texts:
            words = [word2idx.get(w, 1) for w in text[:config.base_config.max_seq_len]]
            tmp = [0] * config.base_config.max_seq_len
            tmp[:len(words)] = words
            self.cnn_ids.append(tmp)

        self.cnn_ids = torch.tensor(self.cnn_ids)
        self.labels = torch.tensor(self.labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.cnn_ids[idx], self.labels[idx], self.input_ids[idx], \
               self.token_type_ids[idx], self.attention_mask[idx]


