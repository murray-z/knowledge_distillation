# -*- coding: utf-8 -*-


import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
from config.bert_config import BertConfig
from utils import *


tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
config = BertConfig()


def encode_fn(text_list):
    """将输入句子编码成BERT需要格式"""
    tokenizers = tokenizer(
        text_list,
        padding=True,
        truncation=True,
        max_length=config.base_config.max_seq_len,
        return_tensors='pt',  # 返回的类型为pytorch tensor
        is_split_into_words=True
    )
    input_ids = tokenizers['input_ids']
    token_type_ids = tokenizers['token_type_ids']
    attention_mask = tokenizers['attention_mask']
    return input_ids, token_type_ids, attention_mask


class BertDataSet(Dataset):
    def __init__(self, data_path):
        texts, labels = [], []
        label2idx = config.label2idx
        with open(data_path) as f:
            for idx, line in enumerate(f):
                line = json.loads(line)
                labels.append(label2idx[line["label"]])
                texts.append(line["sentence"].split())
        self.labels = torch.tensor(labels)
        self.input_ids, self.token_type_ids, self.attention_mask = encode_fn(texts)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.input_ids[index], self.token_type_ids[index], self.attention_mask[index], self.labels[index]

