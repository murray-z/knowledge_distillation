# -*- coding: utf-8 -*-


import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AdamW
from bert import Bert
from config.bert_config import BertConfig
from data_helper_bert import BertDataSet
from sklearn.metrics import classification_report
from utils import get_logger


def dev(model, data_loader, config):
    device = config.device
    idx2label = {idx: label for label, idx in config.label2idx.items()}
    model.to(device)
    model.eval()
    pred_labels, true_labels = [], []
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            input_ids, token_type_ids, attention_mask, labels = batch[0].to(device), batch[1].to(device), batch[
                2].to(device), batch[3].to(device)
            logits = model(input_ids, token_type_ids, attention_mask)
            preds = torch.argmax(logits, dim=1)
            pred_labels.extend(preds.cpu().tolist())
            true_labels.extend(labels.cpu().tolist())
    pred_labels = [idx2label[i] for i in pred_labels]
    true_labels = [idx2label[i] for i in true_labels]
    acc = sum([1 if p == t else 0 for p, t in zip(pred_labels, true_labels)]) * 1. / len(pred_labels)
    table = classification_report(true_labels, pred_labels)
    return acc, table


def train():
    config = BertConfig()

    logger = get_logger(config.log_path)

    model = Bert(config)

    device = config.device

    train_dataset = BertDataSet(config.base_config.train_data_path)
    dev_dataset = BertDataSet(config.base_config.dev_data_path)

    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    dev_dataloader = DataLoader(dev_dataset, batch_size=config.batch_size, shuffle=False)

    optimizer = AdamW(model.parameters(), lr=config.lr)
    criterion = nn.CrossEntropyLoss()

    model.to(device)
    model.train()

    best_acc = 0.

    for epoch in range(config.epochs):
        for i, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            input_ids, token_type_ids, attention_mask, labels = batch[0].to(device), batch[1].to(device), batch[
                2].to(
                device), batch[3].to(device)
            logits = model(input_ids, token_type_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                preds = torch.argmax(logits, dim=1)
                acc = torch.sum(preds == labels)*1. / len(labels)
                logger.info("TRAIN: epoch: {} step: {} acc: {}, loss: {}".format(epoch, i, acc, loss.item()))

        acc, cls_report = dev(model, dev_dataloader, config)
        logger.info("DEV: epoch: {} acc: {}".format(epoch, acc))
        logger.info("DEV classification report:\n{}".format(cls_report))

        if acc > best_acc:
            torch.save(model.state_dict(), config.model_path)
            best_acc = acc

    test_dataset = BertDataSet(config.base_config.test_data_path)
    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    best_model = Bert(config)
    best_model.load_state_dict(torch.load(config.model_path))
    acc, cls_report = dev(best_model, test_dataloader, config)
    logger.info("TEST: ACC:{}".format(acc))
    logger.info("TEST classification report:\n{}".format(cls_report))


if __name__ == "__main__":
    train()
