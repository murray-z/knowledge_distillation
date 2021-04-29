# -*- coding: utf-8 -*-


import torch
from torch.optim import Adam
import torch.nn as nn
import torch.nn.functional as F
from bert import Bert
from textcnn import TextCNN
from config.kd_config import KDConfig
from utils import get_logger
from data_helper_kd import KDdataset
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
import numpy as np


def loss_fn_kd(students_output, labels, teacher_outputs, T, alpha):
    KD_loss = nn.KLDivLoss()(F.log_softmax(students_output / T, dim=1),
                             F.softmax(teacher_outputs / T, dim=1)) * (alpha * T * T) + \
              F.cross_entropy(students_output, labels) * (1. - alpha)
    return KD_loss


def dev(model, data_loader, config):
    device = config.device
    idx2label = {idx: label for label, idx in config.textcnn_config.label2idx.items()}
    model.to(device)
    model.eval()
    pred_labels, true_labels = [], []
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            cnn_ids, labels, input_ids, token_type_ids, attention_mask = batch[0].to(device), batch[1].to(device), \
                                                                         batch[2].to(device), batch[3].to(device), \
                                                                         batch[4].to(device)
            logits = model(cnn_ids)
            preds = torch.argmax(logits, dim=1)
            pred_labels.extend(preds.cpu().tolist())
            true_labels.extend(labels.cpu().tolist())
    pred_labels = [idx2label[i] for i in pred_labels]
    true_labels = [idx2label[i] for i in true_labels]
    acc = sum([1 if p == t else 0 for p, t in zip(pred_labels, true_labels)]) * 1. / len(pred_labels)
    table = classification_report(true_labels, pred_labels)
    return acc, table


def train():
    config = KDConfig()

    logger = get_logger(config.log_path, "train_KD")

    device = config.device

    # 加载bert模型，作为teacher
    logger.info("load bert .....")
    bert = Bert(config.bert_config)
    bert.load_state_dict(torch.load(config.bert_config.model_path))
    bert.to(device)
    bert.eval()

    # 冻结bert参数
    for name, p in bert.named_parameters():
        p.requires_grad = False

    # 加载textcnn模型，作为student
    textcnn = TextCNN(config.textcnn_config)
    textcnn.to(device)
    textcnn.train()

    # 加载数据集
    logger.info("load train/dev data .....")
    train_loader = DataLoader(KDdataset(config.base_config.train_data_path), batch_size=config.batch_size, shuffle=True)
    dev_loader = DataLoader(KDdataset(config.base_config.dev_data_path), batch_size=config.batch_size, shuffle=False)

    optimizer = Adam(textcnn.parameters(), lr=config.lr)

    # 开始训练
    logger.info("start training .....")
    best_acc = 0.
    for epoch in range(config.epochs):
        for i, batch in enumerate(train_loader):
            cnn_ids, labels, input_ids, token_type_ids, attention_mask = batch[0].to(device), batch[1].to(device), \
                                                                         batch[2].to(device), batch[3].to(device), \
                                                                         batch[4].to(device)
            optimizer.zero_grad()
            students_output = textcnn(cnn_ids)
            teacher_output = bert(input_ids, token_type_ids, attention_mask)
            loss = loss_fn_kd(students_output, labels, teacher_output, config.T, config.alpha)
            loss.backward()
            optimizer.step()

            # 打印信息
            if i % 100 == 0:
                labels = labels.data.cpu().numpy()
                preds = torch.argmax(students_output, dim=1)
                preds = preds.data.cpu().numpy()
                acc = np.sum(preds == labels) * 1. / len(preds)
                logger.info("TRAIN: epoch: {} step: {} acc: {} loss: {} ".format(epoch + 1, i, acc, loss.item()))

        acc, table = dev(textcnn, dev_loader, config)

        logger.info("DEV: acc: {} ".format(acc))
        logger.info("DEV classification report: \n{}".format(table))

        if acc > best_acc:
            torch.save(textcnn.state_dict(), config.model_path)
            best_acc = acc

    logger.info("start testing ......")
    test_loader = DataLoader(KDdataset(config.base_config.test_data_path), batch_size=config.batch_size, shuffle=False)
    best_model = TextCNN(config.textcnn_config)
    best_model.load_state_dict(torch.load(config.model_path))
    acc, table = dev(best_model, test_loader, config)

    logger.info("TEST acc: {}".format(acc))
    logger.info("TEST classification report:\n{}".format(table))


if __name__ == "__main__":
    train()
