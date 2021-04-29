# -*- coding: utf-8 -*-

import json
import logging


def dump_json(obj, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False)


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.loads(f.read())


def get_logger(log_path, name):
    logger = logging.getLogger(name)
    logger.setLevel(level=logging.INFO)
    Handler = logging.FileHandler(log_path, mode='w')
    Handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    Handler.setFormatter(formatter)
    logger.addHandler(Handler)
    return logger
