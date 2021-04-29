# 概述
> 采用知识蒸馏，训练bert指导textcnn


# 运行
### 0. 配置config下文件
- 数据格式见：./data/data_sample.json

### 1. 数据预处理，获得word2idx, label2idx
- python preprocess_data.py

### 2. 训练bert
- python train_bert.py 

### 3. 训练textcnn(主要用于观察没有蒸馏时的性能)
- python train_textcnn.py

### 4. 训练蒸馏模型
- python train_KD.py