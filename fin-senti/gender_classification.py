#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Time        : 2017/01/22 下午6:52
# @Author      : Zoe
# @File        : gender_classification.py
# @Description : 性别分类测试

import nltk
from nltk.corpus import names
import random

# print([name for name in names.words('male.txt')])

# 特征提取器函数
def gender_features(word):
    return 【？？？】


# 准备数据
labeled_names = (【？？？】)
random.shuffle(labeled_names)

# 调用特征提取器函数，得到数据特征集
featuresets = [【？？？】 for (n, gender) in labeled_names]

# 将数据划分训练集与测试集，训练朴素贝叶斯分类器
train_set, test_set = 【？？？】, 【？？？】
classifier = nltk.【？？？】.train(train_set)

# 测试"Neo"为男名还是女名
print(classifier.classify(【？？？】))
# 计算分类准确率
print(nltk.classify.accuracy(【？？？】))
# 显示信息量最大的前五个特征
print(classifier.【？？？】(5))
