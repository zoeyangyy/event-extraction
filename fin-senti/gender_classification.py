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

import nltk
from nltk.corpus import names
import random


def gender_features(word):
    return  {'last_letter':word[-1]}


labeled_names = ([(name,'male') for name in names.words('male.txt')]+
                 [(name, 'female') for name in names.words('female.txt')])
random.shuffle(labeled_names)


featuresets = [(gender_features(n), gender) for (n, gender) in labeled_names]


train_set, test_set = featuresets[500:], featuresets[:500]
classifier = nltk.NaiveBayesClassifier.train(train_set)


print(classifier.classify(gender_features('Neo')))

print(nltk.classify.accuracy(classifier, test_set))

