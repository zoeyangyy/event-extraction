#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Time        : 2018/1/24 下午5:03
# @Author      : Zoe
# @File        : fin-sentiment.py
# @Description : 金融新闻  / 情感词典 / 文本分类

import jieba
import nltk
import random
import collections

from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression

def LoadDict():
    # Stop word
    stop_words = [w.strip() for w in open('./Dict/stopWord.txt', 'r', encoding='GBK').readlines()]
    stop_words.extend(['\n', '\t', ' '])

    # Sentiment word
    pos_words = open('./Dict/pos_word.txt').readlines()
    pos_dict = {}
    for w in pos_words:
        word, score = w.strip().split(',')
        pos_dict[word] = float(score)
    pos_words_own = open('./Dict/pos_word_own.txt').readlines()
    for w in pos_words_own:
        pos_dict[w.strip()] = float(5)

    neg_words = open('./Dict/neg_word.txt').readlines()
    neg_dict = {}
    for w in neg_words:
        word, score = w.strip().split(',')
        neg_dict[word] = -float(score)
    neg_words_own = open('./Dict/neg_word_own.txt').readlines()
    for w in neg_words_own:
        neg_dict[w.strip()] = -float(5)

    # Deny word ['不', '没', '无', '非', '莫', '弗', '勿', '毋', '未', '否', '别', '無', '休', '难道']
    deny_words = open('./Dict/deny_word.txt').readlines()
    deny_dict = {}
    for w in deny_words:
        word = w.strip()
        deny_dict[word] = float(-1)

    # Degree word {'百分之百': 10.0, '倍加': 10.0, ...}
    degree_words = open('./Dict/degree_word.txt').readlines()
    degree_dict = {}
    for w in degree_words:
        word, score = w.strip().split(',')
        degree_dict[word] = float(score)

    return stop_words, pos_dict, neg_dict, deny_dict, degree_dict


def get_features(news):
    features = collections.defaultdict(int)
    score = 0
    news_list = news.split(',')
    features['num'] = len(news_list)
    for one in news_list:
        word_list = news_dict[int(one)]
        word_list = [word for word in word_list if word not in stop_words]
        degree = 1
        for word in word_list:
            if word in degree_dict:
                degree = degree_dict[word]
            if word in pos_dict:
                score += degree * pos_dict[word]
                degree = 1
                features['pos'] += 1
            if word in news_dict:
                score += degree * neg_dict[word]
                degree = 1
                features['neg'] += 1
            if word in deny_dict:
                features['deny'] = 1
    features['score'] = score

    return features


if __name__ == '__main__':
    stop_words, pos_dict, neg_dict, deny_dict, degree_dict = LoadDict()

    # load news file
    with open('news.txt', 'r') as inputFile:
        news = [eval(one) for one in inputFile.readlines()]
    news_dict = dict()
    for one in news:
        news_dict[one['id']] = jieba.cut(one['title'])

    # load training and testing file
    with open('train.txt', 'r') as inputFile:
        trainSet = [one.split() for one in inputFile.readlines()]
    with open('test.txt', 'r') as inputFile:
        testSet = [one.split() for one in inputFile.readlines()]

    train_set = [(get_features(news), label) for (label, news) in trainSet]
    test_set = [(get_features(news), label) for (label, news) in testSet]
    random.shuffle(train_set)


    classifier = nltk.NaiveBayesClassifier.train(train_set)
    print(nltk.classify.accuracy(classifier, test_set))

    # classifier = SklearnClassifier(BernoulliNB()).train(train_set)
    # print(nltk.classify.accuracy(classifier, test_set))
    #
    # classifier = SklearnClassifier(LogisticRegression()).train(train_set)
    # print(nltk.classify.accuracy(classifier, test_set))
    #
    # classifier = SklearnClassifier(SVC()).train(train_set)
    # print(nltk.classify.accuracy(classifier, test_set))
    #
    # classifier = SklearnClassifier(LinearSVC()).train(train_set)
    # print(nltk.classify.accuracy(classifier, test_set))
