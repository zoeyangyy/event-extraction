#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Time        : 2017/11/20 上午10:52
# @Author      : Zoe
# @File        : temp.py
# @Description :

import  matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import pylab
from matplotlib.font_manager import FontManager, FontProperties
from matplotlib.ticker import  MultipleLocator
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn import preprocessing
import nltk
import random
import re
import os
import collections
import json
import datetime
import time
import pickle
import tensorflow as tf
import csv
import math

def getChineseFont():
    return FontProperties(fname='/System/Library/Fonts/PingFang.ttc')
def getTimeFont():
    return FontProperties(fname='/System/Library/Fonts/Times.ttc')

def draw():
    x = [i for i in range(0, 70)]
    mp = []
    ap = []
    for i in x:
        mp.append(120000*i-3000*(i * i))
        ap.append(60000*i-1000*i*i)
    plt.plot(x, mp, label='mp')
    plt.plot(x, ap, label='ap')
    max_ap = max(ap)
    max_index = ap.index(max_ap)
    plt.plot(max_index, max_ap, 'm*')
    plt.legend()
    plt.grid(True)
    plt.show()

# draw()


def draw2():
    fig = plt.figure()
    ax = Axes3D(fig)

    X = np.arange(0, 400, 1)
    Y = np.arange(0, 400, 1)
    X, Y = np.meshgrid(X, Y)
    Z = 1200* (X ** 2)*Y - 3* (X**3)* (Y**2)

    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')
    plt.show()

# draw2()


def drawPillar():
    n_groups = 5
    means_men = (20, 35, 30, 35, 27)
    means_women = (25, 32, 34, 20, 25)

    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.35

    opacity = 0.4
    rects1 = plt.bar(index, means_men, bar_width, alpha=opacity, color='b', label='Men')
    rects2 = plt.bar(index + bar_width, means_women, bar_width, alpha=opacity, color='r', label='Women')

    plt.xlabel('Group')
    plt.ylabel('Scores')
    plt.title('Scores by group and gender')
    plt.xticks(index + bar_width, ('A', 'B', 'C', 'D', 'E'))
    plt.ylim(0, 40)
    plt.legend()

    plt.tight_layout()
    plt.show()


# drawPillar()


def result():
    with open('raw_file/tensor_result_y.txt', 'r') as f_result:
        a = f_result.readlines()
        real = list()
        for i in a:
            real.append(int(i.strip()))
        num = list()
        for _ in range(len(a)):
            num.append(random.randint(0, 216))
        print(real)
        print('****************************')
        print(num)
        right, count = 0, 0
        for index, n in enumerate(real):
            count += 1
            if n == num[index]:
                right += 1
        print(right/count)

        # 0.004806248122559327


def generat_class():
    with open('/Users/zoe/Documents/event_extraction/majorEventDump/Class.txt', 'r', encoding='utf8') as f_r:
        typeClass = collections.defaultdict(list)
        typeList = list()
        for line in f_r.readlines():
            if line != '\n':
                typeList.append(line.strip())
            else:
                typeClass[typeList[0]] = typeList[1:]
                typeList = list()
        # print(typeClass)

    with open('/Users/zoe/Documents/event_extraction/majorEventDump/Class.json', 'w', encoding='utf8') as f_w:
        json.dump(typeClass, f_w, indent=1, ensure_ascii=False)

# generat_class()


def compareTrainTest():
    with open('../data/majorEventAll.json', 'r',
              encoding='utf-8') as inputFile:
        events = json.load(inputFile)

    with open('../data/typeCodeDump.json', 'r',
              encoding='utf-8') as inputFile:
        code2type = json.load(inputFile)

    with open('../data/Class.json', 'r', encoding='utf8') as inputFile:
        typeClass = json.load(inputFile)

    typeDict = dict()
    typeList = [t for t in typeClass.keys()]

    for t in typeClass.keys():
        for c in typeClass[t]:
            typeDict[c] = typeList.index(t)

    # print(code2type)
    ### 获得按时间排序的公司事件链条

    # error = collections.defaultdict(int)
    countdict = collections.defaultdict(int)
    countdictTest = collections.defaultdict(int)
    dictTrain = collections.defaultdict(list)
    dictTest = collections.defaultdict(list)
    dictDev = collections.defaultdict(list)

    for event in events:
        company = event['S_INFO_WINDCODE']
        for s_event in event['event']:
            try:
                s_event['type'] = typeDict[code2type[s_event['type']]]
                new_event = dict()
                new_event['date'] = s_event['date']
                new_event['type'] = s_event['type']
                if datetime.datetime(2000, 1, 1) <= datetime.datetime.strptime(s_event['date'], '%Y%m%d') < datetime.datetime(2016, 1, 1):
                    # countdict[code2type[s_event['type']]] += 1
                    countdict[s_event['type']] += 1
                    dictTrain[company].append(new_event)
                elif datetime.datetime(2016, 1, 1) <= datetime.datetime.strptime(s_event['date'], '%Y%m%d') < datetime.datetime(2017, 1, 1):
                    # countdict[code2type[s_event['type']]] += 1
                    countdict[s_event['type']] += 1
                    dictDev[company].append(new_event)
                elif datetime.datetime.strptime(s_event['date'], '%Y%m%d') >= datetime.datetime(2017, 1, 1):
                    # countdictTest[code2type[s_event['type']]] += 1
                    # countdictTest[s_event['type']] += 1
                    countdict[s_event['type']] += 1
                    dictTest[company].append(new_event)
            except:
                # error[code2type[s_event['type']]] += 1
                print('error')

    with open('/Users/zoe/Documents/event_extraction/majorEventDump/TrainSet.json', 'w') as f_w:
        json.dump(dictTrain, f_w, indent=1)
    with open('/Users/zoe/Documents/event_extraction/majorEventDump/DevSet.json', 'w') as f_w:
        json.dump(dictDev, f_w, indent=1)
    with open('/Users/zoe/Documents/event_extraction/majorEventDump/TestSet.json', 'w') as f_w:
        json.dump(dictTest, f_w, indent=1)

    # print(error, len(error))
    # with open('temp.txt', 'w') as f_w:
    #     json.dump(error, f_w, ensure_ascii=False, indent=1)

    countdict = sorted(countdict.items(), key=lambda d:d[1], reverse=True)
    countdict = {typeList[one[0]]:one[1] for one in countdict}
    # print(countdict)
    for i in countdict:
        print(i)
    for i in countdict:
        print(countdict[i])


    # countdictTest = sorted(countdictTest.items(), key=lambda d:d[1], reverse=True)
    # countdictTest = {typeList[one[0]]:one[1] for one in countdictTest}
    # print(countdictTest)
    #
    # with open('/Users/zoe/Documents/event_extraction/majorEventDump/TrainClass_all_2000.json', 'w', encoding='utf8') as f_w:
    #     json.dump(countdict, f_w, ensure_ascii=False, indent=1)

    # with open('/Users/zoe/Documents/event_extraction/majorEventDump/TestClass_L.json', 'w', encoding='utf8') as f_w:
    #     json.dump(countdictTest, f_w, ensure_ascii=False, indent=1)

# compareTrainTest()

# 6-gram results:
# 107
# 85882
# 0.001245895531077525

def language_model():
    with open('/Users/zoe/Documents/event_extraction/majorEventDump/TrainSet.json', 'r',
              encoding='utf-8') as inputFile:
        eventsTrain = json.load(inputFile)
    with open('/Users/zoe/Documents/event_extraction/majorEventDump/DevSet.json', 'r',
              encoding='utf-8') as inputFile:
        eventsDev = json.load(inputFile)
    with open('/Users/zoe/Documents/event_extraction/majorEventDump/TestSet.json', 'r',
              encoding='utf-8') as inputFile:
        eventsTest = json.load(inputFile)

    freqDist = collections.defaultdict(int)
    pairFreqDist = collections.defaultdict(int)
    trigramFreqDist = collections.defaultdict(int)

    for company, eventSeq in eventsTrain.items():
        for beginIdx, e in enumerate(eventSeq[:-1]):
            # Start_Date = datetime.datetime.strptime(eventSeq[beginIdx-5]['date'], '%Y%m%d')
            # freqDist[(eventSeq[beginIdx]['type'],eventSeq[beginIdx+1]['type'],eventSeq[beginIdx+2]['type'],
            #           eventSeq[beginIdx+3]['type'],eventSeq[beginIdx+4]['type'],eventSeq[beginIdx+5]['type'])] +=1
            freqDist[(eventSeq[beginIdx]['type'], eventSeq[beginIdx+1]['type'])] += 1

    correct = 0
    chains =  0
    for company, eventSeq in eventsTest.items():
        for beginIdx, e in enumerate(eventSeq[:-1]):
            max = 0
            chains += 1
            for i in range(25):
                count = freqDist[(eventSeq[beginIdx]['type'], i)]
                if count > max:
                    max = count
                    pred = i
            if i == eventSeq[beginIdx+1]['type']:
                correct += 1
    print(correct)
    print(chains)
    print(correct/chains)

# language_model()

    # # save model    json key must be string
    # freqDist = dict((':'.join([str(i) for i in k]), v) for k, v in freqDist.items())
    # with open('/Users/zoe/Documents/event_extraction/majorEventDump/6gram.json', 'w', encoding='utf-8') as inputFile:
    #     json.dump(freqDist, inputFile, indent=1)

# language_model()


def nltkBayes():
    def gender_features(list):
        return {'one': list[0]}
    # , 'two': list[1], 'three': list[2], 'four': list[3], 'five': list[4]
    with open('/Users/zoe/Documents/event_extraction/majorEventDump/TrainSet.json', 'r',
              encoding='utf-8') as inputFile:
        eventsTrain = json.load(inputFile)
    with open('/Users/zoe/Documents/event_extraction/majorEventDump/TestSet.json', 'r',
              encoding='utf-8') as inputFile:
        eventsTest = json.load(inputFile)

    mat_list = list()
    mat = np.zeros(shape=(6))
    for company, eventSeq in eventsTrain.items():
        for beginIdx, e in enumerate(eventSeq[:-5]):
            for i in range(6):
                mat[i] = eventSeq[beginIdx+i]['type']
            mat_list.append(mat)
            mat = np.zeros(shape=(6))
    labeled_data = ([(one[0:5], one[5]) for one in mat_list])
    random.shuffle(labeled_data)


    y_list = list()
    mat = np.zeros(shape=(6))
    for company, eventSeq in eventsTest.items():
        for beginIdx, e in enumerate(eventSeq[:-5]):
            for i in range(6):
                mat[i] = eventSeq[beginIdx + i]['type']
            y_list.append(mat)
            mat = np.zeros(shape=(6))
    labeled_y = ([(one[0:5], one[5]) for one in y_list])
    random.shuffle(labeled_y)

    train_set = [(gender_features(n), label) for (n, label) in labeled_data]
    test_set = [(gender_features(n), label) for (n, label) in labeled_y]

    classifier = nltk.NaiveBayesClassifier.train(train_set)

    print(nltk.classify.accuracy(classifier, test_set))

    print(classifier.show_most_informative_features(5))

# 0.357839826739 只用一个事件
# 0.511934980555
# nltkBayes()


def plt_class_distribution():
    train = {
     "停牌": 631893,
     "股东大会": 419181,
     "业绩披露": 314574,
     "分红": 194256,
     "交易": 163237,
     "公司资料变更": 157890,
     "IPO": 39084,
     "股权质押": 36546,
     "增发": 29281,
     "股改": 26553,
     "资产交易": 14308,
     "重大项目": 11154,
     "配股": 10521,
     "红色预警": 10154,
     "对外投资": 8993,
     "财税政策": 5137,
     "发审委审核会议": 4637,
     "经营事件": 2597,
     "澄清传闻": 1568,
     "评级": 1285,
     "股权激励": 1109,
     "其他": 850,
     "换股": 592,
     "项目投资": 391,
     "突发事件": 29
    }
    test = {
     "停牌": 49029,
     "股东大会": 60859,
     "业绩披露": 40905,
     "分红": 20212,
     "交易": 22451,
     "公司资料变更": 19670,
     "IPO": 7462,
     "股权质押": 14753,
     "增发": 3656,
     "股改": 13,
     "资产交易": 4153,
     "重大项目": 2549,
     "配股": 147,
     "红色预警": 2334,
     "对外投资": 4854,
     "财税政策": 1177,
     "发审委审核会议": 284,
     "经营事件": 360,
     "澄清传闻": 312,
     "评级": 217,
     "股权激励": 379,
     "其他": 777,
     "换股": 10,
     "项目投资": 28,
     "突发事件": 0
    }

    n_groups = 25
    means_men = train.values()
    means_women = test.values()

    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.25

    opacity = 0.4
    rects1 = plt.bar(index, means_men, bar_width, alpha=opacity, color='b', label='train')
    rects2 = plt.bar(index + bar_width, means_women, bar_width, alpha=opacity, color='r', label='test')

    plt.xlabel('Group')
    plt.ylabel('#')
    plt.title('Class distribution of Data')
    plt.xticks(index + bar_width, train.keys(), fontproperties=getChineseFont(), rotation = 90)
    plt.legend()

    plt.tight_layout()
    # plt.show()
    plt.savefig('/Users/zoe/Documents/event_extraction/第一次报告/Class_distribution.png')


def time_test():
    # 64 * 128   64 * 128 * 25
    sess = tf.Session()
    alpha = {
        0: tf.Variable(tf.random_normal([128, 25])),
        1: tf.Variable(tf.random_normal([128, 25]))
    }

    alpha_mat = tf.constant(0.1)
    for index_i in range(2):
        alpha_mat_sub = tf.constant(0.1, shape=[3])
        for index_j in range(2):
            alpha_mat_sub = tf.concat([alpha_mat_sub, tf.squeeze([[1., 1., 1.]])], 0)
        alpha_mat_sub = tf.reshape(alpha_mat_sub[3:], [2, 3])
        alpha_mat = tf.add(alpha_mat, alpha_mat_sub)
        print(sess.run(alpha_mat_sub))


def draw_result():
    epoch = [one for one in range(1,82)]
    # baseline = [0.573659, 0.589987, 0.590476, 0.578782, 0.579058]
    # position = [0.559008, 0.58593, 0.591706, 0.590526, 0.590573]
    # time = [0.552299, 0.508444, 0.490502, 0.48826, 0.478486]
    # self = [0.0272559, 0.199495, 0.199624, 0.199512, 0.199754]
    # all = [0.552007, 0.59035, 0.591128, 0.587372, 0.592395]
    li = [0.037479,
0.545478,
0.538736,
0.552359,
0.553861,
0.550915,
0.564642,
0.545303,
0.546642,
0.559089,
0.541822,
0.561918,
0.545245,
0.560823,
0.558576,
0.570406,
0.558844,
0.564398,
0.567425,
0.569789,
0.562593,
0.563746,
0.569137,
0.568298,
0.569637,
0.56888,
0.567157,
0.569986,
0.567891,
0.568089,
0.569451,
0.56491,
0.569148,
0.569812,
0.567972,
0.568263,
0.566598,
0.565888,
0.56661,
0.565993,
0.573794,
0.566226,
0.567099,
0.567052,
0.568322,
0.568112,
0.564072,
0.572874,
0.567297,
0.571535,
0.567774,
0.567961,
0.561056,
0.568368,
0.570173,
0.566226,
0.567879,
0.56817,
0.571454,
0.565737,
0.572012,
0.567355,
0.571221,
0.563618,
0.571174,
0.566622,
0.566202,
0.570149,
0.568438,
0.560323,
0.566447,
0.563292,
0.563629,
0.568974,
0.569497,
0.56746,
0.568531,
0.570103,
0.569882,
0.571267,
0.569952]
    position = [float(line.strip()) for line in open('remote/testing_result.txt', 'r').readlines()]
    plt.ylabel('accuracy')
    plt.xlabel("epoch")
    plt.plot(epoch, li, color='grey', linestyle='solid', label='baseline')
    plt.plot(epoch, position, color='#FF9500', linestyle='solid', label='position')
    # plt.plot(epoch, time, color='#0C5DA5', linestyle='solid', label='time')
    # plt.plot(epoch, self, color='#00AC6B', linestyle='solid', label='self')
    # plt.plot(epoch, all, color='#E7003E', linestyle='solid', label='all')

    plt.title('testing result')
    plt.xticks(epoch)
    plt.legend()
    plt.show()

# draw_result()


def draw_result2():
    with open('remote/test_result_new.txt', 'r', encoding='utf8') as file:
        content = file.readlines()
    result = list()
    for line in content:
        try:
            result.append(float(line.strip()))
        except:
            pass

    epoch = [one for one in range(1, 98)]
    fig = plt.figure()
    plt.ylabel('accuracy')
    plt.xlabel("epoch")

    color_li = ['grey', '#FF9500', '#0C5DA5', '#00AC6B', '#E7003E', 'black', 'green', 'red', 'pink', 'blue']
    label_li = ['baseline mlp', 'position mlp', 'time mlp', 'event mlp', 'all mlp', 'baseline gcn', 'position gcn', 'time gcn', 'event gcn', 'all gcn']

    for i in range(10):
        plt.plot(epoch, result[65*i:65*(i+1)] + result[650+16*i:650+16*(i+1)] + result[650+160+16*i : 650+160+16*(i+1)], color=color_li[i], linestyle='solid', label=label_li[i])
        print(max(result[650+160+16*i : 650+160+16*(i+1)]))
    # plt.title('testing result')
    # ax = fig.add_subplot(111)
    # xticks = np.arange(16, 98, 16)
    # ax.set_xticks(xticks)  # 生成刻度
    # ax.set_xticklabels(np.arange(1, 7))  # 生成x轴标签
    # plt.ylim([0.50,0.60])
    # plt.legend()
    # plt.savefig("/Users/zoe/Documents/event_extraction/coling2018/picture/result.png")
    # plt.show()

# draw_result2()


def drawHist():
    n_groups = 4
    means_men = (0.527827, 0.536711, 0.541647, 0.534405)


    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.35

    opacity = 1
    rects1 = plt.bar(index, means_men, bar_width, alpha=opacity, color='grey', label='Men')
    # rects2 = plt.bar(index + bar_width, means_women, bar_width, alpha=opacity, color='r', label='Women')

    plt.xlabel('')
    plt.ylabel('accuracy')
    plt.title('Testing Results')
    plt.xticks(index, ('BiLSTM', 'position', 'time', 'event'))
    plt.ylim(0.4, 0.6)
    # plt.legend()

    plt.tight_layout()
    plt.show()

# drawHist()

def absolute_path():
    PROJECT_PATH = os.path.dirname(os.path.realpath(__file__))  # 获取项目根目录
    data_file_path = os.path.join(PROJECT_PATH, "Dict/stopWord.txt")  # 文件路径
    stop_words = [w.strip() for w in open(data_file_path, 'r', encoding='GBK').readlines()]
    stop_words.extend(['\n','\t',' '])


Chain_Lens = 1


def adjacency_matrix():
    global Chain_Lens
    regularization = 0.3

    with open('/Users/zoe/Documents/event_extraction/majorEventDump/TrainSet.json', 'r',
              encoding='utf-8') as inputFile:
        eventsTrain = json.load(inputFile)
    with open('/Users/zoe/Documents/event_extraction/majorEventDump/TestSet.json', 'r',
              encoding='utf-8') as inputFile:
        eventsTest = json.load(inputFile)

    adj_mat = collections.defaultdict()

    for i in range(25):
        adj_mat[i] = collections.defaultdict(int)

    # ********临近矩阵的生成********
    for company, eventSeq in eventsTrain.items():
        for beginIdx, e in enumerate(eventSeq[1:]):
            adj_mat[eventSeq[beginIdx]['type']][eventSeq[beginIdx+1]['type']] += 1

    pickle.dump(adj_mat, open('/Users/zoe/Documents/event_extraction/majorEventDump/adjacency.data', 'wb'))

    dic_count = collections.defaultdict(int)
    for i in range(25):
        dic_count[i] = 1
        for j in range(25):
            dic_count[i] += adj_mat[i][j]
        for j in range(25):
            adj_mat[i][j] = adj_mat[i][j]/dic_count[i]

    pickle.dump(adj_mat, open('/Users/zoe/Documents/event_extraction/majorEventDump/adjacency.regular', 'wb'))

    dic_seri = dict()
    for i in range(25):
        dic_seri[i] = pd.Series(adj_mat[i])

    dataFrame = pd.DataFrame(dic_seri)

    matplotlib.rcParams['xtick.direction'] = 'in'
    matplotlib.rcParams['ytick.direction'] = 'in'
    fig = plt.figure()  # 调用figure创建一个绘图对象
    ax = fig.add_subplot(111)
    cax = ax.matshow(dataFrame, vmin=0, vmax=1)  # 绘制热力图，从0到1
    fig.colorbar(cax)  # 将matshow生成热力图设置为颜色渐变条
    ticks = np.arange(0, 25, 5)  # 生成0-25，步长为5
    ax.set_xticks(ticks)  # 生成刻度
    ax.set_yticks(ticks)
    ax.set_xticklabels(np.arange(0, 26, 5))  # 生成x轴标签
    ax.set_yticklabels(np.arange(0, 26, 5))
    plt.savefig("/Users/zoe/Documents/event_extraction/latex/picture/adjacency1.png")
    plt.show()

# adjacency_matrix()


def generate_chain(eventsList):
    x_mat_list = list()
    x_mat = np.zeros(shape=(Chain_Lens*2))
    y_tag_list = list()
    y_tag = np.zeros(shape=(25))
    for company, eventSeq in eventsList.items():
        if len(eventSeq) > Chain_Lens:
            for beginIdx, e in enumerate(eventSeq):
                if beginIdx >= Chain_Lens:
                    for i in range(Chain_Lens):
                        x_mat[Chain_Lens - i - 1] = eventSeq[beginIdx - i - 1]['type']
                    Start_Date = datetime.datetime.strptime(eventSeq[beginIdx-Chain_Lens]['date'], '%Y%m%d')
                    for i in range(Chain_Lens):
                        This_Date = datetime.datetime.strptime(eventSeq[beginIdx - i - 1]['date'], '%Y%m%d')
                        timeDelta = 0
                        if This_Date-Start_Date < datetime.timedelta(4):
                            timeDelta = 1
                        elif This_Date-Start_Date < datetime.timedelta(8):
                            timeDelta = 2
                        elif This_Date-Start_Date < datetime.timedelta(31):
                            timeDelta = 3
                        else:
                            timeDelta = 4
                        x_mat[2*Chain_Lens - i - 1] = timeDelta
                    x_mat_list.append(x_mat)
                    x_mat = np.zeros(shape=(Chain_Lens*2))
                    y_tag[e['type']] = 1
                    y_tag_list.append(y_tag)
                    y_tag = np.zeros(shape=(25))
    return x_mat_list, y_tag_list


def get_pickle():
    global Chain_Lens
    # with open('/Users/zoe/Documents/event_extraction/majorEventDump/Category.json', 'r') as inputFile:
    #     Category = json.load(inputFile)
    with open('../data/TrainSet.json', 'r',
              encoding='utf-8') as inputFile:
        eventsTrain = json.load(inputFile)
    with open('../data/DevSet.json', 'r',
              encoding='utf-8') as inputFile:
        eventsDev = json.load(inputFile)
    with open('../data/TestSet.json', 'r',
              encoding='utf-8') as inputFile:
        eventsTest = json.load(inputFile)

    # ********数据链条的生成********
    x_train,y_train = generate_chain(eventsTrain)
    f_w = open('../data/pickle.data.1.train','wb')
    pickle.dump(np.array(x_train).astype(int), f_w)
    pickle.dump(np.array(y_train).astype(int), f_w, protocol=4)
    f_w.close()

    x_dev,y_dev = generate_chain(eventsDev)
    f_w = open('../data/pickle.data.1.dev','wb')
    pickle.dump(np.array(x_dev).astype(int), f_w)
    pickle.dump(np.array(y_dev).astype(int), f_w, protocol=4)
    f_w.close()

    x_test, y_test = generate_chain(eventsTest)
    f_w = open('../data/pickle.data.1.test', 'wb')
    pickle.dump(np.array(x_test).astype(int), f_w)
    pickle.dump(np.array(y_test).astype(int), f_w, protocol=4)
    f_w.close()

# get_pickle()

def scale():
    adjacency_mat = pickle.load(open('/Users/zoe/Documents/event_extraction/majorEventDump/adjacency.data', 'rb'))
    # adjacency_mat = dict(adjacency_mat)
    myarray = np.zeros((25, 25), dtype='float32')
    for key1, row in adjacency_mat.items():
        for key2, value in row.items():
            myarray[key1, key2] = value

    X_scaled = preprocessing.scale(myarray)
    print(X_scaled)
    myarray = myarray + np.eye(25)
    X_scaled = preprocessing.scale(myarray)
    print(np.around(X_scaled, decimals=2))


def adjacency_new():
    f = open('/Users/zoe/Desktop/event', 'r')
    a = f.readlines()
    dic = dict()
    for i in range(25):
        dic[i] = dict()

    index = 0
    index_2 = 0
    for i in a:
        for j in i.split():
            try:
                dic[index][index_2] = float(j)
                index_2 += 1
                if index_2 == 25:
                    index += 1
                    index_2 = 0
            except:
                pass
    print(dic)
    f_w = open('/Users/zoe/Documents/event_extraction/majorEventDump/adjacency.new', 'wb')
    pickle.dump(dic, f_w)
    f_w.close()

    myarray = np.zeros((25, 25), dtype='float32')
    for key1, row in dic.items():
        for key2, value in row.items():
            myarray[key1, key2] = value

    print(myarray)


def pandas_plot():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"
    names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class'] #设置变量名
    data = pd.read_csv(url, names=names)  #采用pandas读取csv数据
    correlations = data.corr()  #计算变量之间的相关系数矩阵
    # plot correlation matrix
    fig = plt.figure() #调用figure创建一个绘图对象
    ax = fig.add_subplot(111)
    cax = ax.matshow(correlations, vmin=-1, vmax=1)  #绘制热力图，从-1到1
    fig.colorbar(cax)  #将matshow生成热力图设置为颜色渐变条
    ticks = np.arange(0,9,1) #生成0-9，步长为1
    ax.set_xticks(ticks)  #生成刻度
    ax.set_yticks(ticks)
    ax.set_xticklabels(names) #生成x轴标签
    ax.set_yticklabels(names)
    plt.show()

    # pd.plotting.scatter_matrix(data,figsize=(10,10))
    # plt.show()


def combine_pred():
    for dirpath, dirnames, filenames in os.walk('../data/pred'):
        f_act = open('../data/pred_y_actual.txt', 'w')
        f_y = open('../data/pred_y.txt', 'w')
        for name in filenames:
            if re.search('actual', name):
                with open('../data/pred/{}'.format(name), 'r') as f_r:
                    content = f_r.read()
                f_act.write(content)

                s = 'pred_y_'+name[14:]
                with open('../data/pred/{}'.format(s), 'r') as f_r:
                    content = f_r.read()
                f_y.write(content)

        f_act.close()
        f_y.close()


def plot_temperature():
    with open('/Users/zoe/Documents/event_extraction/majorEventDump/pred_y.txt','r') as f_pred:
        pred = f_pred.readlines()

    with open('/Users/zoe/Documents/event_extraction/majorEventDump/pred_y_actual.txt','r') as f_pred:
        actual = f_pred.readlines()

    dic = collections.defaultdict()
    label_li = ["transaction",
"stock suspension",
"Initial Public Offerings",
"additional issue",
"allotment of shares",
"dividend",
"statement disclosure",
"information change",
"shareholders meeting",
"legal issues",
"shareholding reform",
"transaction in assets",
"stock ownership incentive",
"split off",
"review meeting of PORC",
"fiscal taxation policy",
"major project",
"rumor clarification",
"project investment",
"business events",
"natural hazard",
"foreign investment",
"pledge of stock right",
"credit rating",
"others"]
    for i in range(25):
        dic[i] = collections.defaultdict(int)

    for i in range(len(pred)):
        dic[int(actual[i].strip())][int(pred[i].strip())] += 1

    dic_count = collections.defaultdict(int)
    for i in range(25):
        dic_count[i] = 1
        for j in range(25):
            dic_count[i] += dic[i][j]
        for j in range(25):
            dic[i][j] = dic[i][j]/dic_count[i]

    dic_seri = dict()
    for i in range(25):
        dic_seri[i] = pd.Series(dic[i])

    dataFrame = pd.DataFrame(dic_seri)

    matplotlib.rcParams['xtick.direction'] = 'in'
    matplotlib.rcParams['ytick.direction'] = 'in'
    fig = plt.figure(figsize=(10, 6))  # 调用figure创建一个绘图对象
    ax = fig.add_subplot(111)
    cax = ax.matshow(dataFrame, vmin=0, vmax=1)  # 绘制热力图，从0到1
    fig.colorbar(cax)  # 将matshow生成热力图设置为颜色渐变条
    ticks = np.arange(0, 25, 1)  # 生成0-25，步长为5
    ax.set_xticks(ticks)  # 生成刻度
    ax.set_yticks(ticks)
    ax.set_xticklabels(label_li, rotation=90)  # 生成x轴标签
    ax.set_yticklabels(label_li)
    plt.savefig("/Users/zoe/Documents/event_extraction/coling2018/picture/label_new.png",bbox_inches='tight')
    plt.show()

# plot_temperature()


def draw_cost_trend():
    cost = pickle.load(open('remote/cost_trend.txt', 'rb'))
    cost_new = [sum(cost[:(index+1)])/(index+1) for index, d in enumerate(cost)]

    plt.ylabel('cost')
    plt.xlabel("step")
    plt.plot(cost_new, color='grey', linestyle='solid', label='baseline')

    plt.title('traing cost trend')
    plt.show()


def cos_dist(a, b):
    if len(a) != len(b):
        return None
    part_up = 0.0
    a_sq = 0.0
    b_sq = 0.0
    for a1, b1 in zip(a,b):
        part_up += a1*b1
        a_sq += a1**2
        b_sq += b1**2
    part_down = math.sqrt(a_sq*b_sq)
    if part_down == 0.0:
        return None
    else:
        return part_up / part_down


def class_similarity():
    embedding = [[-1.57161295e-01, -1.89983889e-01, 1.03012539e-01, 2.38786176e-01
                     , -3.89475450e-02, 2.33879667e-02, 7.59412870e-02, 3.74499887e-01
                     , -1.93081021e-01, 1.53337449e-01, -1.71474814e-01, -2.54947156e-01
                     , 2.82535926e-02, 8.88032988e-02, 1.14700086e-01, -5.55159785e-02
                     , -1.92168638e-01, -1.33541718e-01, -2.28156790e-01, -1.48895025e-01
                     , -8.63357708e-02, -2.86016047e-01, -1.76630735e-01, -9.04595330e-02
                     , -2.56556481e-01, 1.17997684e-01, 2.43195109e-02, -3.27525474e-02
                     , 4.18942682e-02, 6.33072257e-02, 1.49440020e-01, 3.41351964e-02],
                 [1.99788541e-01, -1.76086202e-01, 8.80397633e-02, -3.00069243e-01
                     , -2.92199910e-01, -1.20761178e-01, -2.79335886e-01, 6.32360950e-02
                     , -2.49200985e-01, -8.80602002e-02, -2.05721289e-01, -2.19713911e-01
                     , -8.35007802e-02, 2.40257680e-01, -1.69042479e-02, 1.08116850e-01
                     , 7.19793364e-02, -2.98958570e-01, -2.18172148e-01, -1.80390049e-02
                     , 1.97767198e-01, -1.11020766e-01, 2.16688141e-01, -1.01792224e-01
                     , 1.75041020e-01, -2.60977298e-01, 6.27459539e-03, 1.70253485e-01
                     , -2.45756313e-01, -1.56736784e-02, -1.03714161e-01, -1.80061162e-01],
                 [1.84804022e-01, -1.49172023e-01, -9.96057242e-02, -7.58738667e-02
                     , -9.01982859e-02, -1.52589634e-01, 5.42213907e-03, -1.36427939e-01
                     , 1.98572944e-03, -2.15640202e-01, -1.33735374e-01, -6.17589504e-02
                     , -2.41983309e-01, -1.02951929e-01, 1.41319439e-01, 4.99109775e-02
                     , -2.07004607e-01, 4.67377156e-01, 7.37479553e-02, 2.74639159e-01
                     , 2.56833524e-01, -3.64811808e-01, -1.65715553e-02, -8.44749138e-02
                     , 2.03846306e-01, 3.04011703e-01, 1.55618057e-01, 1.83124542e-01
                     , 2.50571698e-01, -1.26683652e-01, 6.77567869e-02, 4.12142351e-02],
                 [-3.44454825e-01, -1.19725645e-01, -2.41567045e-01, 1.64009444e-02
                     , -4.65558060e-02, -3.02006602e-01, -3.00329536e-01, -3.04763988e-02
                     , -1.49530083e-01, -1.78949967e-01, -4.87838164e-02, 1.56493559e-01
                     , -2.63772130e-01, 1.59263253e-01, -1.72002375e-01, -1.44270137e-01
                     , -4.29848470e-02, 8.70769960e-04, 1.43208941e-02, -1.24316812e-01
                     , -2.74918586e-01, 2.38501698e-01, 2.73874164e-01, -1.61601245e-01
                     , -9.84503478e-02, -8.15098211e-02, -3.85795355e-01, -5.08690178e-02
                     , 2.22819328e-01, 1.55713037e-01, -1.15502901e-01, -1.99662060e-01],
                 [3.89995694e-01, 8.61455873e-03, -3.27134043e-01, 2.93973416e-01
                     , -1.50635213e-01, 9.30641443e-02, -4.88608070e-02, 1.88589543e-01
                     , 5.22805631e-01, -1.19904995e-01, -3.61040413e-01, -2.72108704e-01
                     , 1.87666446e-01, 1.18682668e-01, 6.34230897e-02, -5.39021969e-01
                     , 2.86272280e-02, -1.09841481e-01, -1.66638032e-01, 5.46262541e-04
                     , 3.17989230e-01, -1.36395633e-01, 1.23468496e-01, -2.76787013e-01
                     , 2.76986778e-01, 3.30761641e-01, -3.42945427e-01, 1.11287376e-02
                     , 1.17339090e-01, -1.09963477e-01, -3.66604269e-01, -1.02637835e-01],
                 [1.73027858e-01, 1.07734777e-01, -2.18573958e-01, -8.88767615e-02
                     , 3.08536112e-01, 1.60154998e-02, -1.71278968e-01, 2.85387129e-01
                     , -1.31571472e-01, 1.19860083e-01, -2.41281644e-01, 3.30133066e-02
                     , 5.91610745e-02, 3.46006230e-02, -7.45422766e-02, -1.53815836e-01
                     , 1.41100019e-01, 4.21905458e-01, -2.56842852e-01, -1.08357947e-02
                     , -2.09523761e-03, 1.38563380e-01, 1.22057222e-01, 1.55064821e-01
                     , 1.91733020e-03, -2.13467956e-01, 2.01885238e-01, -2.94580590e-02
                     , 2.05075651e-01, 9.00369138e-02, -1.90428402e-02, -2.03545734e-01],
                 [9.91890207e-03, 1.21264733e-01, -6.44717067e-02, 6.19965158e-02
                     , -1.03963383e-01, -6.64920136e-02, 2.47683764e-01, -2.28045091e-01
                     , 5.68795614e-02, 8.06040838e-02, -2.44105205e-01, -1.82429478e-01
                     , -1.27871037e-01, -1.47719562e-01, 1.50184259e-01, 1.61502361e-01
                     , -7.78844580e-02, -2.48725086e-01, -3.28932032e-02, 3.03741582e-02
                     , 2.82751136e-02, 2.52249777e-01, 3.98812145e-02, 1.72214255e-01
                     , -1.96113765e-01, -1.08774185e-01, -1.40212342e-01, -2.79010296e-01
                     , 1.58542290e-01, 8.70212689e-02, 3.41010280e-02, 9.91719887e-02],
                 [-1.10554725e-01, 1.77333981e-01, 8.90590549e-02, 2.36014605e-01
                     , 1.74974009e-01, -8.40772167e-02, -1.06100440e-01, -1.77954156e-02
                     , 9.74835828e-02, -1.06165163e-01, -1.65994540e-01, 3.30738902e-01
                     , -1.63636833e-01, 1.14135586e-01, -1.51083261e-01, -1.10907845e-01
                     , -5.23299649e-02, -2.36241832e-01, -9.45965126e-02, 1.28992766e-01
                     , -9.53931659e-02, 9.30104479e-02, 8.12153816e-02, -1.25168160e-01
                     , 1.30806610e-01, -1.01815835e-01, 1.43707782e-01, -1.28596798e-01
                     , -6.46520108e-02, -1.54097587e-01, 2.08808392e-01, 1.96774974e-01],
                 [4.15459014e-02, 2.09396183e-01, 2.65769917e-03, -1.67056933e-01
                     , 5.54902330e-02, 1.71326265e-01, -7.65269473e-02, 2.14417666e-01
                     , 1.48714527e-01, -1.53183848e-01, 2.39692420e-01, -1.79248497e-01
                     , 3.48618925e-02, -8.02942514e-02, -3.65877263e-02, 2.09657431e-01
                     , -9.67885703e-02, -1.00534022e-01, 1.96857393e-01, 1.95017979e-01
                     , -8.67156386e-02, -2.93804079e-01, -1.18134534e-02, -1.01012632e-01
                     , -2.31772602e-01, -3.63908596e-02, -1.00720353e-01, -1.94664210e-01
                     , -1.84578419e-01, -1.76244617e-01, -9.75305438e-02, -1.81270733e-01],
                 [-4.25368473e-02, -4.31258045e-02, 6.14087470e-02, -8.22891891e-02
                     , -1.67862207e-01, 9.79830846e-02, -1.51053751e-02, -1.48468688e-01
                     , -1.04869120e-01, -2.03970760e-01, 9.62731764e-02, 1.44947335e-01
                     , 1.29185408e-01, 5.52954115e-02, -6.68629957e-03, 2.15312690e-01
                     , 9.75432992e-02, -3.05020273e-01, 2.33575284e-01, -3.20455641e-01
                     , -2.25016758e-01, 4.60864529e-02, 1.41503602e-01, 3.56830627e-01
                     , -1.91494420e-01, 2.95162350e-01, 2.48758510e-01, 2.75328159e-01
                     , 1.73583001e-01, 1.75727007e-03, 1.69339944e-02, 4.82044481e-02],
                 [-4.15607393e-02, -2.25391805e-01, -4.07721996e-01, 2.19476566e-01
                     , -4.31279056e-02, 2.71894008e-01, 4.61616665e-01, 3.36920954e-02
                     , -2.09158137e-01, -7.19520450e-02, 2.48945192e-01, 2.37132218e-02
                     , -3.47459286e-01, -1.00528568e-01, 8.64713266e-02, 1.91748202e-01
                     , -1.25940233e-01, 5.66967614e-02, -4.71212655e-01, -1.42285794e-01
                     , -2.82439351e-01, 8.51020142e-02, 7.09017664e-02, -6.84500039e-02
                     , 2.94067949e-01, 2.45161146e-01, 3.12151045e-01, -1.23831248e-02
                     , -1.70564920e-01, 5.47587536e-02, -2.94575840e-01, -5.44316880e-02],
                 [3.49164382e-02, 2.49426648e-01, -1.20256364e-01, 2.26362929e-01
                     , -1.98481932e-01, -3.28893065e-01, -1.25045747e-01, 1.04642473e-03
                     , 3.50284353e-02, -8.66964832e-02, 5.33374883e-02, -2.87301838e-01
                     , -2.24972032e-02, -1.27441520e-02, 1.38132483e-01, -1.22171745e-01
                     , 8.37843716e-02, 4.24691327e-02, 3.03804964e-01, -1.36906013e-01
                     , -2.30597988e-01, -8.32669362e-02, -1.39840141e-01, -1.71898097e-01
                     , 2.25954667e-01, -1.07517615e-02, 5.26782684e-02, 8.16818848e-02
                     , -2.34937564e-01, 4.04823422e-02, 2.11450890e-01, -9.17578712e-02],
                 [-1.10476889e-01, 1.65936351e-02, -5.78392297e-02, 6.63717762e-02
                     , -2.98602283e-02, -4.10140365e-01, 8.69740769e-02, 2.55005211e-01
                     , -2.03108713e-01, -1.07925102e-01, -2.00554840e-02, 9.43102688e-02
                     , 1.54264212e-01, -1.29968688e-01, 1.85932949e-01, -1.75603554e-01
                     , 2.74207681e-01, 4.59730215e-02, 6.89278841e-02, -9.40344781e-02
                     , 1.67962581e-01, -4.88451831e-02, -9.34759155e-02, -8.26862305e-02
                     , -7.16540068e-02, 7.97837302e-02, -4.83215824e-02, -1.20808773e-01
                     , -1.50960505e-01, -1.39707327e-01, 5.23849763e-02, -7.05236569e-02],
                 [5.52817822e-01, -2.06906959e-01, 9.64815095e-02, 2.69579202e-01
                     , -2.16821596e-01, -4.14411366e-01, -1.68734267e-01, -1.31329522e-01
                     , 7.43580386e-02, -2.42378548e-01, -7.66851008e-02, 1.04681037e-01
                     , 3.28492016e-01, -1.43483028e-01, -1.69687673e-01, -1.10302955e-01
                     , -7.03703761e-02, -1.50980830e-01, -1.74261630e-01, -2.09368885e-01
                     , -2.22906023e-01, -3.76896262e-01, 4.35685627e-02, -1.10270001e-01
                     , -4.75683175e-02, -4.01719622e-02, -8.67389962e-02, -1.20092332e-01
                     , 1.35312974e-01, 3.12930644e-01, -2.64616460e-01, 3.76980789e-02],
                 [2.52697710e-02, -4.71901298e-01, 2.73479581e-01, 4.53707784e-01
                     , 1.43773168e-01, -4.76678906e-05, -8.66507590e-02, -9.22416747e-02
                     , 2.67313182e-01, -2.85879504e-02, -6.37832582e-02, 1.22086093e-01
                     , -1.95029631e-01, -7.98411429e-01, 5.01088388e-02, -4.70274091e-01
                     , 5.33486791e-02, -5.64782172e-02, -1.26761064e-01, 4.45410721e-02
                     , -1.70309320e-01, 2.01096997e-01, 3.11620861e-01, 2.75499344e-01
                     , -3.71875167e-01, -6.69641048e-02, -3.06711972e-01, 6.15720227e-02
                     , -3.11016887e-01, -2.41252333e-01, -2.62024939e-01, -4.54306632e-01],
                 [-1.29505210e-02, 1.82982653e-01, -9.94461179e-02, 1.73904940e-01
                     , -8.30368698e-02, 1.43148229e-01, 1.00696422e-01, -2.36088157e-01
                     , -1.08257674e-01, -1.76070675e-01, 2.67563015e-01, -6.08665273e-02
                     , -2.08051428e-02, 3.21527392e-01, 3.17115664e-01, -4.25656438e-01
                     , -4.04887721e-02, 2.44512651e-02, 1.42064288e-01, 1.07943892e-01
                     , -1.19268984e-01, -1.77748904e-01, -1.10366389e-01, 1.52225271e-01
                     , -3.35311820e-03, -3.78567427e-02, 1.74685925e-01, 1.18340127e-01
                     , -9.51714143e-02, 8.54694769e-02, -6.31805956e-02, -2.99161691e-02],
                 [-9.30402651e-02, 1.31838322e-01, -1.03162937e-01, -1.77002057e-01
                     , 1.49315134e-01, -1.04190744e-01, -9.32701584e-03, 8.85644853e-02
                     , -1.56773701e-01, 1.55595526e-01, -4.90804203e-02, 1.06882453e-01
                     , -3.65983136e-02, -1.38074994e-01, 2.20838636e-01, -3.57136637e-01
                     , -3.43095988e-01, -1.15751632e-01, 4.20542479e-01, -3.30136046e-02
                     , 1.02171086e-01, 4.67335768e-02, -1.08730800e-01, -2.72763908e-01
                     , 1.99251890e-01, -3.85273322e-02, 1.96627155e-01, 2.37832218e-01
                     , -2.26591766e-01, 1.43809453e-01, -2.12656647e-01, -3.20823304e-02],
                 [-5.44764027e-02, -1.05936281e-01, -1.11657999e-01, 3.19843054e-01
                     , -8.25895891e-02, 3.19374412e-01, -1.82648301e-02, 1.29633278e-01
                     , -2.79737651e-01, -2.28826404e-01, -4.47968058e-02, 1.72189265e-01
                     , 1.19763613e-01, -2.11106211e-01, 8.57028440e-02, -5.34971543e-02
                     , 7.01206028e-02, -5.89777566e-02, 2.02022865e-01, -2.16981262e-01
                     , -5.23463301e-02, -4.06637415e-02, 9.29354578e-02, -8.27897638e-02
                     , -1.26002863e-01, -2.09782243e-01, -5.26544899e-02, 2.25323334e-01
                     , -4.41675968e-02, 2.77444512e-01, -1.41317710e-01, -1.05302287e-02],
                 [4.49550897e-02, 8.24588817e-03, 1.57987639e-01, 3.08873832e-01
                     , 4.31946337e-01, -5.74961081e-02, -1.34244189e-01, -2.08436280e-01
                     , 1.70339122e-01, 1.71600766e-02, -1.72197700e-01, 9.10824761e-02
                     , 8.56112614e-02, 9.21183676e-02, 7.82220662e-02, -3.30149420e-02
                     , -9.91600528e-02, 1.21023335e-01, 2.05243304e-01, -9.47265252e-02
                     , -2.51982529e-02, -4.24959324e-02, -1.11870512e-01, -1.64854079e-01
                     , -3.93017121e-02, 2.70831794e-01, -1.54808402e-01, 3.29439968e-01
                     , -1.93772599e-01, -1.51459768e-01, -1.69324785e-01, -4.11759466e-02],
                 [-2.59021074e-02, -1.45834774e-01, -1.56072527e-02, 2.22313493e-01
                     , -9.87347439e-02, 1.96333900e-01, -1.80625707e-01, -1.53088525e-01
                     , -1.91269554e-02, -1.75026461e-01, 2.79903803e-02, -6.93758354e-02
                     , 1.19310945e-01, 8.98085460e-02, 1.11482166e-01, -1.47953555e-01
                     , -8.11022222e-02, 1.99737906e-01, 5.13409555e-01, 9.90326926e-02
                     , 2.33794183e-01, -1.43636957e-01, 1.40856177e-01, -8.59865844e-02
                     , -6.00796379e-03, -1.45836800e-01, -1.87109292e-01, 2.01016083e-01
                     , -1.14388108e-01, 1.06938526e-01, 3.49598438e-01, 8.97224322e-02],
                 [-1.55582488e-01, -1.96618155e-01, -2.43088558e-01, -2.61629075e-01
                     , 5.37945144e-02, 7.09758028e-02, -2.62059957e-01, -1.16205625e-01
                     , 2.54338086e-01, 2.32253999e-01, 6.56636208e-02, 2.49807507e-01
                     , 1.24168672e-01, 1.60726815e-01, 2.15751916e-01, -1.71129003e-01
                     , 1.27141222e-01, 2.40539275e-02, 5.44608720e-02, 4.09176290e-01
                     , -2.64324434e-02, -1.74178421e-01, -8.56204331e-02, 1.52798206e-01
                     , 9.59542543e-02, 1.07159168e-01, 2.15657502e-01, -1.20541014e-01
                     , -1.97945222e-01, -4.93767932e-02, -2.43772138e-02, 2.03278199e-01],
                 [9.04538035e-02, -2.03401998e-01, 2.12705150e-01, 2.29457825e-01
                     , -7.60262180e-03, -8.55764300e-02, 6.22496381e-02, 2.27163017e-01
                     , -8.59895442e-03, 1.57783791e-01, 4.59265918e-01, -3.05141538e-01
                     , -7.36725181e-02, 1.11072063e-01, -5.78133985e-02, -2.68817037e-01
                     , -3.86595167e-02, 7.65901208e-02, 3.28476697e-01, 3.76873016e-02
                     , -1.05840981e-01, -4.89083529e-02, -1.34486616e-01, 1.08015455e-01
                     , 4.69722182e-01, -2.16655001e-01, -7.86301047e-02, -1.32868856e-01
                     , 2.05401331e-02, -9.57152247e-02, -9.09780487e-02, 1.46489590e-01],
                 [1.65558204e-01, -1.29308268e-01, -1.44686565e-01, -7.41932467e-02
                     , 2.96519756e-01, -1.27630234e-01, 1.29205972e-01, 3.36858541e-01
                     , 5.60355224e-02, -1.00208730e-01, 2.91471869e-01, -1.90562867e-02
                     , -3.66248995e-01, 2.30075955e-01, -1.28442869e-01, -1.07713893e-01
                     , 1.27795592e-01, -5.13137542e-02, 1.33562386e-01, -1.08249389e-01
                     , 2.98404872e-01, -1.50339436e-02, -1.08006492e-01, 5.86737059e-02
                     , -2.39131972e-02, -2.78357379e-02, -2.01797888e-01, 2.36955494e-01
                     , -1.65426075e-01, 1.41928524e-01, 3.32876146e-01, 1.80592373e-01],
                 [-4.26367253e-01, -1.96125075e-01, 3.22032757e-02, -5.36839887e-02
                     , 1.03842631e-01, 8.12997743e-02, 9.64962691e-03, 1.14311442e-01
                     , -2.56910473e-01, -7.11107701e-02, 1.07793935e-01, 1.19004637e-01
                     , 5.14970496e-02, 6.84414208e-02, -8.84890482e-02, 1.42680183e-01
                     , -2.12377489e-01, 7.60182589e-02, 1.17774464e-01, -2.32399311e-02
                     , 2.39671737e-01, -1.07937768e-01, 2.87295401e-01, 2.81250656e-01
                     , -7.54216835e-02, -2.56643835e-02, -2.43056446e-01, 2.53925055e-01
                     , -6.17264733e-02, -1.22956567e-01, -1.29628971e-01, 4.00376648e-01],
                 [-1.79506525e-01, -2.44528696e-01, -1.42990395e-01, 1.96824878e-01
                     , -1.31022319e-01, -2.08813369e-01, 5.41574322e-03, -1.66577086e-01
                     , 7.70321069e-03, -9.38002244e-02, -8.63091350e-02, 3.26616853e-01
                     , 5.43950349e-02, 4.85129990e-02, -1.70099527e-01, -1.61791164e-02
                     , 2.08088860e-01, 2.82616746e-02, 2.67219335e-01, 2.88097292e-01
                     , 2.19273090e-01, -3.42016429e-01, -1.19789809e-01, -7.62471110e-02
                     , -1.03423446e-01, -9.38234106e-02, 1.59136936e-01, -9.97497961e-02
                     , 1.87239259e-01, 4.11633879e-01, -4.04454231e-01, 4.61508669e-02]]

    label_li = ["transaction",
                "stock suspension",
                "Initial Public Offerings",
                "additional issue",
                "allotment of shares",
                "dividend",
                "statement disclosure",
                "information change",
                "shareholders meeting",
                "legal issues",
                "shareholding reform",
                "transaction in assets",
                "stock ownership incentive",
                "split off",
                "review meeting of PORC",
                "fiscal taxation policy",
                "major project",
                "rumor clarification",
                "project investment",
                "business events",
                "natural hazard",
                "foreign investment",
                "pledge of stock right",
                "credit rating",
                "others"]

    with open('/Users/zoe/Desktop/1.txt', 'r') as f:
        col = f.readlines()
    colo = [i[1:8] for i in col]

    a = [colo[i] for i in np.random.randint(524, size=25)]

    # pca = PCA(n_components=2)
    # pca.fit(embedding)
    # print(pca.explained_variance_ratio_)

    X_pca = PCA(n_components=2).fit_transform(embedding)

    plt.figure(figsize=(8, 8))
    plt.subplot(111)
    for i in range(25):
        plt.scatter(X_pca[i, 0], X_pca[i, 1], c=a[i], label=label_li[i])
    plt.legend(loc='right')
    plt.savefig("/Users/zoe/Documents/event_extraction/coling2018/picture/class_pca.png")
    plt.show()

    # , c = label_li
    # dic = collections.defaultdict()
    #
    # for i in range(25):
    #     dic[i] = collections.defaultdict(int)
    #
    # for i in range(25):
    #     for j in range(25):
    #         dic[i][j] = cos_dist(embedding[i], embedding[j])

    # dic_seri = dict()
    # for i in range(25):
    #     dic_seri[i] = pd.Series(dic[i])

    # dataFrame = pd.DataFrame(dic_seri)
    #
    # matplotlib.rcParams['xtick.direction'] = 'in'
    # matplotlib.rcParams['ytick.direction'] = 'in'
    # fig = plt.figure(figsize=(10,8))  # 调用figure创建一个绘图对象
    # ax = fig.add_subplot(111)
    # cax = ax.matshow(dataFrame, vmin=0, vmax=1)  # 绘制热力图，从0到1
    # fig.colorbar(cax)  # 将matshow生成热力图设置为颜色渐变条
    # ticks = np.arange(0, 25, 1)  # 生成0-25，步长为5
    # ax.set_xticks(ticks)  # 生成刻度
    # ax.set_yticks(ticks)
    # ax.set_xticklabels(label_li, rotation=90)  # 生成x轴标签
    # ax.set_yticklabels(label_li)
    # plt.savefig("/Users/zoe/Documents/event_extraction/coling2018/picture/class_similarity.png")
    # plt.show()

# class_similarity()


def compareTrainTest_smallclass():
    with open('/Users/zoe/Documents/event_extraction/majorEventDump/majorEventAll.json', 'r',
              encoding='utf-8') as inputFile:
        events = json.load(inputFile)

    # with open('/Users/zoe/Documents/event_extraction/majorEventDump/typeCodeDump.json', 'r',
    #           encoding='utf-8') as inputFile:
    #     code2type = json.load(inputFile)
    #
    # with open('/Users/zoe/Documents/event_extraction/majorEventDump/Class.json', 'r', encoding='utf8') as inputFile:
    #     typeClass = json.load(inputFile)
    #
    # typeDict = dict()
    # # typeList = [t for t in typeClass.keys()]
    #
    # for t in typeClass.keys():
    #     for c in typeClass[t]:
    #         typeDict[c] = len(typeDict)
    #
    # typeList = {typeDict[one]: one for one in typeDict}

    # print(code2type)
    ### 获得按时间排序的公司事件链条

    # error = collections.defaultdict(int)
    # countdict = collections.defaultdict(int)
    # countdictTest = collections.defaultdict(int)
    # dictTrain = collections.defaultdict(list)
    # dictTest = collections.defaultdict(list)
    # dictDev = collections.defaultdict(list)
    countEachCp = collections.defaultdict(int)

    for event in events:
        company = event['S_INFO_WINDCODE']
        countEachCp[len(event['event'])] += 1


    # with open('/Users/zoe/Documents/event_extraction/majorEventDump/TrainSetSC.json', 'w') as f_w:
    #     json.dump(dictTrain, f_w, indent=1)
    # with open('/Users/zoe/Documents/event_extraction/majorEventDump/DevSetSC.json', 'w') as f_w:
    #     json.dump(dictDev, f_w, indent=1)
    # with open('/Users/zoe/Documents/event_extraction/majorEventDump/TestSetSC.json', 'w') as f_w:
    #     json.dump(dictTest, f_w, indent=1)

    # print(error, len(error))
    # with open('temp.txt', 'w') as f_w:
    #     json.dump(error, f_w, ensure_ascii=False, indent=1)



    groupedDict= collections.defaultdict(int)

    for key in countEachCp:
        groupedDict[int(key)//100] += countEachCp[key]

    countdict = sorted(groupedDict.items(), key=lambda d:d[0], reverse=False)
    countdict = {one[0]:one[1] for one in countdict}
    # print(countdict)

    # countdictTest = sorted(countdictTest.items(), key=lambda d:d[1], reverse=True)
    # countdictTest = {one[0]:one[1] for one in countdictTest}
    # print(countdictTest)
    #
    with open('/Users/zoe/Documents/event_extraction/majorEventDump/EventLenEachCp.json', 'w', encoding='utf8') as f_w:
        json.dump(countdict, f_w, ensure_ascii=False, indent=1)

    # with open('/Users/zoe/Documents/event_extraction/majorEventDump/TestClass_L.json', 'w', encoding='utf8') as f_w:
    #     json.dump(countdictTest, f_w, ensure_ascii=False, indent=1)

# compareTrainTest_smallclass()


def SC_similarity():
    with open('/Users/zoe/Documents/event_extraction/majorEventDump/Class.json', 'r', encoding='utf8') as file:
        cla = json.load(file)
    class_length = [len(cla[i]) for i in cla]

    embedding = pickle.load(open('remote/SC_embedding.txt', 'rb'))
    label_li = ["stock trading",
                "stock suspension",
                "IPO",
                "SPO",
                "allotment of shares",
                "dividend",
                "performance disclosure",
                "information change",
                "shareholders meeting",
                "legal issues",
                "shareholding reform",
                "transaction in assets",
                "stock ownership incentive",
                "split off",
                "review meeting of PORC",
                "fiscal taxation policy",
                "major project",
                "rumor clarification",
                "project investment",
                "business events",
                "natural hazard",
                "investment",
                "pledge of stock right",
                "credit rating",
                "others"]

    with open('/Users/zoe/Desktop/1.txt', 'r') as f:
        col = f.readlines()
    colo = [i[1:8] for i in col]

    a = [colo[i] for i in np.random.randint(524, size=25)]

    # pca = PCA(n_components=2)
    # pca.fit(embedding)

    X_pca = PCA(n_components=3)
    # X_pca.fit(embedding)
    # print(X_pca.explained_variance_ratio_)
    X_pca = X_pca.fit_transform(embedding)

    plt.figure(figsize=(8, 8))
    axes = plt.subplot(111)
    index = 0
    size = [50,60,0,0,70]
    mark = ["s","o","","","*"]
    x = []
    y = []
    pltlist = []
    for i in range(20):
        for j in range(class_length[i]):
            x.append(X_pca[index+j, 0])
            y.append(X_pca[index+j, 1])
        if i in [0,1,4]:
            pltlist.append(axes.scatter(x, y, c=a[i], s=size[i],marker=mark[i]))
        index += class_length[i]
        x = []
        y = []
    matplotlib.rc('font', family='Times New Roman')
    axes.legend(pltlist, label_li)
    plt.savefig('/Users/zoe/Documents/event_extraction/coling2018/picture/SC_plot.png', dpi=600)
    plt.show()

SC_similarity()
