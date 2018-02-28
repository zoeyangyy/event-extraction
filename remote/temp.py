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
import pandas as pd
import numpy as np
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

def getChineseFont():
    return FontProperties(fname='/System/Library/Fonts/PingFang.ttc')

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
    with open('/Users/zoe/Documents/event_extraction/majorEventDump/majorEventAll.json', 'r',
              encoding='utf-8') as inputFile:
        events = json.load(inputFile)

    with open('/Users/zoe/Documents/event_extraction/majorEventDump/typeCodeDump.json', 'r',
              encoding='utf-8') as inputFile:
        code2type = json.load(inputFile)

    with open('/Users/zoe/Documents/event_extraction/majorEventDump/Class.json', 'r', encoding='utf8') as inputFile:
        typeClass = json.load(inputFile)

    typeDict = dict()
    typeList = [t for t in typeClass.keys()]
    for t in typeClass.keys():
        for c in typeClass[t]:
            typeDict[c] = typeList.index(t)

    # print(code2type)
    ### 获得按时间排序的公司事件链条

    # error = collections.defaultdict(int)
    # countdict = collections.defaultdict(int)
    # countdictTest = collections.defaultdict(int)
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
                if datetime.datetime.strptime(s_event['date'], '%Y%m%d') < datetime.datetime(2017, 1, 1):
                    # countdict[code2type[s_event['type']]] += 1
                    # countdict[typeList[s_event['type']]] += 1
                    dictTrain[company].append(new_event)
                elif datetime.datetime(2017, 1, 1) <= datetime.datetime.strptime(s_event['date'], '%Y%m%d') < datetime.datetime(2017, 7, 1):
                    # countdict[code2type[s_event['type']]] += 1
                    # countdict[typeList[s_event['type']]] += 1
                    dictDev[company].append(new_event)
                elif datetime.datetime.strptime(s_event['date'], '%Y%m%d') >= datetime.datetime(2017, 7, 1):
                    # countdictTest[code2type[s_event['type']]] += 1
                    # countdictTest[typeList[s_event['type']]] += 1
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
    #
    # countdict = sorted(countdict.items(), key=lambda d:d[1], reverse=True)
    # countdict = {one[0]:one[1] for one in countdict}
    #
    # countdictTest = sorted(countdictTest.items(), key=lambda d:d[1], reverse=True)
    # countdictTest = {one[0]:one[1] for one in countdictTest}
    #
    # with open('/Users/zoe/Documents/event_extraction/majorEventDump/TrainClass_L.json', 'w') as f_w:
    #     json.dump(countdict, f_w, ensure_ascii=False, indent=1)
    # with open('/Users/zoe/Documents/event_extraction/majorEventDump/TestClass_L.json', 'w') as f_w:
    #     json.dump(countdictTest, f_w, ensure_ascii=False, indent=1)

# compareTrainTest()


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


def a():
    # 64*5*128
    # 2*3*2
    states = [[[0,0],[1,1],[2,2]],[[3,3],[4,4],[5,5]]]
    sess = tf.Session()
    results = tf.constant(0)
    print(sess.run(results))

    for i in range(3):
        result_beta = tf.slice(states, [0,i,0],[-1,1,-1]) * (i+1)
        result_beta = tf.reshape(result_beta,[2,2])
        print(sess.run(result_beta))
        results = tf.add(results, result_beta)
        print(results.shape[0])
    print(sess.run(results))


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
    epoch = [one for one in range(1,6)]
    baseline = [0.573659, 0.589987, 0.590476, 0.578782, 0.579058]
    position = [0.559008, 0.58593, 0.591706, 0.590526, 0.590573]
    time = [0.552299, 0.508444, 0.490502, 0.48826, 0.478486]
    self = [0.0272559, 0.199495, 0.199624, 0.199512, 0.199754]
    all = [0.552007, 0.59035, 0.591128, 0.587372, 0.592395]
    plt.ylabel('accuracy')
    plt.xlabel("epoch")
    plt.plot(epoch, baseline, color='grey', linestyle='solid', label='baseline')
    # plt.plot(epoch, position, color='#FF9500', linestyle='solid', label='position')
    # plt.plot(epoch, time, color='#0C5DA5', linestyle='solid', label='time')
    # plt.plot(epoch, self, color='#00AC6B', linestyle='solid', label='self')
    plt.plot(epoch, all, color='#E7003E', linestyle='solid', label='all')

    plt.title('testing result')
    plt.xticks(epoch)
    plt.legend()
    plt.show()

# draw_result()


def absolute_path():
    PROJECT_PATH = os.path.dirname(os.path.realpath(__file__))  # 获取项目根目录
    data_file_path = os.path.join(PROJECT_PATH, "Dict/stopWord.txt")  # 文件路径
    stop_words = [w.strip() for w in open(data_file_path, 'r', encoding='GBK').readlines()]
    stop_words.extend(['\n','\t',' '])


Chain_Lens = 5


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
            if adj_mat[i][j]/dic_count[i] > regularization:
                adj_mat[i][j] = 1
            else:
                adj_mat[i][j] = 0

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
    plt.savefig("/Users/zoe/Documents/event_extraction/latex/picture/adjacency.png")
    plt.show()


adjacency_matrix()


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
                    Start_Date = datetime.datetime.strptime(eventSeq[beginIdx-5]['date'], '%Y%m%d')
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
                        x_mat[Chain_Lens - i + 4] = timeDelta
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
    with open('/Users/zoe/Documents/event_extraction/majorEventDump/TrainSet.json', 'r',
              encoding='utf-8') as inputFile:
        eventsTrain = json.load(inputFile)
    with open('/Users/zoe/Documents/event_extraction/majorEventDump/DevSet.json', 'r',
              encoding='utf-8') as inputFile:
        eventsDev = json.load(inputFile)
    with open('/Users/zoe/Documents/event_extraction/majorEventDump/TestSet.json', 'r',
              encoding='utf-8') as inputFile:
        eventsTest = json.load(inputFile)

    # ********数据链条的生成********
    x_train,y_train = generate_chain(eventsTrain)
    f_w = open('/Users/zoe/Documents/event_extraction/majorEventDump/pickle.data.train','wb')
    pickle.dump(np.array(x_train).astype(int), f_w)
    pickle.dump(np.array(y_train).astype(int), f_w)
    f_w.close()

    x_dev,y_dev = generate_chain(eventsDev)
    f_w = open('/Users/zoe/Documents/event_extraction/majorEventDump/pickle.data.dev','wb')
    pickle.dump(np.array(x_dev).astype(int), f_w)
    pickle.dump(np.array(y_dev).astype(int), f_w)
    f_w.close()

    x_test, y_test = generate_chain(eventsTest)
    f_w = open('/Users/zoe/Documents/event_extraction/majorEventDump/pickle.data.test', 'wb')
    pickle.dump(np.array(x_test).astype(int), f_w)
    pickle.dump(np.array(y_test).astype(int), f_w)
    f_w.close()

# get_pickle()


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
    fig = plt.figure()  # 调用figure创建一个绘图对象
    ax = fig.add_subplot(111)
    cax = ax.matshow(dataFrame, vmin=0, vmax=1)  # 绘制热力图，从0到1
    fig.colorbar(cax)  # 将matshow生成热力图设置为颜色渐变条
    ticks = np.arange(0, 25, 5)  # 生成0-25，步长为5
    ax.set_xticks(ticks)  # 生成刻度
    ax.set_yticks(ticks)
    ax.set_xticklabels(np.arange(0, 26, 5))  # 生成x轴标签
    ax.set_yticklabels(np.arange(0, 26, 5))
    plt.savefig("/Users/zoe/Documents/event_extraction/latex/picture/label.png")
    plt.show()
