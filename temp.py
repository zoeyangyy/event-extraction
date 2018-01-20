#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Time        : 2017/11/20 上午10:52
# @Author      : Zoe
# @File        : temp.py
# @Description :

import  matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pylab
from matplotlib.font_manager import FontManager, FontProperties
import pandas as pd
import numpy as np
import random
import re
import collections
import json
import datetime
import tensorflow as tf

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
    with open('/Users/zoe/Documents/event_extraction/majorEventDump/Class.txt', 'r') as f_r:
        typeClass = collections.defaultdict(list)
        typeList = list()
        for line in f_r.readlines():
            if line != '\n':
                typeList.append(line.strip())
            else:
                typeClass[typeList[0]] = typeList[1:]
                typeList = list()
        # print(typeClass)

    with open('/Users/zoe/Documents/event_extraction/majorEventDump/Class.json', 'w') as f_w:
        json.dump(typeClass, f_w, indent=1, ensure_ascii=False)

# generat_class()


def compareTrainTest():
    with open('/Users/zoe/Documents/event_extraction/majorEventDump/majorEventAll.json', 'r',
              encoding='utf-8') as inputFile:
        events = json.load(inputFile)

    with open('/Users/zoe/Documents/event_extraction/majorEventDump/typeCodeDump.json', 'r',
              encoding='utf-8') as inputFile:
        code2type = json.load(inputFile)

    with open('/Users/zoe/Documents/event_extraction/majorEventDump/Class.json', 'r') as inputFile:
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
                else:
                    # countdictTest[code2type[s_event['type']]] += 1
                    # countdictTest[typeList[s_event['type']]] += 1
                    dictTest[company].append(new_event)
            except:
                # error[code2type[s_event['type']]] += 1
                s_event['type'] = '其他'

    with open('/Users/zoe/Documents/event_extraction/majorEventDump/TrainSet_time.json', 'w') as f_w:
        json.dump(dictTrain, f_w, indent=1)
    with open('/Users/zoe/Documents/event_extraction/majorEventDump/TestSet_time.json', 'w') as f_w:
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

states = [[0,1],[5,6]]
states = np.array(states)
sta = [[[0,0,0],[1,1,1]],[[5,5,5],[6,6,6]]]
sta = np.array(sta)
# print(states.shape)
# print(sta.shape)

for index,i in enumerate(states):
    print(index)
    print(i)
    for inndex in enumerate(i):
        print(inndex)

# for i in range(2):
#     restates = np.reshape(states[i],[1,-1])
#     print(restates.shape)
#     print(np.matmul(restates, sta[i]))

# 64 * 128   64 * 128 * 25
sess = tf.Session()

sess.run()
# tensor.get_shape().as_list()