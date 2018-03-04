#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Time        : 2017/11/30 下午12:20
# @Author      : Zoe
# @File        : wande-event.py
# @Description : 1. 训练测试集按2017年划分 ok
#                 2. 抽取batch的方法：只用每次shuffle的时候，training accuracy才高？？？
#                   3. Recall@10  / 换成大类 / testing accuracy = 0.581037
#                       4. self-attention LSTM model
#                           5. 求 阿尔法 ，分别基于 事件/位置/时间信息。
                # 需要修改：20 / LSTM

import json
import pickle
import datetime
import time
import collections
import numpy as np
import tensorflow as tf
import random
import matplotlib.pyplot as plt
import os,sys,getopt

# dev或test数据集
opts, args = getopt.getopt(sys.argv[1:], "t:n:c:", ["type=","note=","cf="])
trainType = 'event'
note = ''
classifier = 'mlp'
for op, value in opts:
    if op == "--type":
        trainType = value
    if op == '--note':
        note = value
    if op == '--cf':
        classifier = value

model_save_path = '../data/ckpt/'+trainType+classifier+'.ckpt'
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"

config = tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)
config.gpu_options.per_process_gpu_memory_fraction = 0.9 # 占用GPU90%的显存
config.gpu_options.allow_growth = True

def extract_one_company():
    f = open('/Users/zoe/Documents/event_extraction/majorEventDump/majorEventDump.json','r')
    a = f.read()
    f.close()

    f = open('/Users/zoe/Documents/event_extraction/majorEventDump/typeCodeDump.json','r')
    b = f.read()
    f.close()

    majorEvent = json.loads(a)
    typeCode = json.loads(b)

    # {'S_INFO_WINDCODE': '000418.SZ', 'S_EVENT_HAPDATE': '20140815', 'S_EVENT_EXPDATE': '20140815', 'S_EVENT_CATEGORYCODE': '204008001'}

    type = set()
    dic = list()

    for one in majorEvent:
        if one['S_INFO_WINDCODE'] == '601988.SH':
            d = dict()
            d['S_INFO_WINDCODE'] = one['S_INFO_WINDCODE']
            d['S_EVENT_HAPDATE'] = one['S_EVENT_HAPDATE']
            d['S_EVENT_EXPDATE'] = one['S_EVENT_EXPDATE']
            d['S_EVENT'] = typeCode[one['S_EVENT_CATEGORYCODE']]
            if typeCode[one['S_EVENT_CATEGORYCODE']] not in type:
                type.add(typeCode[one['S_EVENT_CATEGORYCODE']])
            dic.append(d)

    print(type)
    print(len(type))
    f = open('event.txt', 'w')
    for one in dic:
        f.write(json.dumps(one,ensure_ascii=False)+'\n')
    f.close()


def test():
    with open('/Users/zoe/Documents/event_extraction/majorEventDump/majorEventDump.json', 'r') as f:
        majorEvent = json.load(f)
    with open('/Users/zoe/Documents/event_extraction/majorEventDump/typeCodeDump.json', 'r') as f:
        typeCode = json.load(f)

    # {'S_INFO_WINDCODE': '000859.SZ', 'S_EVENT_HAPDATE': '20121231', 'S_EVENT_EXPDATE': '20160105',
    #  'S_EVENT_CATEGORYCODE': '204011005' 股权转让完成}
    # {'S_INFO_WINDCODE': '600610.SH', 'S_EVENT_HAPDATE': '19970418', 'S_EVENT_EXPDATE': '19901231',
    #  'S_EVENT_CATEGORYCODE': '204006021' 披露年报}
    type = set()
    dicMax = dict()
    dicMin = dict()
    maxInterval = 0
    minInterval = 0
    # for one in majorEvent:
    #     if one['S_INFO_WINDCODE'] == '600610.SH' and one['S_EVENT_HAPDATE'] == '19970418':
    #         print(one)
    #         print(typeCode[one['S_EVENT_CATEGORYCODE']])
    for one in majorEvent:
        try:
            hap = datetime.datetime.strptime(one['S_EVENT_HAPDATE'], '%Y%m%d')
            exp = datetime.datetime.strptime(one['S_EVENT_EXPDATE'], '%Y%m%d')
            if (exp-hap).days > maxInterval:
                maxInterval = (exp-hap).days
                dicMax = one
            if (exp-hap).days < minInterval:
                minInterval = (exp-hap).days
                dicMin = one
        except:
            print(one)
    print(dicMax)
    print(dicMin)


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
        print(typeClass)

    with open('/Users/zoe/Documents/event_extraction/majorEventDump/Class.json', 'w') as f_w:
        json.dump(typeClass, f_w, indent=1, ensure_ascii=False)


def minifile():
    with open('/Users/zoe/Documents/event_extraction/majorEventDump/majorEventDump.json', 'r',
              encoding='utf-8') as inputFile:
        events = json.load(inputFile)

    eventsGroupByCompany = collections.defaultdict(list)

    for event in events:
        try:
            company = event['S_INFO_WINDCODE']
            parseEvent = {
                'type': event['S_EVENT_CATEGORYCODE'],
                'date': datetime.datetime.strptime(event['S_EVENT_HAPDATE'], '%Y%m%d'),
            }
            eventsGroupByCompany[company].append(parseEvent)
        except:
            continue

### 为了统计最早最晚事件、平均事件时间间隔、事件频率 ###

    minDate = datetime.datetime.now()
    minEvent = dict()
    maxDate = datetime.datetime.strptime('20160101', '%Y%m%d')
    maxEvent = dict()
    intervalDate = 0
    eventDict = dict()
    for company, eventSeq in eventsGroupByCompany.items():
        sortedEventSeq = sorted(eventSeq, key=lambda e: e['date'])

        for index, event in enumerate(sortedEventSeq):
            if datetime.datetime.strftime(event['date'], '%Y%m%d') not in eventDict:
                eventDict[datetime.datetime.strftime(event['date'], '%Y%m%d')] = 1
            else:
                eventDict[datetime.datetime.strftime(event['date'], '%Y%m%d')] += 1
            if index > 0:
                intervalDate += (event['date']-lastDate).days
            if event['date'] < minDate:
                minDate = event['date']
                minEvent = event
                minEvent['company'] = company
            if event['date'] > maxDate:
                maxDate = event['date']
                maxEvent = event
                maxEvent['company'] = company
            lastDate = event['date']

    # with open('/Users/zoe/Documents/event_extraction/majorEventDump/eventCount.json', 'w',
    #           encoding='utf-8') as inputFile:
    #     json.dump(eventDict, inputFile, indent=1)

    print(intervalDate)
    print(minEvent)
    print(maxEvent)

### 共 3556 个公司，2342415个事件，总间隔15278034天,平均事件间隔6.52天
    # 1987 - 12 - 22  {'type': '204002012', 'date': datetime.datetime(1987, 12, 22, 0, 0)，'company': '000633.SZ'}
    # 2017 - 12 - 29  {'type': '204008004', 'date': datetime.datetime(2017, 12, 29, 0, 0)，'company': '601727.SH'}

    companyNum = 1000
    companyIndex = 0
    companyDict = dict()

    with open('/Users/zoe/Documents/event_extraction/majorEventDump/majorEventAll.json', 'w') as outputFile:
        allDict = list()
        for company, eventSeq in eventsGroupByCompany.items():
            print(company)
            # if companyIndex > companyNum:
            #     break
            companyIndex += 1
            sortedEventSeq = sorted(eventSeq, key=lambda e: e['date'])
            for e in sortedEventSeq:
                e['date'] = datetime.datetime.strftime(e['date'], '%Y%m%d')
            companyDict['S_INFO_WINDCODE'] = company
            companyDict['event'] = sortedEventSeq
            allDict.append(companyDict)
            companyDict = dict()
        json.dump(allDict, outputFile, indent=1)

# minifile()


def plot():
    with open('/Users/zoe/Documents/event_extraction/majorEventDump/eventCount.json', 'r',
              encoding='utf-8') as inputFile:
        eventCount = json.load(inputFile)

    keys = eventCount.keys()
    vals = eventCount.values()
    eventList = [(key, val) for key, val in zip(keys, vals)]
    sortedEvent = sorted(eventList, key=lambda e:e[0])
    y = [val for key,val in sortedEvent]
    print(len(y))
    x = range(0, len(y))
    plt.plot(x, y, '')
    plt.xticks((0, 2200, 4400, 6600, 8800), ('1987-12', '1995-06', '2002-12', '2010-06', '2017-12'))
    plt.xlabel('date')
    plt.ylabel('count')
    plt.title('Events per day')
    plt.show()


# plot()


def get_xy():
    global Category
    global Chain_Lens

    with open('/Users/zoe/Documents/event_extraction/majorEventDump/majorEvent50.json', 'r',
              encoding='utf-8') as inputFile:
        events = json.load(inputFile)

    with open('/Users/zoe/Documents/event_extraction/majorEventDump/typeCodeDump.json', 'r',
              encoding='utf-8') as inputFile:
        code2type = json.load(inputFile)

    with open('/Users/zoe/Documents/event_extraction/majorEventDump/Class.json', 'r') as inputFile:
        typeClass = json.load(inputFile)

### 获得按时间排序的公司事件链条
    eventsGroupByCompany = collections.defaultdict(list)
    for event in events:
        company = event['S_INFO_WINDCODE']
        parseEvent = event['event']
        eventsGroupByCompany[company] = parseEvent

    for company, eventSeq in eventsGroupByCompany.items():
        for event in eventSeq:
            if event['type'] not in Category:
                Category[event['type']] = len(Category)
            event['type'] = Category[event['type']]
            # event['date'] = datetime.datetime.strptime(event['date'], '%Y%m%d')

    x_mat_list = list()
    x_mat = np.zeros(shape=(Chain_Lens))
    y_tag_list = list()
    y_tag = np.zeros(shape=(len(Category)))
    x_test = list()
    y_test = list()

    for company, eventSeq in eventsGroupByCompany.items():
        if len(eventSeq) > Chain_Lens:
            ratio = (int)(len(eventSeq) * 0.7)
            for beginIdx, e in enumerate(eventSeq[:ratio]):
                if beginIdx >= Chain_Lens:
                    for i in range(Chain_Lens):
                        x_mat[Chain_Lens - i - 1] = eventSeq[beginIdx - i - 1]['type']
                    x_mat_list.append(x_mat)
                    x_mat = np.zeros(shape=(Chain_Lens))
                    y_tag[e['type']] = 1
                    y_tag_list.append(y_tag)
                    y_tag = np.zeros(shape=(len(Category)))

            for beginIdx, e in enumerate(eventSeq[ratio:]):
                if beginIdx >= Chain_Lens:
                    for i in range(Chain_Lens):
                        x_mat[Chain_Lens - i - 1] = eventSeq[beginIdx - i - 1]['type']
                    x_test.append(x_mat)
                    x_mat = np.zeros(shape=(Chain_Lens))
                    y_tag[e['type']] = 1
                    y_test.append(y_tag)
                    y_tag = np.zeros(shape=(len(Category)))

    return np.array(x_mat_list).astype(int), np.array(y_tag_list).astype(int),\
           np.array(x_test).astype(int),np.array(y_test).astype(int)


def get_xy_new():
    global Chain_Lens

    with open('../data/TrainSet.json', 'r',
              encoding='utf-8') as inputFile:
        eventsTrain = json.load(inputFile)
    with open('../data/TestSet.json', 'r',
              encoding='utf-8') as inputFile:
        eventsTest = json.load(inputFile)

    for company, eventSeq in eventsTest.items():
        for event in eventSeq:
            if event['type'] not in Category:
                Category[event['type']] = len(Category)
            event['type'] = Category[event['type']]
    for company, eventSeq in eventsTrain.items():
        for event in eventSeq:
            if event['type'] not in Category:
                Category[event['type']] = len(Category)
            event['type'] = Category[event['type']]

    x_mat_list = list()
    x_mat = np.zeros(shape=(Chain_Lens*2))
    y_tag_list = list()
    y_tag = np.zeros(shape=(len(Category)))
    x_test = list()
    y_test = list()

    # ********数据链条的生成********
    for company, eventSeq in eventsTrain.items():
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
                    y_tag = np.zeros(shape=(len(Category)))

    for company, eventSeq in eventsTest.items():
        if len(eventSeq) > Chain_Lens:
            for beginIdx, e in enumerate(eventSeq):
                if beginIdx >= Chain_Lens:
                    for i in range(Chain_Lens):
                        x_mat[Chain_Lens - i - 1] = eventSeq[beginIdx - i - 1]['type']
                    Start_Date = datetime.datetime.strptime(eventSeq[beginIdx - 5]['date'], '%Y%m%d')
                    for i in range(Chain_Lens):
                        This_Date = datetime.datetime.strptime(eventSeq[beginIdx - i - 1]['date'], '%Y%m%d')
                        timeDelta = 0
                        if This_Date - Start_Date < datetime.timedelta(4):
                            timeDelta = 1
                        elif This_Date - Start_Date < datetime.timedelta(8):
                            timeDelta = 2
                        elif This_Date - Start_Date < datetime.timedelta(31):
                            timeDelta = 3
                        else:
                            timeDelta = 4
                        x_mat[Chain_Lens - i + 4] = timeDelta
                    x_test.append(x_mat)
                    x_mat = np.zeros(shape=(Chain_Lens*2))
                    y_tag[e['type']] = 1
                    y_test.append(y_tag)
                    y_tag = np.zeros(shape=(len(Category)))

    return np.array(x_mat_list).astype(int), np.array(y_tag_list).astype(int),\
           np.array(x_test).astype(int),np.array(y_test).astype(int)

# x_mat_list, y_tag_list, x_test, y_test = get_xy_new()

f_data = open('../data/pickle.data.train', 'rb')
x_mat_list = pickle.load(f_data)
y_tag_list = pickle.load(f_data)
f_data.close()

print('***DATA SHAPE***\n', x_mat_list.shape, y_tag_list.shape)

# # # 生成Category.json文件。换成pickle data后，直接载入之前生成的Category.json。
# # with open('../data/Category.json', 'w', encoding='utf-8') as outputFile:
# #     json.dump(Category, outputFile, indent=1)
# with open('../data/Category.json','r') as inputFile:
#     Category = json.load(inputFile)

# shuffle x y
def shuffle_xy(x_mat_list, y_tag_list):
    zip_list = list(zip(x_mat_list, y_tag_list))
    random.shuffle(zip_list)
    x_mat_list[:], y_tag_list[:] = zip(*zip_list)
    return x_mat_list, y_tag_list


lr = 0.001
# 需要改大
epoch = 5
_batch_size = 128
training_iters = x_mat_list.shape[0]/_batch_size
vocab_size = 25    # 样本中事件类型个数，根据处理数据的时候得到
embedding_size = 20
trainNum = 100000
Chain_Lens = 5

n_steps = Chain_Lens # 链条长度
n_hidden_units = 128 # 神经元数目
n_classes = 25

x = tf.placeholder(tf.int32, [None, n_steps*2])
y = tf.placeholder(tf.int32, [None, n_classes])
output_kp = tf.placeholder(tf.float32, [])

# TODO 看一下参数的训练过程
weights = {
    # （feature_dim，128）
    'baseline': tf.Variable(tf.random_normal([n_hidden_units * 2, n_hidden_units])),
    'position': tf.Variable(tf.random_normal([n_hidden_units * 3, n_hidden_units])),
    'time': tf.Variable(tf.random_normal([n_hidden_units * 3, n_hidden_units])),
    'event': tf.Variable(tf.random_normal([n_hidden_units * 3, n_hidden_units])),

    'baseline_gcn': tf.Variable(tf.random_normal([n_hidden_units * 2+embedding_size, n_hidden_units])),
    # （128，n_classes）
    'out': tf.Variable(tf.random_normal([n_hidden_units, n_classes])),
    'out_gcn': tf.Variable(tf.random_normal([n_hidden_units, 1]))
}
biases = {
    'baseline': tf.Variable(tf.constant(0.1, shape=[n_hidden_units])),
    'position': tf.Variable(tf.constant(0.1, shape=[n_hidden_units])),
    'time': tf.Variable(tf.constant(0.1, shape=[n_hidden_units])),
    'event': tf.Variable(tf.constant(0.1, shape=[n_hidden_units])),
    # （n_classes）
    'out': tf.Variable(tf.constant(0.1, shape=[n_classes])),
}

time_v = {
    1: tf.Variable(tf.random_normal([n_hidden_units, n_hidden_units])),
    2: tf.Variable(tf.random_normal([n_hidden_units, n_hidden_units])),
    3: tf.Variable(tf.random_normal([n_hidden_units, n_hidden_units])),
    4: tf.Variable(tf.random_normal([n_hidden_units, n_hidden_units]))
}
position = {
    0: tf.Variable(tf.random_normal([n_hidden_units, n_hidden_units])),
    1: tf.Variable(tf.random_normal([n_hidden_units, n_hidden_units])),
    2: tf.Variable(tf.random_normal([n_hidden_units, n_hidden_units])),
    3: tf.Variable(tf.random_normal([n_hidden_units, n_hidden_units])),
    4: tf.Variable(tf.random_normal([n_hidden_units, n_hidden_units]))
}

event = list()
for i in range(n_classes):
    event_sub = list()
    for j in range(n_classes):
        event_sub.append(tf.Variable(tf.random_normal([n_hidden_units, n_hidden_units])))
    event.append(event_sub)


batchNum = 0
batch_xs = np.ones(shape=(_batch_size, Chain_Lens*2)).astype(int)
batch_ys = np.ones(shape=(_batch_size, Chain_Lens*2)).astype(int)


def next_batch():
    global batchNum, x_mat_list, y_tag_list
    if (batchNum + 1) * _batch_size > x_mat_list.shape[0]:
        x_mat_list, y_tag_list = shuffle_xy(x_mat_list, y_tag_list)
        batchNum = 0
    batch_x = x_mat_list[batchNum * _batch_size: (batchNum + 1) * _batch_size]
    batch_y = y_tag_list[batchNum * _batch_size: (batchNum + 1) * _batch_size]
    batchNum += 1
    return batch_x, batch_y


# TODO 特征维度 concat instead of add     128 or else?
def LSTM(X, weights, biases, time_v, position, event):
    # hidden layer for input to cell

    embedding = tf.get_variable("embedding", [vocab_size, embedding_size], dtype=tf.float32)
    X_in = tf.nn.embedding_lookup(embedding, X[:, :Chain_Lens])
    # => (64 batch, 128 hidden)

    # TODO labeling embedding 用上01矩阵
    # label embedding
    label_embedding = tf.nn.embedding_lookup(embedding, [i for i in range(n_classes)])

    adjacency_mat = pickle.load(open('../data/adjacency.regular', 'rb'))
    hidden_label_em = tf.constant([0.1])

    for i in range(label_embedding.shape[0]):
        q = tf.constant(0.1, shape=[embedding_size])
        for j in range(label_embedding.shape[0]):
            q = tf.add(q, label_embedding[j]*adjacency_mat[i][j])
            q = tf.add(q, label_embedding[j]*adjacency_mat[j][i])
        hidden_label_em = tf.concat([hidden_label_em, q], 0)
    hidden_label_em = tf.reshape(hidden_label_em[1:],[n_classes, embedding_size])

    # label embedding

    # cell
    fw_lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)
    fw_lstm_cell = tf.nn.rnn_cell.DropoutWrapper(fw_lstm_cell, output_keep_prob=output_kp)

    bw_lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)
    bw_lstm_cell = tf.nn.rnn_cell.DropoutWrapper(bw_lstm_cell, output_keep_prob=output_kp)

    fw_init_state = fw_lstm_cell.zero_state(_batch_size, dtype=tf.float32)
    bw_init_state = fw_lstm_cell.zero_state(_batch_size, dtype=tf.float32)
    outputs, states = tf.nn.bidirectional_dynamic_rnn(fw_lstm_cell, bw_lstm_cell, X_in,
            initial_state_fw=fw_init_state, initial_state_bw=bw_init_state, time_major=False)
    # outputs, states = tf.nn.dynamic_rnn(fw_lstm_cell, X_in, initial_state=fw_init_state, time_major=False)

    # ********LSTM*******
    outputs = tf.add(outputs[0], outputs[1])
    tf_results = tf.concat([states[0][1], states[1][1]], 1)
    # ********LSTM*******

    if trainType == 'position' or trainType == 'all':
        # ********position attention*******
        tf_position = tf.constant(0.1)
        for i in range(Chain_Lens):
            # batch_number * Chain_Lens * n_hidden_units  =>  按某i个Chain_Lens取数据
            result_beta = tf.reshape(tf.slice(outputs, [0, i, 0], [-1, 1, -1]), [-1, n_hidden_units])
            result_beta = tf.matmul(result_beta, position[i])
            tf_position = tf.add(tf_position, result_beta)
        # ********position attention*******
        tf_results = tf.concat([tf_results, tf_position], 1)

    if trainType == 'time' or trainType == 'all':
        # ********time attention*******
        tf_time = tf.constant(0.1)
        for i in range(Chain_Lens):
            # batch_number * Chain_Lens * n_hidden_units  =>  按某i个Chain_Lens取数据
            result_alpha = tf.reshape(tf.slice(outputs, [0, i, 0], [-1, 1, -1]), [-1, n_hidden_units])
            result_sub = tf.constant(0.1, shape=[n_hidden_units])
            for index in range(_batch_size):
                reshape_r_a = tf.reshape(result_alpha[index], [1, -1])
                result_sub = tf.concat([result_sub, tf.squeeze(tf.matmul(reshape_r_a, time_v[batch_xs[index][i+5]]))], 0)
            # batch_number * n_hidden_units
            result_sub = tf.reshape(result_sub[n_hidden_units:], [_batch_size, n_hidden_units])
            tf_time = tf.add(tf_time, result_sub)
        # ********time attention*******
        tf_results = tf.concat([tf_results, tf_time], 1)

    # TODO self-attention 只考虑前面的事件
    if trainType == 'event' or trainType == 'all':
        # ********event attention*******
        tf_event = tf.constant(0.1)
        for i in range(Chain_Lens):
            # batch_number * Chain_Lens * n_hidden_units  =>  按某i个Chain_Lens取数据
            result_event = tf.reshape(tf.slice(outputs, [0, i, 0], [-1, 1, -1]), [-1, n_hidden_units])
            result_sub = tf.constant(0.1, shape=[n_hidden_units])
            for index in range(_batch_size):
                reshape_e = tf.reshape(result_event[index], [1, -1])
                event_sum = tf.constant(0.1, shape=[n_hidden_units, n_hidden_units])
                for j in range(i):
                    tf.add(event_sum, event[batch_xs[index][i]][batch_xs[index][j]])
                result_sub = tf.concat([result_sub, tf.squeeze(tf.matmul(reshape_e, event_sum))],0)
            # batch_number * n_hidden_units
            result_sub = tf.reshape(result_sub[n_hidden_units:], [_batch_size, n_hidden_units])
            tf_event = tf.add(tf_event, result_sub)
        # ********event attention*******
        tf_results = tf.concat([tf_results, tf_event], 1)

    if classifier == 'mlp':
        # mlp classifer
        mlp_l1 = tf.matmul(tf_results, weights[trainType]) + biases[trainType]
        mlp_l2 = tf.nn.relu(mlp_l1)
        results = tf.matmul(mlp_l2, weights['out']) + biases['out']
        # mlp classifer

    # TODO 最后的GCN MLP部分
    if classifier == 'gcn':
        # gcn classifier
        tf_sequence = tf.tile(tf_results, [n_classes, 1])
        tf_label = tf.tile(hidden_label_em, [_batch_size, 1])
        tf_concat = tf.reshape(tf.concat([tf_sequence, tf_label], 1), [_batch_size * n_classes, -1])
        gcn_l1 = tf.reshape(tf.matmul(tf_concat, weights[trainType+'_gcn']),
                            [_batch_size, n_classes, -1])+biases[trainType]
        gcn_l2 = tf.nn.relu(gcn_l1)
        results = tf.reshape(tf.matmul(tf.reshape(gcn_l2, [_batch_size*n_classes,-1]), weights['out_gcn']),
                             [_batch_size, n_classes]) + biases['out']

        # gcn classifier

    return results


pred = LSTM(x, weights, biases, time_v, position, event)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
# 预测结果
pred_y = tf.cast(tf.argmax(pred, 1), tf.int32)
train_op = tf.train.AdamOptimizer(lr).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()
saver = tf.train.Saver(max_to_keep = epoch)


def test_step(data_x, data_y):
    # data_x, data_y = shuffle_xy(data_x, data_y)
    data_y = np.reshape(data_y, [_batch_size, -1])
    test_accuracy, test_cost, pred = sess.run([accuracy, cost, pred_y], feed_dict={
        x: data_x,
        y: data_y,
        output_kp: 1.0
    })
    # np.savetxt('../data/pred_y.txt', pred, fmt='%d')
    # data_y_actual = tf.cast(tf.argmax(data_y, 1), tf.int32).eval()
    # np.savetxt('../data/pred_y_actual.txt', data_y_actual, fmt='%d')

    return test_accuracy, test_cost


with tf.Session(config=config) as sess:
    # training
    sess.run(init)
    epoch_i = 0
    print('***TRAINING PROCESS***')
    with open('train_result.txt', 'a') as file:
        # file.write('{}__{}__{}__{}:\n'.format(trainType, classifier, note,
        #                                       time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
        while epoch_i < epoch:
            step = 0
            while step < training_iters:
                batch_xs, batch_ys = next_batch()
                batch_ys = np.reshape(batch_ys, [_batch_size, -1])

                _, total_cost = sess.run([train_op, cost], feed_dict={
                    x: batch_xs,
                    y: batch_ys,
                    output_kp: 0.5
                })
                if step % 1000 == 0:
                    train_accuracy = sess.run(accuracy, feed_dict={
                        x: batch_xs,
                        y: batch_ys,
                        output_kp: 0.5
                    })
                    print("step = %d, total cost = %g, training accuracy = %g" % (step,total_cost, train_accuracy))
                step += 1
            saver.save(sess, model_save_path, global_step=epoch_i)
            epoch_i += 1
            # testing
            print ('***TRAINING RESULT***EPOCH=', epoch_i)
            x_mat_list, y_tag_list = shuffle_xy(x_mat_list, y_tag_list)
            x_mat_list = x_mat_list[0:trainNum]
            y_tag_list = y_tag_list[0:trainNum]
            step = 0
            test_accuracy, test_cost = 0.0, 0.0
            while step < (trainNum / _batch_size):
                batch_xs, batch_ys = next_batch()
                batch_accuracy, batch_cost = test_step(batch_xs, batch_ys)
                test_accuracy += batch_accuracy
                test_cost += batch_cost
                step += 1
            test_accuracy /= step
            test_cost /= step
            print ("training instance = %d, total cost = %g, training accuracy = %g" % (trainNum, test_cost, test_accuracy))
            # file.write('***TRAINING RESULT***EPOCH='+str(epoch_i)+'\n')
            # file.write("training instance = %d, total cost = %g, training accuracy = %g" %
            #            (trainNum, test_cost, test_accuracy)+'\n')
