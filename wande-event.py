#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Time        : 2017/11/30 下午12:20
# @Author      : Zoe
# @File        : wande-event.py
# @Description :
import json
import datetime
import collections
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
model_save_path = 'ckpt/event.ckpt'
import random
import matplotlib.pyplot as plt

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

# test()

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

    companyNum = 50
    companyIndex = 0
    companyDict = dict()

    with open('/Users/zoe/Documents/event_extraction/majorEventDump/majorEventMini.json', 'w') as outputFile:
        allDict = list()
        for company, eventSeq in eventsGroupByCompany.items():
            print(company)
            if companyIndex > companyNum:
                break
            companyIndex += 1
            sortedEventSeq = sorted(eventSeq, key=lambda e: e['date'])
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

Chain_Lens = 5
Category = dict()

def get_xy():
    global Category
    global Chain_Lens

    with open('/Users/zoe/Documents/event_extraction/majorEventDump/majorEventMini.json', 'r',
              encoding='utf-8') as inputFile:
        events = json.load(inputFile)

    with open('/Users/zoe/Documents/event_extraction/majorEventDump/typeCodeDump.json', 'r',
              encoding='utf-8') as inputFile:
        code2type = json.load(inputFile)

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
                        x_mat[Chain_Lens-i-1] = eventSeq[beginIdx-i-1]['type']
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

x_mat_list, y_tag_list, x_test, y_test = get_xy()
print(x_mat_list.shape, y_tag_list.shape, x_test.shape, y_test.shape)


# shuffle x y
def shuffle_xy(x_mat_list, y_tag_list):
    zip_list = list(zip(x_mat_list, y_tag_list))
    random.shuffle(zip_list)
    x_mat_list[:], y_tag_list[:] = zip(*zip_list)
    return x_mat_list, y_tag_list

lr = 0.001
training_iters = 10000
_batch_size = 64
layer_num = 2        # bi-lstm 层数
vocab_size = len(Category)    # 样本中事件类型个数，根据处理数据的时候得到
input_size = embedding_size = 20

n_steps = Chain_Lens # 链条长度
n_hidden_units = 128 # 神经元数目
n_classes = len(Category)

x = tf.placeholder(tf.int32, [None, n_steps])
y = tf.placeholder(tf.int32, [None, n_classes])
batch_size = tf.placeholder(tf.int32, [])

weights = {
    # （Chain_Lens，128）
    'in': tf.Variable(tf.random_normal([n_steps,n_hidden_units])),
    # （128，len(Category）
    'out': tf.Variable(tf.random_normal([n_hidden_units, n_classes]))
}
biases = {
    'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units])),
    'out': tf.Variable(tf.constant(0.1, shape=[n_classes]))
}

# f = open('raw_file/tensor_result.txt', 'w')

def RNN(X, weights, biases):
    # hidden layer for input to cell

    embedding = tf.get_variable("embedding", [vocab_size, embedding_size], dtype=tf.float32)
    X_in = tf.nn.embedding_lookup(embedding, X)
    # => (64 batch, 128 hidden)

    # cell
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)
    lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=0.5)

    _init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)
    outputs, states = tf.nn.dynamic_rnn(lstm_cell, X_in, initial_state=_init_state, time_major=False)
    # hidden layer for output as the final results

    # results = [64, n_classes]
    results = tf.matmul(states[1], weights['out']) + biases['out']
    return results

def lstm(X, weights, biases):
    # hidden layer for input to cell
    # () => (-1, 5, 128)
    X_in = tf.matmul(X, weights['in']) + biases['in']
    embedding = tf.get_variable("embedding", [vocab_size, embedding_size], dtype=tf.float32)
    # X_inputs.shape = [batchsize, timestep_size]  ->  inputs.shape = [batchsize, timestep_size, embedding_size]
    X_in = tf.nn.embedding_lookup(embedding, X_in)

    # => (64 batch, 128 hidden)
    # X_in = tf.matmul(X, weights['in']) + biases['in']
    # X_in = tf.reshape(X_in, [-1, 1, n_hidden_units])

    stacked_fw = []
    for i in range(layer_num):
        lstm_fw_cell = rnn.BasicLSTMCell(num_units=n_hidden_units, forget_bias=1.0, state_is_tuple=True)
        stacked_fw.append(rnn.DropoutWrapper(cell=lstm_fw_cell, input_keep_prob=1.0, output_keep_prob=0.5))
    stacked_bw = []
    for i in range(layer_num):
        lstm_bw_cell = rnn.BasicLSTMCell(num_units=n_hidden_units, forget_bias=1.0, state_is_tuple=True)
        stacked_bw.append(rnn.DropoutWrapper(cell=lstm_bw_cell, input_keep_prob=1.0, output_keep_prob=0.5))

    cell_fw = rnn.MultiRNNCell(cells=stacked_fw, state_is_tuple=True)
    cell_bw = rnn.MultiRNNCell(cells=stacked_bw, state_is_tuple=True)
    # ** 4.初始状态
    initial_state_fw = cell_fw.zero_state(batch_size, tf.float32)
    initial_state_bw = cell_bw.zero_state(batch_size, tf.float32)

    outputs, _, _ = rnn.static_bidirectional_rnn(cell_fw, cell_bw, X_in,
                    initial_state_fw = initial_state_fw, initial_state_bw = initial_state_bw, dtype=tf.float32)

    # results = [64, n_classes]
    results = tf.matmul(outputs[-1], weights['out']) + biases['out']
    return results


pred = RNN(x, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
# 预测结果
pred_y = tf.cast(tf.argmax(pred, 1), tf.int32)
train_op = tf.train.AdamOptimizer(lr).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred, 1),tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()
saver = tf.train.Saver(max_to_keep = 5)


def test_step(data_x, data_y):

    data_x, data_y = shuffle_xy(data_x, data_y)
    data_y = np.reshape(data_y, [y_test.shape[0], -1])
    test_accuracy, test_cost = sess.run([accuracy, cost], feed_dict={
        x: data_x,
        y: data_y,
        batch_size: y_test.shape[0]
    })
    return test_accuracy, test_cost, [0]

    # _batch_size = 1
    # fetches = [accuracy, cost, t]
    # _y = dataset.y
    # data_size = _y.shape[0]
    # batch_num = int(data_size / _batch_size)
    # start_time = time.time()
    # _costs = 0.0
    # _accs = 0.0
    # result_pred=[]
    # for i in range(batch_num):
    #     X_batch, y_batch = dataset.next_batch(_batch_size)
    #     feed_dict = {X_inputs:X_batch, y_inputs:y_batch, lr:1e-2, batch_size:_batch_size, keep_prob:1.0}
    #     _acc, _cost, _y_pred  = sess.run(fetches, feed_dict)
    #     _accs += _acc
    #     _costs += _cost
    #     result_pred.append(_y_pred)
    # mean_acc= _accs / batch_num
    # mean_cost = _costs / batch_num
    # return mean_acc, mean_cost, result_pred


with tf.Session() as sess:
    # training
    sess.run(init)
    step = 0
    print('***TRAINING PROCESS***')
    while step * _batch_size < training_iters:
        start = random.randint(0, x_mat_list.shape[0]-_batch_size)
        x_mat_list, y_tag_list = shuffle_xy(x_mat_list, y_tag_list)
        batch_xs, batch_ys= x_mat_list[start:start+_batch_size], y_tag_list[start:start+_batch_size]
        batch_ys = np.reshape(batch_ys, [_batch_size, -1])

        _, total_cost = sess.run([train_op, cost], feed_dict={
            x: batch_xs,
            y: batch_ys,
            batch_size: 64
        })
        if step % 20 == 0:
            train_accuracy = sess.run(accuracy, feed_dict={
                x: batch_xs,
                y: batch_ys,
                batch_size: 64
            })
            print("step = %d, total cost = %g, training accuracy = %g" % (step,total_cost, train_accuracy))
            saver.save(sess, model_save_path + 'points', global_step=step)
        step += 1
    # testing
    print ('***TEST RESULT***')
    test_accuracy, test_cost, result_y = test_step(x_test, y_test)
    print ("test instance = %d, total cost = %g, training accuracy = %g" % (y_test.shape[0], test_cost, test_accuracy))
