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
import random


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
        if one['S_INFO_WINDCODE'] == '000418.SZ':
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

    with open('/Users/zoe/Documents/event_extraction/majorEventDump/majorEventMini.json', 'w') as outputFile:
        json.dump(events[0:10000], outputFile)

# minifile()

Chain_Lens = 5
Category = dict()

def get_xy():
    global Category
    global Chain_Lens

    with open('/Users/zoe/Documents/event_extraction/majorEventDump/majorEventMini.json', 'r',
              encoding='utf-8') as inputFile:
        events = json.load(inputFile)

    for event in events:
        if event['S_EVENT_CATEGORYCODE'] not in Category:
            Category[event['S_EVENT_CATEGORYCODE']] = len(Category)

    eventsGroupByCompany = collections.defaultdict(list)
    for event in events:
        try:
            company = event['S_INFO_WINDCODE']
            parseEvent = {
                'type': Category[event['S_EVENT_CATEGORYCODE']],
                'date': datetime.datetime.strptime(event['S_EVENT_HAPDATE'], '%Y%m%d'),
            }
            eventsGroupByCompany[company].append(parseEvent)
        except:
            continue

    x_mat_list = list()
    x_mat = np.zeros(shape=(Chain_Lens))
    y_tag_list = list()
    y_tag = np.zeros(shape=(len(Category)))
    # y_tag = np.zeros(shape=(1))

    for company, eventSeq in eventsGroupByCompany.items():
        if len(eventSeq) > Chain_Lens:
            sortedEventSeq = sorted(eventSeq, key=lambda e: e['date'])
            for beginIdx, e in enumerate(sortedEventSeq):
                if beginIdx >= Chain_Lens:
                    for i in range(Chain_Lens):
                        x_mat[Chain_Lens-i-1] = sortedEventSeq[beginIdx-i-1]['type']
                    x_mat_list.append(x_mat)
                    x_mat = np.zeros(shape=(Chain_Lens))
                    y_tag[e['type']] = 1
                    # y_tag = e['type']
                    y_tag_list.append(y_tag)
                    y_tag = np.zeros(shape=(len(Category)))
                    # y_tag = np.zeros(shape=(1))

    return np.array(x_mat_list).astype(int), np.array(y_tag_list).astype(int)

x_mat_list,y_tag_list = get_xy()
# shuffle x y
zip_list = list(zip(x_mat_list, y_tag_list))
random.shuffle(zip_list)
x_mat_list[:], y_tag_list[:] = zip(*zip_list)

lr = 0.001
training_iters = 100000
batch_size = 64

n_steps = Chain_Lens # 链条长度
n_hidden_units = 128 # 神经元数目
n_classes = len(Category)

x = tf.placeholder(tf.float32,[None, n_steps])
y = tf.placeholder(tf.float32,[None, n_classes])

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

    # => (64 batch, 128 hidden)
    X_in = tf.matmul(X, weights['in'])+biases['in']
    X_in = tf.reshape(X_in, [-1, 1, n_hidden_units])

    # cell
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)
    lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=0.5)

    _init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)
    outputs, states = tf.nn.dynamic_rnn(lstm_cell, X_in, initial_state=_init_state, time_major=False)
    # hidden layer for output as the final results

    # results = [64, n_classes]
    results = tf.matmul(states[1], weights['out']) + biases['out']
    return results

pred = RNN(x, weights, biases)
print(pred.shape)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
train_op = tf.train.AdamOptimizer(lr).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    step = 0
    while step * batch_size < training_iters:

        start = random.randint(0, x_mat_list.shape[0]-batch_size)
        batch_xs, batch_ys= x_mat_list[start:start+batch_size], y_tag_list[start:start+batch_size]

        batch_ys = np.reshape(batch_ys, [batch_size, -1])

        sess.run([train_op],feed_dict={
            x: batch_xs,
            y: batch_ys,
        })
        if step % 20 == 0:
            print(sess.run(accuracy, feed_dict={
                x: batch_xs,
                y: batch_ys
            }))
        step += 1

