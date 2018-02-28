#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Time        : 2018/1/16 下午2:54
# @Author      : Zoe
# @File        : wande-test.py
# @Description :  需要修改：LSTM / 258 / 260


import json
import datetime
import sys, getopt
import time
import pickle
import collections
import numpy as np
import tensorflow as tf
import random
import matplotlib.pyplot as plt
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

config = tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)
config.gpu_options.per_process_gpu_memory_fraction = 0.9 # 占用GPU90%的显存
config.gpu_options.allow_growth = True

# dev或test数据集
opts, args = getopt.getopt(sys.argv[1:], "t:d:n:", ["type=", "data=",'note='])
trainType = 'all'
data_type = ''
note = ''
for op, value in opts:
    if op == "--type":
        trainType = value
    if op == "--data":
        data_type = value
    if op == "--note":
        note = value


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

    return np.array(x_test).astype(int), np.array(y_test).astype(int)


# x_test, y_test = get_xy_new()
f_data = open('../data/pickle.data.{}'.format(data_type), 'rb')
x_test = pickle.load(f_data)
y_test = pickle.load(f_data)
f_data.close()

print('***DATA SHAPE***\n', x_test.shape, y_test.shape)


# shuffle x y
def shuffle_xy(x_mat_list, y_tag_list):
    zip_list = list(zip(x_mat_list, y_tag_list))
    random.shuffle(zip_list)
    x_mat_list[:], y_tag_list[:] = zip(*zip_list)
    return x_mat_list, y_tag_list


lr = 0.001
epoch = 10
_batch_size = 128
training_iters = y_test.shape[0] / _batch_size
vocab_size = 25  # 样本中事件类型个数，根据处理数据的时候得到
embedding_size = 20
Chain_Lens = 5

n_steps = Chain_Lens  # 链条长度
n_hidden_units = 128  # 神经元数目
n_classes = 25

x = tf.placeholder(tf.int32, [None, n_steps*2])
y = tf.placeholder(tf.int32, [None, n_classes])
output_kp = tf.placeholder(tf.float32, [])

weights = {
    # （Chain_Lens，128）
    'in': tf.Variable(tf.random_normal([n_steps, n_hidden_units])),
    # （128，len(Category）
    'out': tf.Variable(tf.random_normal([n_hidden_units, n_classes]))
}
biases = {
    'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units])),
    'out': tf.Variable(tf.constant(0.1, shape=[n_classes])),
    'alpha': tf.Variable(tf.constant(0.1, shape=[n_classes])),
    'beta': tf.Variable(tf.constant(0.1, shape=[n_classes])),
    'event': tf.Variable(tf.constant(0.1, shape=[n_classes]))
}
alpha = {
    1: tf.Variable(tf.random_normal([n_hidden_units, n_classes])),
    2: tf.Variable(tf.random_normal([n_hidden_units, n_classes])),
    3: tf.Variable(tf.random_normal([n_hidden_units, n_classes])),
    4: tf.Variable(tf.random_normal([n_hidden_units, n_classes]))
}
beta = {
    0: tf.Variable(tf.random_normal([n_hidden_units, n_classes])),
    1: tf.Variable(tf.random_normal([n_hidden_units, n_classes])),
    2: tf.Variable(tf.random_normal([n_hidden_units, n_classes])),
    3: tf.Variable(tf.random_normal([n_hidden_units, n_classes])),
    4: tf.Variable(tf.random_normal([n_hidden_units, n_classes]))
}

event = list()
for i in range(n_classes):
    event_sub = list()
    for j in range(n_classes):
        event_sub.append(tf.Variable(tf.random_normal([n_hidden_units, n_classes])))
    event.append(event_sub)

batchNum = 0
batch_xs = np.ones(shape=(_batch_size, Chain_Lens*2)).astype(int)
batch_ys = np.ones(shape=(_batch_size, Chain_Lens*2)).astype(int)


def next_batch():
    global batchNum, x_test, y_test
    if (batchNum + 1) * _batch_size > x_test.shape[0]:
        batchNum = 0
    batch_x = x_test[batchNum * _batch_size: (batchNum + 1) * _batch_size]
    batch_y = y_test[batchNum * _batch_size: (batchNum + 1) * _batch_size]
    batchNum += 1
    return batch_x, batch_y


def LSTM(X, weights, biases, beta):
    # hidden layer for input to cell
    embedding = tf.get_variable("embedding", [vocab_size, embedding_size], dtype=tf.float32)
    X_in = tf.nn.embedding_lookup(embedding, X[:, :Chain_Lens])
    # => (64 batch, 128 hidden)

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

    outputs = tf.add(outputs[0], outputs[1])
    results = tf.constant(0.1)
    # ********LSTM*******
    results = tf.add(results, tf.matmul(tf.add(states[0][1], states[1][1]), weights['out']) + biases['out'])
    # ********LSTM*******

    if trainType == 'position' or trainType == 'all':
        # ********position attention*******
        # outputs = tf.add(outputs[0], outputs[1])
        # results = tf.constant(0.1)

        for i in range(5):
            # batch_number * Chain_Lens * n_hidden_units  =>  按某i个Chain_Lens取数据
            result_beta = tf.reshape(tf.slice(outputs, [0, i, 0], [-1, 1, -1]), [-1, n_hidden_units])
            result_beta = tf.matmul(result_beta, beta[i]) + biases['beta']
            results = tf.add(results, result_beta)
        # ********position attention*******

    if trainType == 'time' or trainType == 'all':
        # ********time attention*******
        # outputs = tf.add(outputs[0], outputs[1])
        # results = tf.constant(0.1)

        for i in range(Chain_Lens):
            # batch_number * Chain_Lens * n_hidden_units  =>  按某i个Chain_Lens取数据
            result_alpha = tf.reshape(tf.slice(outputs, [0, i, 0], [-1, 1, -1]), [-1, n_hidden_units])
            result_sub = tf.constant(0.1, shape=[n_classes])
            for index in range(_batch_size):
                reshape_r_a = tf.reshape(result_alpha[index], [1, -1])
                # 问题：由于X_batch[index]的真实数据拿不到，得不到alpha[]的值
                result_sub = tf.concat([result_sub, tf.squeeze(tf.matmul(reshape_r_a, alpha[batch_xs[index][i+5]]) + biases['alpha'])], 0)
            # batch_number * n_classes
            result_sub = tf.reshape(result_sub[n_classes:], [_batch_size, n_classes])
            results = tf.add(results, result_sub)
        # ********time attention*******

    if trainType == 'event' or trainType == 'all':
        # ********event attention*******
        # outputs = tf.add(outputs[0], outputs[1])
        # results = tf.constant(0.1)

        adjacency_mat = pickle.load(open('../data/adjacency.regular', 'rb'))

        assist_list = [i for i in range(5)]
        for i in range(Chain_Lens):
            # batch_number * Chain_Lens * n_hidden_units  =>  按某i个Chain_Lens取数据
            result_event = tf.reshape(tf.slice(outputs, [0, i, 0], [-1, 1, -1]), [-1, n_hidden_units])
            result_sub = tf.constant(0.1, shape=[n_classes])
            for index in range(_batch_size):
                reshape_e = tf.reshape(result_event[index], [1, -1])
                assist_list.remove(i)
                event_sum = tf.constant(0.1, shape=[n_hidden_units, n_classes])
                for j in assist_list:
                    weight = adjacency_mat[batch_xs[index][i]][batch_xs[index][j]]
                    tf.add(event_sum, event[batch_xs[index][i]][batch_xs[index][j]] * weight)
                result_sub = tf.concat(
                    [result_sub, tf.squeeze(tf.matmul(reshape_e, event_sum) + biases['event'])],
                    0)
                assist_list.append(i)
            # batch_number * n_classes
            result_sub = tf.reshape(result_sub[n_classes:], [_batch_size, n_classes])
            results = tf.add(results, result_sub)
        # ********event attention*******

    return results


pred = LSTM(x, weights, biases, beta)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
# 预测结果
pred_y = tf.cast(tf.argmax(pred, 1), tf.int32)
# train_op = tf.train.AdamOptimizer(lr).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


init = tf.global_variables_initializer()
saver = tf.train.Saver(max_to_keep = epoch)


def test_step(data_x, data_y, step):
    # data_x, data_y = shuffle_xy(data_x, data_y)
    data_y = np.reshape(data_y, [_batch_size, -1])
    test_accuracy, test_cost, pred = sess.run([accuracy, cost, pred_y], feed_dict={
        x: data_x,
        y: data_y,
        output_kp: 1.0
    })

    np.savetxt('../data/pred/pred_y_{}.txt'.format(step), pred, fmt='%d')
    # data_y_actual = tf.cast(tf.argmax(data_y, 1), tf.int32).eval()
    data_y_actual = np.argmax(data_y, 1)
    np.savetxt('../data//pred/pred_y_actual_{}.txt'.format(step), data_y_actual, fmt='%d')

    return test_accuracy, test_cost


with tf.Session(config=config) as sess:
    # training
    sess.run(init)
    with open('../data/{}_result.txt'.format(data_type), 'a') as file:
        file.write(trainType+'  '+note+'   '+time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ':\n')
        for i in range(epoch):
            saver.restore(sess, 'ckpt/{}.ckpt-{}'.format(trainType, i))
            # testing
            print('***TEST RESULT***')
            step = 0
            test_accuracy, test_cost = 0.0, 0.0
            while step < training_iters:
                batch_xs, batch_ys = next_batch()
                batch_accuracy, batch_cost = test_step(batch_xs, batch_ys, step)
                test_accuracy += batch_accuracy
                test_cost += batch_cost
                step += 1
            test_accuracy /= step
            test_cost /= step
            print("test instance = %d, total cost = %g, testing accuracy = %g" % (y_test.shape[0], test_cost, test_accuracy))
            file.write('***TESTING RESULT***EPOCH=' + str(i+1) +'\n')
            file.write("testing instance = %d, total cost = %g, testing accuracy = %g" %
                       (y_test.shape[0], test_cost, test_accuracy) + '\n')
        file.write('\n')
