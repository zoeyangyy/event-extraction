#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Time        : 2018/1/16 下午2:54
# @Author      : Zoe
# @File        : wande-test.py
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


Chain_Lens = 5
Category = dict()


def get_xy_new():
    global Chain_Lens

    with open('/Users/zoe/Documents/event_extraction/majorEventDump/TrainSet.json', 'r',
              encoding='utf-8') as inputFile:
        eventsTrain = json.load(inputFile)
    with open('/Users/zoe/Documents/event_extraction/majorEventDump/TestSet.json', 'r',
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
    x_mat = np.zeros(shape=(Chain_Lens))
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
                    x_test.append(x_mat)
                    x_mat = np.zeros(shape=(Chain_Lens))
                    y_tag[e['type']] = 1
                    y_test.append(y_tag)
                    y_tag = np.zeros(shape=(len(Category)))

    return np.array(x_test).astype(int), np.array(y_test).astype(int)


x_test, y_test = get_xy_new()
print('***DATA SHAPE***\n', x_test.shape, y_test.shape)

# shuffle x y
def shuffle_xy(x_mat_list, y_tag_list):
    zip_list = list(zip(x_mat_list, y_tag_list))
    random.shuffle(zip_list)
    x_mat_list[:], y_tag_list[:] = zip(*zip_list)
    return x_mat_list, y_tag_list

lr = 0.001
vocab_size = len(Category)  # 样本中事件类型个数，根据处理数据的时候得到
input_size = embedding_size = 20

n_steps = Chain_Lens  # 链条长度
n_hidden_units = 128  # 神经元数目
n_classes = len(Category)

x = tf.placeholder(tf.int32, [None, n_steps])
y = tf.placeholder(tf.int32, [None, n_classes])
batch_size = tf.placeholder(tf.int32, [])
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
    'beta': tf.Variable(tf.constant(0.1, shape=[n_classes])),
}
alpha = {
    '3_day': tf.Variable(tf.random_normal([n_hidden_units, n_classes])),
    '1_week': tf.Variable(tf.random_normal([n_hidden_units, n_classes])),
    '1_month': tf.Variable(tf.random_normal([n_hidden_units, n_classes])),
    'other': tf.Variable(tf.random_normal([n_hidden_units, n_classes])),
}
beta = {
    0: tf.Variable(tf.random_normal([n_hidden_units, n_classes])),
    1: tf.Variable(tf.random_normal([n_hidden_units, n_classes])),
    2: tf.Variable(tf.random_normal([n_hidden_units, n_classes])),
    3: tf.Variable(tf.random_normal([n_hidden_units, n_classes])),
    4: tf.Variable(tf.random_normal([n_hidden_units, n_classes]))
}


def LSTM(X, weights, biases, beta):
    # hidden layer for input to cell

    embedding = tf.get_variable("embedding", [vocab_size, embedding_size], dtype=tf.float32)
    X_in = tf.nn.embedding_lookup(embedding, X)
    # => (64 batch, 128 hidden)

    # cell
    fw_lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)
    fw_lstm_cell = tf.nn.rnn_cell.DropoutWrapper(fw_lstm_cell, output_keep_prob=output_kp)

    bw_lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)
    bw_lstm_cell = tf.nn.rnn_cell.DropoutWrapper(bw_lstm_cell, output_keep_prob=output_kp)

    fw_init_state = fw_lstm_cell.zero_state(batch_size, dtype=tf.float32)
    bw_init_state = fw_lstm_cell.zero_state(batch_size, dtype=tf.float32)
    outputs, states = tf.nn.bidirectional_dynamic_rnn(fw_lstm_cell, bw_lstm_cell, X_in,
                                                      initial_state_fw=fw_init_state, initial_state_bw=bw_init_state,
                                                      time_major=False)
    # ********LSTM*******
    # outputs, states = tf.nn.dynamic_rnn(fw_lstm_cell, X_in, initial_state=fw_init_state, time_major=False)
    # results = tf.matmul(tf.add(states[0][1], states[1][1]), weights['out']) + biases['out']
    # ********LSTM*******

    outputs = tf.add(outputs[0], outputs[1])
    results = tf.constant(0.1)

    # ********position attention*******
    for i in range(5):
        # batch_number * Chain_Lens * n_hidden_units  =>  按某i个Chain_Lens取数据
        result_beta = tf.reshape(tf.slice(outputs, [0, i, 0], [-1, 1, -1]), [-1, n_hidden_units])
        result_beta = tf.matmul(result_beta, beta[i]) + biases['beta']
        results = tf.add(results, result_beta)



    return results


pred = LSTM(x, weights, biases, beta)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
# 预测结果
pred_y = tf.cast(tf.argmax(pred, 1), tf.int32)
train_op = tf.train.AdamOptimizer(lr).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()
saver = tf.train.Saver(max_to_keep=5)


def test_step(data_x, data_y):
    data_x, data_y = shuffle_xy(data_x, data_y)
    data_y = np.reshape(data_y, [y_test.shape[0], -1])
    test_accuracy, test_cost, pred = sess.run([accuracy, cost, pred_y], feed_dict={
        x: data_x,
        y: data_y,
        batch_size: y_test.shape[0],
        output_kp: 1.0
    })
    np.savetxt('/Users/zoe/Documents/event_extraction/majorEventDump/pred_y.txt', pred, fmt='%d')
    data_y_actual = tf.cast(tf.argmax(data_y, 1), tf.int32).eval()
    np.savetxt('/Users/zoe/Documents/event_extraction/majorEventDump/pred_y_actual.txt', data_y_actual, fmt='%d')

    return test_accuracy, test_cost


with tf.Session() as sess:
    # training
    sess.run(init)
    # model_file = tf.train.latest_checkpoint('ckpt/')
    for i in range(5):
        saver.restore(sess, 'ckpt/event_position.ckpt-'+str(i))
        # testing
        print('***TEST RESULT***')
        test_accuracy, test_cost = test_step(x_test, y_test)
        print("test instance = %d, total cost = %g, testing accuracy = %g" % (y_test.shape[0], test_cost, test_accuracy))
        with open('testing_result.txt', 'a') as file:
            file.write('***TESTING RESULT***EPOCH=' + str(i+1) +'\n')
            file.write("testing instance = %d, total cost = %g, testing accuracy = %g" %
                       (y_test.shape[0], test_cost, test_accuracy) + '\n')