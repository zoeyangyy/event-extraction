#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Time        : 2018/1/16 下午2:54
# @Author      : Zoe
# @File        : wande-test.py
# @Description :


import json
import datetime
import sys, getopt
import time
import pickle
import collections
import numpy as np
import tensorflow as tf
import config
import random
import matplotlib.pyplot as plt
import os
from sklearn import preprocessing

# dev或test数据集
opts, args = getopt.getopt(sys.argv[1:], "t:d:n:c:v:p:", ["type=", "data=",'note=','cf=','cuda=','top='])
trainType = 'all'
data_type = 'test'
note = ''
classifier = 'mlp'
cuda = '1'
topN = 1
for op, value in opts:
    if op == "--type":
        trainType = value
    if op == "--data":
        data_type = value
    if op == "--note":
        note = value
    if op == '--cf':
        classifier = value
    if op == '--cuda':
        cuda = value
    if op == '--top':
        topN = int(value)

os.environ["CUDA_VISIBLE_DEVICES"] = cuda

config = tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)
config.gpu_options.per_process_gpu_memory_fraction = 0.9 # 占用GPU90%的显存
config.gpu_options.allow_growth = True

flags = tf.app.flags
FLAGS = flags.FLAGS


f_data = open('../data/pickle.data.5.{}'.format(data_type), 'rb')
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


training_iters = y_test.shape[0] / FLAGS._batch_size

x = tf.placeholder(tf.int32, [None, FLAGS.n_steps*2])
y = tf.placeholder(tf.int32, [None, FLAGS.n_classes])
output_kp = tf.placeholder(tf.float32, [])

weights = {
    # （feature_dim，128）
    'weight_add': tf.Variable(tf.random_normal([FLAGS.n_hidden_units, FLAGS.n_hidden_units])),
    'baseline_gcn': tf.Variable(tf.random_normal([FLAGS.n_hidden_units +FLAGS.embedding_size, FLAGS.n_hidden_units])),
    'gcn_2': tf.Variable(tf.random_normal([FLAGS.n_hidden_units, FLAGS.n_hidden_units])),
    'gcn_3': tf.Variable(tf.random_normal([FLAGS.n_hidden_units, FLAGS.n_hidden_units])),

    'attention': tf.Variable(tf.random_normal([FLAGS.n_hidden_units, FLAGS.n_hidden_units])),
    'attention_2': tf.Variable(tf.random_normal([FLAGS.n_hidden_units, 1])),
    # （128，n_classes）
    'out': tf.Variable(tf.random_normal([FLAGS.n_hidden_units, FLAGS.n_classes])),
    'out_gcn': tf.Variable(tf.random_normal([FLAGS.n_hidden_units, 1]))
}
biases = {
    'l1': tf.Variable(tf.constant(0.1, shape=[FLAGS.n_hidden_units])),
    'l2': tf.Variable(tf.constant(0.1, shape=[FLAGS.n_hidden_units])),
    'attention': tf.Variable(tf.constant(0.1, shape=[FLAGS.n_hidden_units])),
    # （n_classes）
    'out': tf.Variable(tf.constant(0.1, shape=[FLAGS.n_classes])),
}
add_weights = {
    'baseline': tf.Variable(tf.constant(0.25)),
    'position': tf.Variable(tf.constant(0.25)),
    'time': tf.Variable(tf.constant(0.25)),
    'event': tf.Variable(tf.constant(0.25))
}

time_v = tf.get_variable('time', [4])
position = tf.get_variable('position', [5])
event = tf.get_variable('event', [FLAGS.n_classes, FLAGS.n_classes])

baseline_gcn = list()
for _ in range(FLAGS.n_classes):
    baseline_gcn.append(tf.Variable(tf.random_normal([FLAGS.n_hidden_units + FLAGS.embedding_size, FLAGS.n_hidden_units])))


batchNum = 0
batch_xs = np.ones(shape=(FLAGS._batch_size, FLAGS.Chain_Lens*2)).astype(int)
batch_ys = np.ones(shape=(FLAGS._batch_size, FLAGS.Chain_Lens*2)).astype(int)


def next_batch():
    global batchNum, x_test, y_test
    if (batchNum + 1) * FLAGS._batch_size > x_test.shape[0]:
        batchNum = 0
    batch_x = x_test[batchNum * FLAGS._batch_size: (batchNum + 1) * FLAGS._batch_size]
    batch_y = y_test[batchNum * FLAGS._batch_size: (batchNum + 1) * FLAGS._batch_size]
    batchNum += 1
    return batch_x, batch_y


def LSTM(X, weights, biases, time_v, position, event):
    # hidden layer for input to cell

    embedding = tf.get_variable("embedding", [FLAGS.vocab_size, FLAGS.embedding_size], dtype=tf.float32)
    X_in = tf.nn.embedding_lookup(embedding, X[:, :FLAGS.Chain_Lens])
    # => (64 batch, 128 hidden)

    # cell
    def unit_lstm():
        fw_lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(FLAGS.n_hidden_units, forget_bias=1.0, state_is_tuple=True)
        fw_lstm_cell = tf.nn.rnn_cell.DropoutWrapper(fw_lstm_cell, output_keep_prob=output_kp)
        return fw_lstm_cell

    fw_cell = tf.nn.rnn_cell.MultiRNNCell([unit_lstm() for i in range(3)], state_is_tuple=True)
    fw_init_state = fw_cell.zero_state(FLAGS._batch_size, dtype=tf.float32)
    outputs, states = tf.nn.dynamic_rnn(fw_cell, X_in, initial_state=fw_init_state, time_major=False)

    # ********LSTM*******

    tf_results = tf.constant(0.0001)
    tf_baseline = tf.constant(0.0001)
    for i in range(FLAGS.Chain_Lens):
        # batch_number * Chain_Lens * n_hidden_units  =>  按某i个Chain_Lens取数据
        result_beta = tf.reshape(tf.slice(outputs, [0, i, 0], [-1, 1, -1]), [-1, FLAGS.n_hidden_units])
        result_beta = result_beta * (1 / FLAGS.Chain_Lens)
        tf_baseline = tf.add(tf_baseline, result_beta)
    # tf_results = tf.add(tf_results, tf_baseline * add_weights['baseline'])
    # tf_results = tf.add(tf_results, tf_baseline)
    tf_results = tf.add(tf_results, tf_baseline * (1-add_weights['position']-add_weights['time']-add_weights['event']))

    # tf_results = states[1]
    # ********LSTM*******

    if trainType == 'attention':
        # ********attention*******
        tf_attention = tf.constant(0.1, shape=[FLAGS._batch_size, 1])
        for i in range(FLAGS.Chain_Lens):
            result_beta = tf.reshape(tf.slice(outputs, [0, i, 0], [-1, 1, -1]), [-1, FLAGS.n_hidden_units])
            result_beta = tf.nn.tanh(tf.matmul(result_beta, weights['attention']) + biases['attention'])
            tf_attention = tf.concat([tf_attention, result_beta],1)
        tf_attention = tf.reshape(tf.slice(tf_attention, [0, 1], [-1,-1]), [FLAGS._batch_size, FLAGS.Chain_Lens, -1])

        tf_other = tf.constant(0.001, shape=[1])
        for i in range(FLAGS._batch_size):
            soft = tf.reshape(tf.nn.softmax(tf.squeeze(tf.matmul(tf_attention[i], weights['attention_2']))),[-1,1])
            tf_other = tf.concat([tf_other, tf.reshape(tf.matmul(tf.transpose(outputs[i]), soft), [-1])], 0)
        tf_other = tf.reshape(tf_other[1:], [FLAGS._batch_size, -1])
        # ********attention*******
        tf_results = tf.add(tf_results, tf_other)

    if trainType == 'position' or trainType == 'all':
        # ********position attention*******
        tf_position = tf.constant(0.0001)
        for i in range(FLAGS.Chain_Lens):
            # batch_number * Chain_Lens * n_hidden_units  =>  按某i个Chain_Lens取数据
            result_beta = tf.reshape(tf.slice(outputs, [0, i, 0], [-1, 1, -1]), [-1, FLAGS.n_hidden_units])
            # result_beta = tf.matmul(result_beta, position[i])
            result_beta = result_beta * position[i]
            tf_position = tf.add(tf_position, result_beta)
        # ********position attention*******
        # tf_results = tf.concat([tf_results, tf_position], 1)
        tf_results = tf.add(tf_results, tf_position*add_weights['position'])

    if trainType == 'time' or trainType == 'all':
        # ********time attention*******
        tf_time = tf.constant(0.0001)
        for i in range(FLAGS.Chain_Lens):
            # batch_number * Chain_Lens * n_hidden_units  =>  按某i个Chain_Lens取数据
            result_alpha = tf.reshape(tf.slice(outputs, [0, i, 0], [-1, 1, -1]), [-1, FLAGS.n_hidden_units])
            result_sub = tf.constant(0.1, shape=[FLAGS.n_hidden_units])
            for index in range(FLAGS._batch_size):
                reshape_r_a = tf.reshape(result_alpha[index], [1, -1])
                # result_sub = tf.concat([result_sub, tf.squeeze(tf.matmul(reshape_r_a, time_v[batch_xs[index][i+5]]))], 0)
                result_sub = tf.concat([result_sub, result_alpha[index] * time_v[X[index][i + 5] - 1]], 0)
            # batch_number * n_hidden_units
            result_sub = tf.reshape(result_sub[FLAGS.n_hidden_units:], [FLAGS._batch_size, FLAGS.n_hidden_units])
            tf_time = tf.add(tf_time, result_sub)
        # ********time attention*******
        # tf_results = tf.concat([tf_results, tf_time], 1)
        tf_results = tf.add(tf_results, tf_time* add_weights['time'])

    if trainType == 'event' or trainType == 'all':
        # ********event attention*******
        tf_event = tf.constant(0.0001)
        for i in range(FLAGS.Chain_Lens):
            # batch_number * Chain_Lens * n_hidden_units  =>  按某i个Chain_Lens取数据
            result_event = tf.reshape(tf.slice(outputs, [0, i, 0], [-1, 1, -1]), [-1, FLAGS.n_hidden_units])
            result_sub = tf.constant(0.0001, shape=[FLAGS.n_hidden_units])
            for index in range(FLAGS._batch_size):
                event_sum = tf.constant(0.0001)
                for j in range(i):
                    tf.add(event_sum, event[X[index][i]][X[index][j]])
                result_sub = tf.concat([result_sub, result_event[index] * event_sum],0)
            # batch_number * n_hidden_units
            result_sub = tf.reshape(result_sub[FLAGS.n_hidden_units:], [FLAGS._batch_size, FLAGS.n_hidden_units])
            tf_event = tf.add(tf_event, result_sub)
        # ********event attention*******
        # tf_results = tf.concat([tf_results, tf_event], 1)
        tf_results = tf.add(tf_results, tf_event*add_weights['event'])

    if classifier == 'mlp':
        # mlp classifer
        # mlp_l1 = tf.matmul(tf_results, weights[trainType]) + biases['l1']
        mlp_l1 = tf.matmul(tf_results, weights['weight_add']) + biases['l1']
        mlp_l2 = tf.nn.relu(mlp_l1)
        results = tf.matmul(mlp_l2, weights['out']) + biases['out']
        # mlp classifer

    label_embedding = tf.nn.embedding_lookup(embedding, [i for i in range(FLAGS.n_classes)])


    if classifier == 'gcn':
        # gcn classifier
        tf_sequence = tf.reshape(tf.tile(tf_results, [1, FLAGS.n_classes]), [FLAGS._batch_size * FLAGS.n_classes, -1])
        tf_label = tf.tile(label_embedding, [FLAGS._batch_size, 1])
        tf_concat = tf.reshape(tf.concat([tf_sequence, tf_label], 1), [FLAGS._batch_size, FLAGS.n_classes, -1])
        # gcn_l1 = tf.reshape(tf.matmul(tf_concat, weights[trainType+'_gcn']),[_batch_size, n_classes, -1])+biases[trainType]

        adjacency_mat = pickle.load(open('../data/adjacency.new', 'rb'))
        myarray = np.zeros((25, 25), dtype='float32')
        for key1, row in adjacency_mat.items():
            for key2, value in row.items():
                myarray[key1, key2] = value
        # X_scaled = preprocessing.scale(myarray)

        gcn_l1 = tf.constant(0.1, shape=[FLAGS.n_classes, FLAGS.n_hidden_units])
        for i in range(FLAGS._batch_size):
            # gcn_beta = tf.matmul(tf.matmul(X_scaled, tf_concat[i]), weights['baseline_gcn']) + biases['l1']
            gcn_beta = tf.matmul(tf.matmul(myarray, tf_concat[i]), weights['baseline_gcn']) + biases['l1']
            # gcn_beta = tf.nn.relu(gcn_beta)
            # gcn_beta = tf.matmul(tf.matmul(myarray, gcn_beta), weights['gcn_2']) + biases['l2']
            # gcn_beta = tf.matmul(tf.matmul(myarray, gcn_beta), weights['gcn_2'])
            # gcn_beta = tf.matmul(tf.matmul(myarray, gcn_beta), weights['gcn_3'])
            gcn_l1 = tf.concat([gcn_l1, gcn_beta], 0)
        gcn_l1 = tf.reshape(gcn_l1[FLAGS.n_classes:], shape=[FLAGS._batch_size, FLAGS.n_classes, -1])

        gcn_l2 = tf.nn.relu(gcn_l1)
        results = tf.reshape(tf.matmul(tf.reshape(gcn_l2, [FLAGS._batch_size*FLAGS.n_classes,-1]), weights['out_gcn']),
                             [FLAGS._batch_size, FLAGS.n_classes]) + biases['out']

        # gcn classifier

    return results


pred = LSTM(x, weights, biases, time_v, position, event)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
# 预测结果
pred_y = tf.cast(tf.argmax(pred, 1), tf.int32)
# train_op = tf.train.AdamOptimizer(lr).minimize(cost)

k = topN  # targets对应的索引是否在最大的前k个数据中
output = tf.nn.in_top_k(pred, tf.argmax(y, 1), k)
accuracy = tf.reduce_mean(tf.cast(output, tf.float32))

# correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


init = tf.global_variables_initializer()
saver = tf.train.Saver()


def test_step(data_x, data_y, step):
    # data_x, data_y = shuffle_xy(data_x, data_y)
    data_y = np.reshape(data_y, [FLAGS._batch_size, -1])
    test_accuracy, test_cost, pred = sess.run([accuracy, cost, pred_y], feed_dict={
        x: data_x,
        y: data_y,
        output_kp: 1.0
    })

    # np.savetxt('../data/pred/pred_y_{}.txt'.format(step), pred, fmt='%d')
    # # data_y_actual = tf.cast(tf.argmax(data_y, 1), tf.int32).eval()
    # data_y_actual = np.argmax(data_y, 1)
    # np.savetxt('../data/pred/pred_y_actual_{}.txt'.format(step), data_y_actual, fmt='%d')

    return test_accuracy, test_cost


with tf.Session(config=config) as sess:
    # training
    sess.run(init)
    with open('{}_result_3_15.txt'.format(data_type), 'a') as file:
        file.write('\n{}__{}__top:{}__hidden_units:{}__lr:{}__batch:{}__embedding:{}__{}:\n'.format(trainType, classifier, topN,
            FLAGS.n_hidden_units, FLAGS.lr,FLAGS._batch_size,FLAGS.embedding_size,time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
        # file.write(trainType+'  '+note+'   '+time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ':\n')
        for i in range(0, FLAGS.epoch):
        # for i in range(5, 10):
            for sub_i in range(0, 13164, 1000):
                saver.restore(sess, '../data/ckpt-gcn/{}{}{}.ckpt-{}'.format(trainType, classifier, note, sub_i+i))
                # testing
                print('***TEST RESULT***step={}***{}***{}'.format(sub_i+i, trainType, classifier))
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
                print("{}__test instance = {}, total cost = {:.5f}, testing accuracy = {:.5f}".format(
                    time.strftime("%H:%M:%S", time.localtime()), y_test.shape[0], test_cost.item(), test_accuracy.item()))
                # file.write('***TESTING RESULT***EPOCH=' + str(i) +'\n')
                # file.write("testing instance = %d, total cost = %g, testing accuracy = %g" %
                #            (y_test.shape[0], test_cost, test_accuracy) + '\n')
                file.write("%g" % test_accuracy + '\n')
