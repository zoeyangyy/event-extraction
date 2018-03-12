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
# dev或test数据集
opts, args = getopt.getopt(sys.argv[1:], "t:d:n:c:v:", ["type=", "data=",'note=','cf=','cuda='])
trainType = 'all'
data_type = 'test'
note = ''
classifier = 'mlp'
cuda = '1'
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

os.environ["CUDA_VISIBLE_DEVICES"] = cuda

config = tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)
config.gpu_options.per_process_gpu_memory_fraction = 0.9 # 占用GPU90%的显存
config.gpu_options.allow_growth = True

flags = tf.app.flags
FLAGS = flags.FLAGS


x = tf.placeholder(tf.int32, [None, FLAGS.n_steps*2])
y = tf.placeholder(tf.int32, [None, FLAGS.n_classes])
output_kp = tf.placeholder(tf.float32, [])

weights = {
    # （feature_dim，128）
    'weight_add': tf.Variable(tf.random_normal([FLAGS.n_hidden_units, FLAGS.n_hidden_units])),
    # 'baseline': tf.Variable(tf.random_normal([n_hidden_units * 2, n_hidden_units])),
    # 'position': tf.Variable(tf.random_normal([n_hidden_units * 3, n_hidden_units])),
    # 'time': tf.Variable(tf.random_normal([n_hidden_units * 3, n_hidden_units])),
    # 'event': tf.Variable(tf.random_normal([n_hidden_units * 3, n_hidden_units])),

    'baseline_gcn': tf.Variable(tf.random_normal([FLAGS.n_hidden_units +FLAGS.embedding_size, FLAGS.n_hidden_units])),
    # （128，n_classes）
    'out': tf.Variable(tf.random_normal([FLAGS.n_hidden_units, FLAGS.n_classes])),
    'out_gcn': tf.Variable(tf.random_normal([FLAGS.n_hidden_units, 1]))
}
biases = {
    'l1': tf.Variable(tf.constant(0.1, shape=[FLAGS.n_hidden_units])),
    # （n_classes）
    'out': tf.Variable(tf.constant(0.1, shape=[FLAGS.n_classes])),
}
add_weights = {
    'position': tf.Variable(tf.constant(0.1)),
    'time': tf.Variable(tf.constant(0.1)),
    'event': tf.Variable(tf.constant(0.1))
}

time_v = {
    1: tf.Variable(tf.random_normal([FLAGS.n_hidden_units, FLAGS.n_hidden_units])),
    2: tf.Variable(tf.random_normal([FLAGS.n_hidden_units, FLAGS.n_hidden_units])),
    3: tf.Variable(tf.random_normal([FLAGS.n_hidden_units, FLAGS.n_hidden_units])),
    4: tf.Variable(tf.random_normal([FLAGS.n_hidden_units, FLAGS.n_hidden_units]))
}
position = {
    0: tf.Variable(tf.random_normal([FLAGS.n_hidden_units, FLAGS.n_hidden_units])),
    1: tf.Variable(tf.random_normal([FLAGS.n_hidden_units, FLAGS.n_hidden_units])),
    2: tf.Variable(tf.random_normal([FLAGS.n_hidden_units, FLAGS.n_hidden_units])),
    3: tf.Variable(tf.random_normal([FLAGS.n_hidden_units, FLAGS.n_hidden_units])),
    4: tf.Variable(tf.random_normal([FLAGS.n_hidden_units, FLAGS.n_hidden_units]))
}

event = list()
for i in range(FLAGS.n_classes):
    event_sub = list()
    for j in range(FLAGS.n_classes):
        event_sub.append(tf.Variable(tf.random_normal([FLAGS.n_hidden_units, FLAGS.n_hidden_units])))
    event.append(event_sub)

baseline_gcn = list()
for _ in range(FLAGS.n_classes):
    baseline_gcn.append(tf.Variable(tf.random_normal([FLAGS.n_hidden_units +FLAGS.embedding_size, FLAGS.n_hidden_units])))

embedding = tf.get_variable("embedding", [FLAGS.vocab_size, FLAGS.embedding_size], dtype=tf.float32)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session(config=config) as sess:
    sess.run(init)

    with open('testing_result.txt', 'a') as file:
        for i in range(81000, 16164*FLAGS.epoch, 1000):
            saver.restore(sess, '../data/ckpt/{}{}.ckpt-{}'.format(trainType, classifier, i))
            w = sess.run(embedding)
            print(w)
            file.write("add_weights : {}\n".format(w))
