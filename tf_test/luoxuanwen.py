#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Time        : 2018/1/31 下午2:57
# @Author      : Zoe
# @File        : luoxuanwen.py
# @Description :

import tensorflow as tf
import csv
import numpy as np

fut_file = csv.reader(open('luoxuan.csv', 'r'))
print(fut_file)
data_all = list()
for stu in fut_file:
    data_all.append(stu)
# print(data_all[:5])
price_list = list()

for item in data_all[1:]:
    end_price = float(item[1]) - float(item[4])
    top_price = float(item[2]) - float(item[3])
    price_list.append([end_price, top_price])

price = np.array(price_list, dtype=np.float)
# print(price.shape)
seq_length = 5
data_dim = 2
iterater = 500
x_data = list()
y_data = list()

for index, item in enumerate(price):

    if index + 4 < len(price):
        x_data.append([price[index:index + 4]])

        y_data.append([1, 0] if price[index + 4][0] > 0 else [0, 1])

# print(y_data)
# print(len(x_data))
x = tf.placeholder(tf.float32, [None, seq_length, 2])
y = tf.placeholder(tf.float32, [None, 2])

# 建立网络
i_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=10, state_is_tuple=True, activation=tf.tanh)
outputs, _state = tf.nn.dynamic_rnn(i_cell, x, dtype=tf.float32)
# 预测值
y_pred = tf.contrib.layers.fully_connected(outputs[:, -1], num_outputs=2, activation_fn=None)  # list不能[:,-1],np可以
# 计算loss
loss = tf.reduce_sum(tf.square(y_pred - y))

# 选优化方法
optimizer = tf.train.AdamOptimizer(0.01)
train = optimizer.minimize(loss)

# 算均方误差
targets = tf.placeholder(tf.float32, [None, 1])
predictions = tf.placeholder(tf.float32, [None, 1])
rmse = tf.sqrt(tf.reduce_mean(tf.square(targets - predictions)))

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    # 训练步
    for i in range(iterater):
        _, step_loss = sess.run([train, loss], feed_dict={x: x_data, y: y_data})
        print("[step: {}] loss: {}".format(i, step_loss))