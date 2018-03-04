#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Time        : 2018/2/16 下午12:21
# @Author      : Zoe
# @File        : temp.py
# @Description :

import numpy as np
import collections
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import minimize
import tensorflow as tf
import time

# 128 * 256  and  25 * 20  =》 128 * 25 *276
# a = tf.constant(0.1, shape=[128, 256])
# b = tf.constant(0.2, shape=[25, 20])

with tf.Session() as sess:
    t = tf.constant([1,2,3,4,5,6,7,8]*2, shape=[2, 2, 4])
    g = tf.constant([4,3,2,1]*2*3, shape=[2, 4, 3])

    tf_position = tf.constant(0,shape=[2,1])
    print(sess.run(tf_position))
    for i in range(2):
        # batch_number * Chain_Lens * n_hidden_units  =>  按某i个Chain_Lens取数据
        result_beta = tf.reshape(tf.slice(t, [0, i, 0], [-1, 1, -1]), [-1, 4])
        result_beta = tf.matmul(result_beta, g[i])
        tf_position = tf.concat([tf_position, result_beta], 1)
        print('******',sess.run(tf_position))

    # tf_position = tf.constant(0, shape=[1, 3])
    # for i in range(t.shape[0]):
    #     for j in range(t.shape[1]):
    #         result_beta = tf.matmul(tf.reshape(t[i][j],[1,-1]), g[j])
    #         tf_position = tf.concat([tf_position, result_beta], 0)
    #     print('*****',sess.run(tf_position))

    tf_position = tf.reshape(tf.slice(tf_position, [0,1],[-1,-1]),[2,2,3])
    # tf_position = tf.reshape(tf_position[1:],[2,2,3])

    print(sess.run(t))
    print(sess.run(g))
    print(sess.run(tf_position))
