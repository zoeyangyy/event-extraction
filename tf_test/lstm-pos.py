#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gensim
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
from tensorflow.contrib import rnn
import numpy as np

'''
For Chinese word segmentation.
https://github.com/yongyehuang/Tensorflow-Tutorial/blob/master/Tutorial_6%20-%20Bi-directional%20LSTM%20for%20sequence%20labeling%20(Chinese%20segmentation).ipynb
'''
# ##################### config ######################
decay = 0.85
max_epoch = 5
max_max_epoch = 10
timestep_size = max_len = 32           # 句子长度
vocab_size = 5159    # 样本中不同字的个数，根据处理数据的时候得到
input_size = embedding_size = 100       # 字向量长度
class_num = 5
hidden_size = 128    # 隐含层节点数
layer_num = 2        # bi-lstm 层数
max_grad_norm = 5.0  # 最大梯度（超过此值的梯度将被裁剪）

lr = tf.placeholder(tf.float32)
keep_prob = tf.placeholder(tf.float32)
batch_size = 128  # 注意类型必须为 tf.int32
model_save_path = 'my_net/bi-lstm.ckpt'  # 模型保存位置


def weight_variable(shape):
    """Create a weight variable with appropriate initialization."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    """Create a bias variable with appropriate initialization."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


X_inputs = tf.placeholder(tf.int32, [None, timestep_size], name='X_input')
y_inputs = tf.placeholder(tf.int32, [None, timestep_size], name='y_input')

model = gensim.models.Word2Vec.load("../raw_file/text100.model")

def bi_lstm(X_inputs):
    """build the bi-LSTMs network. Return the y_pred"""
    # ** 0.char embedding，请自行理解 embedding 的原理！！做 NLP 的朋友必须理解这个
    embedding = tf.get_variable("embedding", [vocab_size, embedding_size], dtype=tf.float32)
    # X_inputs.shape = [batchsize, timestep_size]  ->  inputs.shape = [batchsize, timestep_size, embedding_size]
    inputs = tf.nn.embedding_lookup(embedding, X_inputs)
    # ** 1.LSTM 层
    # lstm_fw_cell = rnn.BasicLSTMCell(hidden_size, forget_bias=1.0, state_is_tuple=True)
    # lstm_bw_cell = rnn.BasicLSTMCell(hidden_size, forget_bias=1.0, state_is_tuple=True)
    # # ** 2.dropout
    # lstm_fw_cell = rnn.DropoutWrapper(cell=lstm_fw_cell, input_keep_prob=1.0, output_keep_prob=keep_prob)
    # lstm_bw_cell = rnn.DropoutWrapper(cell=lstm_bw_cell, input_keep_prob=1.0, output_keep_prob=keep_prob)
    # ** 3.多层 LSTM

    stacked_fw = []
    for i in range(layer_num):
        lstm_fw_cell = rnn.BasicLSTMCell(num_units=hidden_size, forget_bias=1.0, state_is_tuple=True)
        stacked_fw.append(rnn.DropoutWrapper(cell=lstm_fw_cell, input_keep_prob=1.0, output_keep_prob=keep_prob))
    stacked_bw = []
    for i in range(layer_num):
        lstm_bw_cell = rnn.BasicLSTMCell(num_units=hidden_size, forget_bias=1.0, state_is_tuple=True)
        stacked_bw.append(rnn.DropoutWrapper(cell=lstm_bw_cell, input_keep_prob=1.0, output_keep_prob=keep_prob))

    cell_fw = rnn.MultiRNNCell(cells=stacked_fw, state_is_tuple=True)
    cell_bw = rnn.MultiRNNCell(cells=stacked_bw, state_is_tuple=True)
    # ** 4.初始状态
    initial_state_fw = cell_fw.zero_state(batch_size, tf.float32)
    initial_state_bw = cell_bw.zero_state(batch_size, tf.float32)

    # 下面两部分是等价的
    # **************************************************************
    # ** 把 inputs 处理成 rnn.static_bidirectional_rnn 的要求形式
    # ** 文档说明
    # inputs: A length T list of inputs, each a tensor of shape
    # [batch_size, input_size], or a nested tuple of such elements.
    # *************************************************************
    # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    # inputs.shape = [batchsize, timestep_size, embedding_size]  ->  timestep_size tensor, each_tensor.shape = [batchsize, embedding_size]
    # inputs = tf.unstack(inputs, timestep_size, 1)
    # ** 5.bi-lstm 计算（tf封装）  一般采用下面 static_bidirectional_rnn 函数调用。
    #   但是为了理解计算的细节，所以把后面的这段代码进行展开自己实现了一遍。
#     try:
#         outputs, _, _ = rnn.static_bidirectional_rnn(cell_fw, cell_bw, inputs,
#                         initial_state_fw = initial_state_fw, initial_state_bw = initial_state_bw, dtype=tf.float32)
#     except Exception: # Old TensorFlow version only returns outputs not states
#         outputs = rnn.static_bidirectional_rnn(cell_fw, cell_bw, inputs,
#                         initial_state_fw = initial_state_fw, initial_state_bw = initial_state_bw, dtype=tf.float32)
#     output = tf.reshape(tf.concat(outputs, 1), [-1, hidden_size * 2])
    # ***********************************************************

    # ***********************************************************
    # ** 5. bi-lstm 计算（展开）
    with tf.variable_scope('bidirectional_rnn'):
        # *** 下面，两个网络是分别计算 output 和 state
        # Forward direction
        outputs_fw = list()
        state_fw = initial_state_fw
        with tf.variable_scope('fw'):
            for timestep in range(timestep_size):
                if timestep > 0:
                    tf.get_variable_scope().reuse_variables()
                (output_fw, state_fw) = cell_fw(inputs[:, timestep, :], state_fw)
                outputs_fw.append(output_fw)

        # backward direction
        outputs_bw = list()
        state_bw = initial_state_bw
        with tf.variable_scope('bw') as bw_scope:
            inputs = tf.reverse(inputs, [1])
            for timestep in range(timestep_size):
                if timestep > 0:
                    tf.get_variable_scope().reuse_variables()
                (output_bw, state_bw) = cell_bw(inputs[:, timestep, :], state_bw)
                outputs_bw.append(output_bw)
        # *** 然后把 output_bw 在 timestep 维度进行翻转
        # outputs_bw.shape = [timestep_size, batch_size, hidden_size]
        outputs_bw = tf.reverse(outputs_bw, [0])
        # 把两个oupputs 拼成 [timestep_size, batch_size, hidden_size*2]
        output = tf.concat([outputs_fw, outputs_bw], 2)
        # output.shape 必须和 y_input.shape=[batch_size,timestep_size] 对齐
        output = tf.transpose(output, perm=[1,0,2])
        output = tf.reshape(output, [-1, hidden_size*2])
    # ***********************************************************

    softmax_w = weight_variable([hidden_size * 2, class_num])
    softmax_b = bias_variable([class_num])
    logits = tf.matmul(output, softmax_w) + softmax_b
    return logits


y_pred = bi_lstm(X_inputs)
# adding extra statistics to monitor
# y_inputs.shape = [batch_size, timestep_size]
correct_prediction = tf.equal(tf.cast(tf.argmax(y_pred, 1), tf.int32), tf.reshape(y_inputs, [-1]))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = tf.reshape(y_inputs, [-1]), logits = y_pred))

# ***** 优化求解 *******
# 获取模型的所有参数
tvars = tf.trainable_variables()
# 获取损失函数对于每个参数的梯度
grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), max_grad_norm)
# 优化器
optimizer = tf.train.AdamOptimizer(learning_rate=lr)
# 梯度下降计算
train_op = optimizer.apply_gradients( zip(grads, tvars),
    global_step=tf.contrib.framework.get_or_create_global_step())
print('Finished creating the bi-lstm model.')