#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Time        : 2017/10/4 下午2:40
# @Author      : Zoe
# @File        : cnn-test.py
# @Description : event-extraction lstm function
# 修改：不均衡问题／ 占位符不算cost／mask／dropout／output后加激活函数／embedding_lookup／

import tensorflow as tf
import jieba
import os
import re
import bs4
import random
import gensim
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data


def txt_features():
    f = open('/Users/zoe/Documents/event_extraction/CRF++-0.58/example/sequence/all.txt', 'r')
    content = f.readlines()
    f.close()
    # f = open('raw_file/test.txt', 'a')
    li = list()
    s = 0
    for one in content:
        if one.strip():
            s += 1
            li.append(s)
        else:
            # f.write(s+'\n')
            s = 0
    # f.close()
    d = [li.count(k) for k in set(li)]
    all_sum = np.sum(d)
    sum = 0
    for one in range(1,33):
        sum += d[one]
    print(sum)
    print(all_sum)
    print(sum/all_sum)

    plt.bar(range(len(d)), d, color='y')
    plt.xlim(0, 150)
    plt.xlabel('sentence_length')
    plt.ylabel('sentence_num')
    plt.title('Distribution of the Length of Sentence')
    plt.show()

# txt_features()

def create_text():
    f_all = open('raw_file/test_all.txt', 'a')
    i = 0
    # 三个文件夹的文本结构不同，要分开执行
    dir = '/Users/zoe/Documents/event_extraction/ace_2005_chinese/' + 'wl' + '/adj'
    pathDir = os.listdir(dir)
    for file in pathDir:
        if re.search(r'sgm', file):
            i+=1
            f = open(dir+'/'+file, 'r')
            file_content = f.read()
            f.close()
            soup = bs4.BeautifulSoup(file_content, "html5lib")
            events = soup.find_all('post')
            text = ''.join(events[0].text.split())
            f_all.write(text+'\n')
    print(i)
    f_all.close()


def jieba_cut():
    f = open('raw_file/contents.txt','r')
    a = f.readlines()
    f.close()
    f = open('raw_file/test_jieba.txt', 'w')
    for one in a:
        f.write(' '.join(jieba.cut(one)))
    f.close()

# jieba_cut()

def create_model():
    # inp为输入语料
    inp = 'raw_file/test_jieba.txt'
    # outp1 为输出模型
    outp1 = 'raw_file/text100.model'
    model = Word2Vec(LineSentence(inp), size=100, window=5, min_count=20)
    model.save(outp1)

# create_model()

def model_usage():
    # 导入模型
    model = gensim.models.Word2Vec.load("raw_file/text100.model")
    print(model['人民'])
    result = model.most_similar('人民')  # 求余弦

    word =[]
    for each in result:
        print(each[0], each[1])
        word.append(each[0])

    print(word)

# model_usage()

# model = gensim.models.Word2Vec.load("raw_file/text100.model")

def get_xy():
    with open('/Users/zoe/Documents/event_extraction/CRF++-0.58/example/sequence/all.txt', 'r') as f:
        content = f.readlines()
    s = 0
    x_mat_list = list()
    x_mat = np.zeros(shape=(32, 100))
    y_tag_list = list()
    y_tag = np.zeros(shape=(32, 10))
    # tag = 0 表示没有词，占位
    y_tag_index = ['no_word', 'B_Movement', 'B_Justice', 'B_Transaction', 'B_Contact', 'B_Personnel', 'B_Business', 'B_Life',
             'B_Conflict', 'O']
    for one in content:
        if one.strip() and s < 32:
            try:
                word_embedding = model[one.strip().split()[0]]
            except:
                word_embedding = np.zeros(100)
            x_mat[s] = word_embedding
            y_tag[s][y_tag_index.index(one.strip().split()[2])] = 1
            s += 1
        else:
            s = 0
            x_mat_list.append(x_mat)
            y_tag_list.append(y_tag)
            x_mat = np.zeros(shape=(32, 100))
            y_tag = np.zeros(shape=(32, 10))

    # x_mat.shape = [-1, 32, 100]  =>  (2296, 32, 100)
    return np.array(x_mat_list), np.array(y_tag_list)


def get_xy_index():
    with open('/Users/zoe/Documents/event_extraction/CRF++-0.58/example/sequence/all.txt', 'r') as f:
        content = f.readlines()
    word_dict = {}
    for one in content:
        if one.strip():
            one = one.strip().split()[0]
            if one not in word_dict:
                word_dict[one] = len(word_dict)
    s = 0
    x_mat_list = list()
    x_mat = np.zeros(shape=(32, 1))
    y_tag_list = list()
    y_tag = np.zeros(shape=(32, 10))
    y_tag_index = ['no_word', 'B_Movement', 'B_Justice', 'B_Transaction', 'B_Contact', 'B_Personnel', 'B_Business',
                   'B_Life', 'B_Conflict', 'O']
    for one in content:
        if one.strip() and s < 32:
            x_mat[s] = word_dict[one.strip().split()[0]]
            y_tag[s][y_tag_index.index(one.strip().split()[2])] = 1
            s += 1
        else:
            s = 0
            x_mat_list.append(x_mat)
            y_tag_list.append(y_tag)
            x_mat = np.zeros(shape=(32, 1))
            y_tag = np.zeros(shape=(32, 10))

    return np.array(x_mat_list), np.array(y_tag_list), len(word_dict)

x, y, embeddingLens = get_xy_index()

# print(x_mat_list.shape, y_tag_list.shape,y_tag_list[0][0])
# mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

lr = 0.001
training_iters = 100000
batch_size = 64

n_inputs = 100 # 词word embedding
n_steps = 32 # 句子长度
n_hidden_units = 128 # 神经元数目
# 共九类：{'no word','B_Movement', 'B_Justice', 'B_Transaction', 'B_Contact', 'B_Personnel', 'B_Business', 'B_Life', 'B_Conflict', 'O'}
n_classes = 10

x = tf.placeholder(tf.float32,[None, n_steps, n_inputs])
y = tf.placeholder(tf.float32,[None, n_classes])
embedding = tf.Variable(np.identity(embeddingLens, dtype=np.int32))

weights = {
    # （100，128）
    'in': tf.Variable(tf.random_normal([n_inputs,n_hidden_units])),
    # （128，10）
    'out': tf.Variable(tf.random_normal([n_hidden_units,n_classes]))
}
biases = {
    'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units])),
    'out': tf.Variable(tf.constant(0.1, shape=[n_classes]))
}

# f = open('raw_file/tensor_result.txt', 'w')

def RNN(X, weights, biases):


    # hidden layer for input to cell
    # x = (64 batch, 32 step, 100 input) => (64*32, 100 input)
    X = tf.reshape(X, [-1, n_inputs])
    # => (64*32, 128 hidden) => (64 batch, 32steps, 128 hidden)
    X_in = tf.matmul(X, weights['in'])+biases['in']
    X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])

    # cell
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)
    _init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)
    outputs, states = tf.nn.dynamic_rnn(lstm_cell, X_in, initial_state=_init_state, time_major=False)
    # hidden layer for output as the final results

    outputs = tf.reshape(outputs, [-1, n_hidden_units])

    # results = [64*32, 10 class]  => [64 batch, 32 steps, 10 class]
    results = tf.matmul(outputs, weights['out']) + biases['out']
    # results = tf.reshape(results, [-1, n_steps, n_classes])
    return results


pred = RNN(x, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
train_op = tf.train.AdamOptimizer(lr).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()
x_mat_list,y_tag_list = get_xy()
# shuffle x y
zip_list = list(zip(x_mat_list, y_tag_list))
random.shuffle(zip_list)
x_mat_list[:], y_tag_list[:] = zip(*zip_list)

with tf.Session() as sess:
    sess.run(init)
    step = 0
    while step * batch_size < training_iters:
        # batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        start = random.randint(0, x_mat_list.shape[0]-batch_size)
        batch_xs, batch_ys= x_mat_list[start:start+batch_size], y_tag_list[start:start+batch_size]
        batch_ys = np.reshape(batch_ys, [-1, n_classes])
        # batch_xs = tf.nn.embedding_lookup(embedding, batch_xs)

        # batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs])
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


