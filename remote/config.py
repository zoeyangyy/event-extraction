#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Time        : 2018/3/6 上午8:18
# @Author      : Zoe
# @File        : config.py
# @Description :
import tensorflow as tf

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_float('lr', 0.000005, 'Initial learning rate.')
flags.DEFINE_integer('epoch', 6, 'Number of epochs to train.')
flags.DEFINE_integer('n_hidden_units', 64, 'Number of units in hidden layer.')
flags.DEFINE_integer('_batch_size', 128, '')
flags.DEFINE_integer('vocab_size', 25, '样本中事件类型个数，根据处理数据的时候得到')
flags.DEFINE_integer('embedding_size', 32, '事件embedding的大小')
flags.DEFINE_integer('Chain_Lens', 5, '链条长度')
flags.DEFINE_integer('n_steps', 5, '链条长度')
flags.DEFINE_integer('n_classes', 25, '样本中事件类型个数')

