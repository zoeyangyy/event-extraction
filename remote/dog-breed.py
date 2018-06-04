#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Time        : 2018/5/7 下午3:44
# @Author      : Zoe
# @File        : dog-breed.py
# @Description :

import mxnet as mx
from mxnet import autograd
from mxnet import gluon
from mxnet import image
from mxnet import init
from mxnet import nd
from mxnet.gluon.data import vision
from mxnet.gluon.model_zoo import vision as models
import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt

# %matplotlib inline
# %config InlineBackend.figure_format = 'retina'

import warnings
warnings.filterwarnings("ignore")

ctx = mx.gpu()

preprocessing = [
    image.ForceResizeAug((224, 224)),
    image.ColorNormalizeAug(mean=nd.array([0.485, 0.456, 0.406]),
                            std=nd.array([0.229, 0.224, 0.225]))
]


def transform(data, label):
    data = data.astype('float32') / 255
    for pre in preprocessing:
        data = pre(data)

    data = nd.transpose(data, (2, 0, 1))
    return data, nd.array([label]).asscalar().astype('float32')


def get_features(net, data):
    features = []
    labels = []

    for X, y in tqdm(data):
        feature = net.features(X.as_in_context(ctx))
        features.append(feature.asnumpy())
        labels.append(y.asnumpy())

    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    return features, labels


preprocessing[0] = image.ForceResizeAug((224,224))
imgs = vision.ImageFolderDataset('for_train', transform=transform)
data = gluon.data.DataLoader(imgs, 64)

features_vgg, labels = get_features(models.vgg16_bn(pretrained=True, ctx=ctx), data)
features_resnet, _ = get_features(models.resnet152_v1(pretrained=True, ctx=ctx), data)
features_densenet, _ = get_features(models.densenet161(pretrained=True, ctx=ctx), data)

#
# preprocessing[0] = image.ForceResizeAug((299,299))
# imgs_299 = vision.ImageFolderDataset('for_train', transform=transform)
# data_299 = gluon.data.DataLoader(imgs_299, 64)
#
# features_inception, _ = get_features(models.inception_v3(pretrained=True, ctx=ctx), data_299)

import h5py

# with h5py.File('features.h5', 'a') as f:
#     f['vgg'] = features_vgg
#     f['resnet'] = features_resnet
#     f['densenet'] = features_densenet
#     f['inception_new'] = features_inception
#     f['labels'] = labels


preprocessing[0] = image.ForceResizeAug((224,224))

imgs = vision.ImageFolderDataset('for_train', transform=transform)
data = gluon.data.DataLoader(imgs, 64)

features_vgg, _ = get_features(models.vgg16_bn(pretrained=True, ctx=ctx), data)
features_resnet, _ = get_features(models.resnet152_v1(pretrained=True, ctx=ctx), data)
features_densenet, _ = get_features(models.densenet161(pretrained=True, ctx=ctx), data)

preprocessing[0] = image.ForceResizeAug((299,299))
imgs_299 = vision.ImageFolderDataset('for_pre', transform=transform)
data_299 = gluon.data.DataLoader(imgs_299, 64)

features_inception, _ = get_features(models.inception_v3(pretrained=True, ctx=ctx), data_299)

with h5py.File('features_pre.h5', 'a') as f:
    f['vgg'] = features_vgg
    f['resnet'] = features_resnet
    f['densenet'] = features_densenet
    f['inception'] = features_inception
