#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Time        : 2017/11/20 上午10:52
# @Author      : Zoe
# @File        : temp.py
# @Description :

import  matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


def draw():
    x = [i for i in range(0, 70)]
    mp = []
    ap = []
    for i in x:
        mp.append(120000*i-3000*(i * i))
        ap.append(60000*i-1000*i*i)
    plt.plot(x, mp, label='mp')
    plt.plot(x, ap, label='ap')
    max_ap = max(ap)
    max_index = ap.index(max_ap)
    plt.plot(max_index, max_ap, 'm*')
    plt.legend()
    plt.grid(True)
    plt.show()

draw()

def draw2():
    fig = plt.figure()
    ax = Axes3D(fig)

    X = np.arange(0, 400, 1)
    Y = np.arange(0, 400, 1)
    X, Y = np.meshgrid(X, Y)
    Z = 1200* (X ** 2)*Y - 3* (X**3)* (Y**2)

    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')
    plt.show()

# draw2()