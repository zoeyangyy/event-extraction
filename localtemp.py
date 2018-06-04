import math
from functools import reduce

from matplotlib import pyplot
import matplotlib.pyplot

import time, threading
import collections
from itertools import combinations
import os
import pandas as pd
import numpy as np


def test1():
    li = [1,1,1,0]

    print(reduce(lambda x,y:x&y, map(lambda x:x==1, li)))


    state, li = ((2,3),[1,0,1,0])
    corners = ((1, 1), (1, 5), (5, 1), (5, 5))
    cost = 0
    for index,corner in enumerate(corners):
        if li[index] == 0:
            cost += abs(state[0]- corner[0]) + abs(state[1]- corner[1])

    print(cost)

    li = [1,2,3,4,5,6]
    print([one for one in li if one%2==0])

    print(list(map(lambda x:x if x%2==0 else None, li)))

    print(list(filter(lambda x:True if x % 2 == 0 else False, li)))


class Student(object):
    pass
s = Student()

def set_age(self, age): # 定义一个函数作为实例方法
    self.age = age

Student.set_age = set_age

# >>> from types import MethodType
# s.set_age = set_age # 给实例绑定一个方法
# s.set_age(s, 25) # 调用实例方法
# print(s.age)


# 新线程执行的代码:
def loop():
    print('thread %s is running...' % threading.current_thread().name)
    n = 0
    while n < 5:
        n = n + 1
        print('thread %s >>> %s' % (threading.current_thread().name, n))
        time.sleep(1)
    print('thread %s ended.' % threading.current_thread().name)

def loop_s():
    print('thread %s is running...' % threading.current_thread().name)
    t = threading.Thread(target=loop, name='LoopThread')
    t.start()
    # t.join()
    print('thread %s ended.' % threading.current_thread().name)



data = [('1','aaa','1927'),('2','bbb','1937'),('2','ccc','1937'),('2','d','1927')]


def func(data):
    edge_dict = collections.defaultdict(list)
    for d in data:
        edge_dict[d[2]].append(d[1])

    edge_list = []
    for key,li in edge_dict.items():
        edge_list += list(combinations(li, 2))
    return edge_list

