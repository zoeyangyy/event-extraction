#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Time        : 2017/11/15 下午2:40
# @Author      : Zoe
# @File        : 130006assignment.py
# @Description :

def trigger():
    f = open('/Users/zoe/Documents/data130006助教/小作业3/trigger_train.txt', 'r')
    train = f.readlines()
    f.close()

    f_test = open('/Users/zoe/Documents/data130006助教/小作业3/trigger_test.txt', 'r')
    test = f_test.readlines()
    f_test.close()

    # 1918 669
    f = open('/Users/zoe/Documents/data130006助教/小作业3/trigger_train_new.txt', 'w')

    sentence = list()
    tag = set()
    num = 0
    train_num = 0
    for i in train:
        if i.strip():
            sentence.append(i)
            tag.add(i.split()[1])
        else:
            if len(tag)==1:
                num += 1
            if len(tag)>1:
                train_num += 1
                for one in sentence:
                    f.write(one)
                f.write('\n')
            sentence = list()
            tag = set()

    print(train_num)
    print(num)


def argument():
    f = open('/Users/zoe/Documents/data130006助教/小作业3/argu_train_old.txt', 'r')
    train = f.readlines()
    f.close()

    f_test = open('/Users/zoe/Documents/data130006助教/小作业3/argu_test_old.txt', 'r')
    test = f_test.readlines()
    f_test.close()

    # 2131    155
    # test 剩余  997   删了 50
    f = open('/Users/zoe/Documents/data130006助教/小作业3/argu_train.txt', 'w')

    sentence = list()
    tag = set()
    num = 0
    train_num = 0
    for i in train:
        if i.strip():
            sentence.append(i)
            tag.add(i.split()[1])
        else:
            if len(tag) == 1:
                num += 1
            if len(tag) > 1:
                train_num += 1
                for one in sentence:
                    f.write(one)
                f.write('\n')
            sentence = list()
            tag = set()

    print(train_num)
    print(num)

# argument()

def blankline():
    f = open('/Users/zoe/Documents/data130006助教/小作业3/argu_train.txt', 'r')
    a = f.readlines()
    f.close()

    f = open('/Users/zoe/Documents/data130006助教/小作业3/argument_train.txt', 'w')
    li = list()
    for one in a:
        li.append(one)
        if li[-1]=='\n' and li[-2]=='\n':
            print(len(li))
            li.pop()
    for one in li:
        f.write(one)

# blankline()

def result():
    f = open('trigger_result.txt', 'r')
    result = f.readlines()
    f.close()

    f = open('argument_result.txt', 'w')
    for one in result:
        if one.strip():
            li = one.split()
            f.write(li[0]+'\t'+li[1]+'\t'+li[1]+'\n')
        else:
            f.write('\n')
    f.close()

# result()

def type():
    f = open('/Users/zoe/Documents/data130006助教/小作业3/argument_train.txt')
    content = f.readlines()
    f.close()

    type = set()
    for one in content:
        if one.strip():
            type.add(one.strip().split()[1])
    print(type)
    print(len(type))

type()