#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Time        : 2018/5/11 下午8:00
# @Author      : Zoe
# @File        : alibaba2.py
# @Description :

# 4      (总共4个结点，编号0,1,2,3)
# 4 2
# 0 1 （从结点0到结点1的一条有向边）
# 1 2 （从结点1到结点2的一条有向边）
# 2 3 （从结点2到结点3的一条有向边）
# 0 2 （从结点0到结点2的一条有向边）

N = int(input().strip())
M = int(input().strip().split()[0])

li = []
for i in range(N):
    li.append([0 for _ in range(N)])

for i in range(M):
    x,y = input().strip().split()
    li[int(x)][int(y)] = 1

Num = [0 for _ in range(N)]
for i in range(N):
    for j in range(N):
        if li[i][j] == 1:
            Num[i] += 1


def digui(i):
    for j in range(N):
        if li[i][j] == 1:
            Num[i] += digui(j)
    return Num[i]


for i in range(N):
    Num = [0 for _ in range(N)]
    for i in range(N):
        for j in range(N):
            if li[i][j] == 1:
                Num[i] += 1
    digui(i)

print(Num)
