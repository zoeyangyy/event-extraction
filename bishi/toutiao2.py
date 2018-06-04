#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Time        : 2018/5/12 上午10:52
# @Author      : Zoe
# @File        : toutiao2.py
# @Description :
import sys

s = sys.stdin.readline().strip()
result = []
while s:
    M, N = [int(i) for i in s.split()]
    pre = []
    for i in range(M):
        pre.append(sys.stdin.readline().strip())
    pre = tuple(pre)
    k = sys.stdin.readline().strip()
    for i in range(N):
        n = sys.stdin.readline().strip()
        if n.startswith(pre):
            result.append('1')
        else:
            result.append('-1')
    result.append('')
    k = sys.stdin.readline().strip()
    s = sys.stdin.readline().strip()

for i in range(len(result)):
    print(result[i])