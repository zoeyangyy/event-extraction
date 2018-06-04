#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Time        : 2018/5/12 上午11:32
# @Author      : Zoe
# @File        : toutiao3.py
# @Description :

import sys
import re

key = sys.stdin.readline().strip()
content = sys.stdin.readline().strip()

pat = '.*'.join(key)
pattern = re.compile(pat)
result = pattern.findall(content)


max = 0
if len(result) == 0:
    print(max)
else:
    max = 100-(len(result[0])-len(key))
    for i in range(len(result)):
        s = 100-(len(result[i])-len(key))
        if s > max:
            max = s
    print(max)
