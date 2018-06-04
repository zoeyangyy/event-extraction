#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Time        : 2018/5/12 上午10:04
# @Author      : Zoe
# @File        : toutiao1.py
# @Description :
import re

line = input().strip()
article = ''
try:
    while line:
        article += line
        line = input().strip()
except EOFError:
    pass
print(article)
pattern = re.compile(r'//.+')
result1 = pattern.findall(article)
print(result1)

pattern = re.compile(r'/\*.+/')
result2 = pattern.findall(article)
print(result2)

pattern = re.compile(r'\".*/\*.+\*/.*\"')
result3 = pattern.findall(article)

pattern = re.compile(r'\".*//.*\"')
result4 = pattern.findall(article)

print(len(result1)+len(result2)-len(result3)-len(result4))

# //f dfs//
# /*fsd*/ /*f fil
# dfs*/
#
# dfsf
# dsf = "/*gf */"
# fdg