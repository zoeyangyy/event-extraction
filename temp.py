#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Time        : 2017/11/20 上午10:52
# @Author      : Zoe
# @File        : temp.py
# @Description :
import json

f = open('/Users/zoe/Documents/event_extraction/majorEventDump/majorEventDump.json','r')
a = f.read()
f.close()

f = open('/Users/zoe/Documents/event_extraction/majorEventDump/typeCodeDump.json','r')
b = f.read()
f.close()

j = json.loads(a)
j2 = json.loads(b)

# {'S_INFO_WINDCODE': '000418.SZ', 'S_EVENT_HAPDATE': '20140815', 'S_EVENT_EXPDATE': '20140815', 'S_EVENT_CATEGORYCODE': '204008001'}

dic = list()
for one in j:
    if one['S_INFO_WINDCODE'] == '000418.SZ':
        d = dict()
        d['S_INFO_WINDCODE'] = one['S_INFO_WINDCODE']
        d['S_EVENT_HAPDATE'] = one['S_EVENT_HAPDATE']
        d['S_EVENT_EXPDATE'] = one['S_EVENT_EXPDATE']
        d['S_EVENT'] = j2[one['S_EVENT_CATEGORYCODE']]
        dic.append(d)

f = open('event.txt', 'w')
for one in dic:
    f.write(json.dumps(one,ensure_ascii=False)+'\n')
f.close()
