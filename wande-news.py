#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Time        : 2018/5/2 下午4:27
# @Author      : Zoe
# @File        : wande-news.py
# @Description :


f = open('/Users/zoe/Documents/event_extraction/wande-data/news_2016_json.txt', 'r', encoding='gb18030')
a = f.readlines()
f.close()

import json
import codecs

f = codecs.open('/Users/zoe/Documents/event_extraction/wande-data/2016.txt','w', 'utf8')
for one in a:
    new_one = dict()
    old = json.loads(one)
    new_one['time'] = old['PUBLISHDATE'][:-3]
    new_one['source'] = old['SOURCE']
    new_one['id'] = int(old['OBJECT_ID'])
    new_one['content'] = old['CONTENT'][:100]
    new_one['url'] = old['URL']
    new_one['title'] = old['TITLE']
    f.write(json.dumps({"index":{}})+'\n')
    f.write(json.dumps(new_one, ensure_ascii=False)+'\n')
    break
f.close()




