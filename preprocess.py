#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Time        : 2017/9/23 下午3:40
# @Author      : Zoe
# @File        : preprocess.py
# @Description : ace2005 chinese corpus
#                trigger % argument | identification & classification
import jieba
import jieba.posseg
import os
import re
import bs4


def trigger_identify(wj):
    dir = '/Users/zoe/Documents/event_extraction/ace_2005_chinese/'+wj+'/adj'
    pathDir = os.listdir(dir)

    f_write = open('/Users/zoe/Documents/event_extraction/CRF++-0.58/example/sequence/'+wj+'.txt','w')
    f_write_test = open('/Users/zoe/Documents/event_extraction/CRF++-0.58/example/sequence/'+wj+'_test.txt','w')
    count = 0
    for file in pathDir:
        if re.search(r'apf.xml', file):
            count += 1
            f = open(dir+'/'+file, 'r')
            file_content = f.read()
            soup = bs4.BeautifulSoup(file_content, "html5lib")
            events = soup.find_all('event')
            for event in events:
                type = event['type']
                subtype = event['subtype']
                event_mentions = event.find_all('event_mention')
                for event_mention in event_mentions:
                    text = ''.join(event_mention.extent.charseq.text.split())
                    if len(event_mention.find_all('anchor')) > 1:
                        print('有两个及以上trigger')
                    anchor = event_mention.anchor.charseq.text
                    if next(jieba.cut(anchor)) != anchor:
                        print('anchor被分词')
                    if count <= 66:
                        for w in jieba.posseg.cut(text):
                            if w.word != anchor:
                                f_write.write(w.word+'\t'+w.flag+'\tO\n')
                            else:
                                f_write.write(w.word+'\t'+w.flag+'\tB_'+subtype+'\n')
                        f_write.write('\n')
                    else:
                        for w in jieba.posseg.cut(text):
                            if w.word != anchor:
                                f_write_test.write(w.word+'\t'+w.flag+'\tO\n')
                            else:
                                f_write_test.write(w.word+'\t'+w.flag+'\tB_'+subtype+'\n')
                        f_write_test.write('\n')
            print(count)
            f.close()
    f_write.close()
    f_write_test.close()


trigger_identify('wl')
# 298+238+97=633 file
# 1398+1382+553 = 3333 mention
# 中文分词的特例：
# 1.涉及此案的管理人士如果被提起诉讼且被裁决有罪，则将面临罚款或入狱。"裁决有罪"为一个动词，但被分词
# 但比较少，553个mention中有8个特例。
# 2.有些新闻extent只有一个词就是trigger
