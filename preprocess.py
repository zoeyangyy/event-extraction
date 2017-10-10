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
import tensorflow as tf


def trigger_identify(wj,add_type):
    dir = '/Users/zoe/Documents/event_extraction/ace_2005_chinese/'+wj+'/adj'
    pathDir = os.listdir(dir)

    f_write = open('/Users/zoe/Documents/event_extraction/CRF++-0.58/example/sequence/all.txt', add_type)
    f_write_test = open('/Users/zoe/Documents/event_extraction/CRF++-0.58/example/sequence/all_test.txt', add_type)

    file_num = len([file for file in pathDir])/4
    count = 0
    sep_tri = 0

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
                    anchor = event_mention.anchor.charseq.text
                    if next(jieba.cut(anchor)) != anchor:
                        sep_tri += 1
                    if count <= file_num * 0.7:
                        for w in jieba.posseg.cut(text):
                            if w.word != anchor:
                                f_write.write(w.word+'\t'+w.flag+'\tO\n')
                            else:
                                f_write.write(w.word+'\t'+w.flag+'\tB_'+type+'\n')
                        f_write.write('\n')
                    else:
                        for w in jieba.posseg.cut(text):
                            if w.word != anchor:
                                f_write_test.write(w.word+'\t'+w.flag+'\tO\n')
                            else:
                                f_write_test.write(w.word+'\t'+w.flag+'\tB_'+type+'\n')
                        f_write_test.write('\n')
            print(count)
            f.close()
    print('anchor被分词:', sep_tri)
    f_write.close()
    f_write_test.close()


# trigger_identify('wl', 'a')

# 298+238+97=633 file
# 1398+1382+553 = 3333 mention
# 中文分词的特例：
# 1.涉及此案的管理人士如果被提起诉讼且被裁决有罪，则将面临罚款或入狱。"裁决有罪"为一个动词，但被分词
# 553个mention: sep_tri = 20
# 1382 mention: sep_tri = 58
# 1398 mention: sep_tri = 103
# 2.有些新闻extent只有一个词就是trigger

def evaluation():
    f_read = open('/Users/zoe/Documents/event_extraction/CRF++-0.58/example/sequence/all_argu_result', 'r')
    contents = f_read.readlines()
    f_read.close()

    m1, m2, m3, m4 = 0, 0, 0, 0
    type_correct = 0
    for word in contents:
        if word.strip():
            li = word.strip().split('\t')
            if li[2] != 'O':
                if li[3] != 'O':
                    m1 += 1
                    if li[3] == li[2]:
                        type_correct += 1
                if li[3] == 'O':
                    m3 += 1
            if li[2] == 'O':
                if li[3] != 'O':
                    m2 += 1
                if li[3] == 'O':
                    m4 += 1
    print(m1, m2, m3, m4, type_correct)
    precision = m1/(m1+m2)
    recall = m1/(m1+m3)
    F1 = 2*precision*recall/(precision+recall)
    print("type_correct: ", round(type_correct/m1, 4))
    print("precision: ",round(precision, 4))
    print("recall: ",round(recall,4))
    print("F1: ",round(F1,4))

# evaluation()

# trigger:
# 540 118 396 13566 499
# type_correct:  0.9241
# precision:  0.8207
# recall:  0.5769
# F1:  0.6775

# argument
# 3489 708 2465 5759 1446
# type_correct:  0.4144
# precision:  0.8313
# recall:  0.586
# F1:  0.6874


def argument_identify(wj,add_type):
    dir = '/Users/zoe/Documents/event_extraction/ace_2005_chinese/'+wj+'/adj'
    pathDir = os.listdir(dir)

    f_write = open('/Users/zoe/Documents/event_extraction/CRF++-0.58/example/sequence/all_argu.txt', add_type)
    f_write_test = open('/Users/zoe/Documents/event_extraction/CRF++-0.58/example/sequence/all_argu_test.txt', add_type)

    file_num = len([file for file in pathDir])/4
    count = 0

    for file in pathDir:
        if re.search(r'apf.xml', file):
            count += 1
            f = open(dir+'/'+file, 'r')
            file_content = f.read()
            soup = bs4.BeautifulSoup(file_content, "html5lib")
            events = soup.find_all('event')
            for event in events:
                event_mentions = event.find_all('event_mention')
                for event_mention in event_mentions:
                    text = ''.join(event_mention.extent.charseq.text.split())
                    arguments = event_mention.find_all('event_mention_argument')
                    argu = dict()
                    for argument in arguments:
                        name = ''.join(argument.charseq.text.split())
                        argu[name] = argument['role']
                    text_li = list()
                    index_list = list()
                    for key in argu.keys():
                        index = text.find(key)
                        index_end = index+len(key)
                        index_list.append(index)
                        index_list.append(index_end)
                    j = 0
                    for i in sorted(index_list):
                        text_li.append(text[j:i])
                        j = i
                    if count <= file_num * 0.7:
                        for text in text_li:
                            if text in list(argu.keys()):
                                type = argu[text]
                                for w in jieba.posseg.cut(text):
                                    f_write.write(w.word+'\t'+w.flag+'\tB_'+type+'\n')
                            else:
                                for w in jieba.posseg.cut(text):
                                    f_write.write(w.word+'\t'+w.flag+'\tO\n')
                        f_write.write('\n')
                    else:
                        for text in text_li:
                            if text in list(argu.keys()):
                                type = argu[text]
                                for w in jieba.posseg.cut(text):
                                    f_write_test.write(w.word+'\t'+w.flag+'\tB_'+type+'\n')
                            else:
                                for w in jieba.posseg.cut(text):
                                    f_write_test.write(w.word+'\t'+w.flag+'\tO\n')
                        f_write_test.write('\n')
            print(count)
            f.close()
    f_write.close()
    f_write_test.close()


# argument_identify('wl', 'a')

# li = list()
# a = dict()
# a['BC'] = 1
# li.append(a)
# if 'B' in list(li[0].keys())[0]:
#     print('T')

embedding = tf.get_variable("embedding", [32, 100], dtype=tf.float32)
print(embedding)
inputs = tf.nn.embedding_lookup(embedding, X_inputs)