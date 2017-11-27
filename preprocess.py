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

    # f_write = open('/Users/zoe/Documents/event_extraction/CRF++-0.58/example/sequence/all.txt', add_type)
    # f_write_test = open('/Users/zoe/Documents/event_extraction/CRF++-0.58/example/sequence/all_test.txt', add_type)

    f_write = open('/Users/zoe/Documents/data130006助教/小作业3/trigger_train.txt', add_type)
    f_write_test = open('/Users/zoe/Documents/data130006助教/小作业3/trigger_test.txt', add_type)

    file_num = len([file for file in pathDir])/4
    count = 0
    sep_tri = 0
    one_word = 0
    train = 0
    test = 0

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
                        continue
                    if text == anchor:
                        one_word += 1
                        continue
                    if count <= file_num * 0.8:
                        for w in jieba.posseg.cut(text):
                            if w.word != anchor:
                                # f_write.write(w.word+'\t'+w.flag+'\tO\n')
                                f_write.write(w.word + '\tO\n')
                            else:
                                f_write.write(w.word+'\tT_'+type+'\n')
                        f_write.write('\n')
                        train += 1
                    else:
                        for w in jieba.posseg.cut(text):
                            if w.word != anchor:
                                f_write_test.write(w.word+'\tO\n')
                            else:
                                f_write_test.write(w.word+'\tT_'+type+'\n')
                        f_write_test.write('\n')
                        test += 1
            print(count)
            f.close()
    print('anchor被分词:', sep_tri)
    print('only one word:', one_word)
    print('train:', train)
    print('test:', test)
    f_write.close()
    f_write_test.close()


# trigger_identify('bn', 'w')
# trigger_identify('nw', 'a')
# trigger_identify('wl', 'a')

# 298+238+97=633 file
# 1398+1382+553 = 3333 mention
# 中文分词的特例：
# 1.涉及此案的管理人士如果被提起诉讼且被裁决有罪，则将面临罚款或入狱。"裁决有罪"为一个动词，但被分词
# 553个mention: sep_tri = 20
# 1382 mention: sep_tri = 58
# 1398 mention: sep_tri = 103
# 2.有些新闻extent只有一个词就是trigger

# anchor被分词: 103
# only one word: 46
# train: 849
# test: 400

# anchor被分词: 58
# only one word: 58
# train: 810
# test: 456

# anchor被分词: 20
# only one word: 34
# train: 306
# test: 193

def evaluation(i):
    # f_read = open('/Users/zoe/Documents/event_extraction/CRF++-0.58/example/sequence/all_argu_result', 'r')
    f_read = open('raw_file/argu_result.txt', 'r')
    contents = f_read.readlines()
    f_read.close()

    m1, m2, m3, m4 = 0, 0, 0, 0
    type_correct = 0
    tp = set()
    for word in contents:
        if word.strip():
            li = word.strip().split()
            if li[2-i] != 'O':
                tp.add(li[2-i])
                if li[3-i] != 'O':
                    m1 += 1
                    if li[3-i] == li[2-i]:
                        type_correct += 1
                if li[3-i] == 'O':
                    m3 += 1
            if li[2-i] == 'O':
                if li[3-i] != 'O':
                    m2 += 1
                if li[3-i] == 'O':
                    m4 += 1
    print(m1, m2, m3, m4, type_correct)
    precision = m1/(m1+m2)
    recall = m1/(m1+m3)
    F1 = 2*precision*recall/(precision+recall)
    print("type_correct: ", round(type_correct/m1, 4))
    print("precision: ",round(precision, 4))
    print("recall: ",round(recall,4))
    print("F1: ",round(F1,4))
    print(tp)
    print(len(tp))

evaluation(2)

# trigger:
# 540 118 396 13566 499
# type_correct:  0.9241

# precision:  0.8207
# recall:  0.5769
# F1:  0.6775

# argument
# 2642 825 2482 6871 1222
# type_correct:  0.4625

# precision:  0.762
# recall:  0.5156
# F1:  0.6151

# pytorch:
# trigger:
# 274 412 112 9382 264
# type_correct:  0.9635
# precision:  0.3994
# recall:  0.7098
# F1:  0.5112

# argument:
# type_correct:  0.3077
# precision:  0.3806
# recall:  0.6249
# F1:  0.4731

def argument_identify(wj,add_type):
    dir = '/Users/zoe/Documents/event_extraction/ace_2005_chinese/'+wj+'/adj'
    pathDir = os.listdir(dir)

    f_write = open('/Users/zoe/Documents/event_extraction/CRF++-0.58/example/sequence/all_argu.txt', add_type)
    f_write_test = open('/Users/zoe/Documents/event_extraction/CRF++-0.58/example/sequence/all_argu_test.txt', add_type)

    # f_write = open('/Users/zoe/Documents/data130006助教/小作业3/argu_train_old.txt', add_type)
    # f_write_test = open('/Users/zoe/Documents/data130006助教/小作业3/argu_test_old.txt', add_type)

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
                    index_list.append(len(text))
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
                                    # f_write.write(w.word+'\t'+w.flag+'\tA_'+type+'\n')
                                    f_write.write(w.word+'\t'+w.flag +'\tA_' + type + '\n')
                            else:
                                for w in jieba.posseg.cut(text):
                                    f_write.write(w.word+'\t'+w.flag+'\tO\n')
                        f_write.write('\n')
                    else:
                        for text in text_li:
                            if text in list(argu.keys()):
                                type = argu[text]
                                for w in jieba.posseg.cut(text):
                                    f_write_test.write(w.word+'\t'+w.flag+'\tA_'+type+'\n')
                            else:
                                for w in jieba.posseg.cut(text):
                                    f_write_test.write(w.word+'\t'+w.flag+'\tO\n')
                        f_write_test.write('\n')
            print(count)
            f.close()
    f_write.close()
    f_write_test.close()

# argument_identify('bn', 'w')
# argument_identify('nw', 'a')
# argument_identify('wl', 'a')

# li = list()
# a = dict()
# a['BC'] = 1
# li.append(a)
# if 'B' in list(li[0].keys())[0]:
#     print('T')

# embedding = tf.get_variable("embedding", [32, 100], dtype=tf.float32)
# print(embedding)
# inputs = tf.nn.embedding_lookup(embedding, X_inputs)