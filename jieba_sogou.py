#encoding=utf-8
import sys
import jieba
# dic_path='sougou.txt'
# #dic_path=dic_path.encode('utf8')
# #print type(dic_path)
# jieba.load_userdict(dic_path)
# print(", ".join(jieba.cut("胆碱酯酶减少胆碱脂酶试剂盒胆碱酯酶试纸胆碱酯酶增加")))


#encoding=utf-8
import jieba

seg_list = jieba.cut("我来到北京清华大学", cut_all=True)
print("Full Mode:", "/ ".join(seg_list))  # 全模式

seg_list = jieba.cut("我来到北京清华大学", cut_all=False)
print("Default Mode:", "/ ".join(seg_list))  # 精确模式

seg_list = jieba.cut("他来到了网易杭研大厦")  # 默认是精确模式
print(", ".join(seg_list))

seg_list = jieba.cut_for_search("小明硕士毕业后在日本京都大学深造")  # 搜索引擎模式
print(", ".join(seg_list))

import jieba.posseg as pseg
words = pseg.cut("我爱北京天安门")
for w in words:
    print(w.word, w.flag)