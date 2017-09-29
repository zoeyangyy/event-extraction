#encoding=utf-8
import sys
import jieba
dic_path='sougou.txt'
#dic_path=dic_path.encode('utf8')
#print type(dic_path)
jieba.load_userdict(dic_path)
print(", ".join(jieba.cut("胆碱酯酶减少胆碱脂酶试剂盒胆碱酯酶试纸胆碱酯酶增加")))

