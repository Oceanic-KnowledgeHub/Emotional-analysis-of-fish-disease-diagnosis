import gensim
import jieba
import re
from gensim.models import Word2Vec
# import warnings
#
# warnings.filterwarnings('ignore')

with open('Fish.txt', 'r',encoding='utf-8')as f: # 读入文本
    lines = []
    for line in f: #分别对每段分词
        temp = jieba.lcut(line)  #结巴分词 精确模式
        words = []
        for i in temp:
            #过滤掉所有的标点符号
            i = re.sub("[\s+\.\!\/_,$%^*(+\"\'””《》]+|[+——！，。？、~@#￥%……&*（）：；‘]+", "", i)
            if len(i) > 0:
                words.append(i)
        if len(words) > 0:
            lines.append(words)
# print(lines[0:5])  # 预览前5行分词结果

# lines=[['环境'],['时节'],['体表'],['体格'],['体内'],['体态'],['鳃']]
word2vec=Word2Vec(lines,vector_size=50,window=2,min_count=2,epochs=10,negative=10,sg=1)
#
# print('鳃：',word2vec.wv.get_vector('鳃'))
# print('体态：',word2vec.wv.get_vector('体态'))
# print('体内：',word2vec.wv.get_vector('体内'))
# print('体格：',word2vec.wv.get_vector('体格'))
# print('体表：',word2vec.wv.get_vector('体表'))
# print('时节：',word2vec.wv.get_vector('时节'))
# print('环境：',word2vec.wv.get_vector('环境'))