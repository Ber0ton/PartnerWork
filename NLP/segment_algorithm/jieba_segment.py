# -*- coding: utf-8 -*-
import jieba
import re


def detailrearrange(line_jieba):
    after_rerange = []
    digit = re.compile('^[0-9]+(.[0-9]{1,3})?$')
    zhPattern = re.compile(u'[\u4e00-\u9fa5]+')
    for i in line_jieba:
        if digit.match(i):
            temp = next(line_jieba)
            if zhPattern.search(temp) and temp != '或':
                after_rerange.append(str(i) + temp)
            else:
                after_rerange.append(i)
                after_rerange.append(temp)
        else:
            after_rerange.append(i)
    lineafter = " ".join(after_rerange)
    return lineafter


def words_divide(readpath, savepath):
    f = open(readpath, encoding='utf-8')
    lines = f.readlines()
    f.close()
    jieba.load_userdict("new_wordlist.Dic")
    f = open(savepath, "a+",encoding='UTF-8')
    for line in lines:
        line_jieba = jieba.cut(line, cut_all=False, HMM=True)
        rerangeline = detailrearrange(line_jieba)
        f.write(rerangeline)
    f.close()


readpath = 'RawText__xinhua.txt'
savepath = 'jieba_Segment.txt'
words_divide(readpath, savepath)

