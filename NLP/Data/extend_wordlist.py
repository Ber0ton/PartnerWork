# -*- coding: utf-8 -*-
import jieba
import re

MY_CONSTANT = 53143

def words_divide(file_path, savepath):
    f = open(savepath, "a+", encoding='utf-8')
    word_set = set()

    # 读取单个文件
    with open(file_path, 'r', encoding='utf-8-sig') as file:
        content = file.read()
        # 使用正则表达式剔除特殊符号、字母和数字
        content = re.sub(r'[a-zA-Z]+', '', content)  # 剔除一串字母串
        content = re.sub(r'\d+', '', content)  # 剔除一串数字
        content = re.sub(r'[^\w\u4e00-\u9fa5]+', '', content)
        line_jieba = jieba.cut(content, cut_all=False, HMM=True)
        word_list = list(line_jieba)
        word_set.update(word_list)

    count = MY_CONSTANT
    for word in word_set:
        count += 1
        f.write(str(count) + " " + word + "\n")

    f.close()

# 设置文件路径和保存路径
file_path = 'RawText__xinhua.txt'
savepath = 'new_wordlist.Dic'
words_divide(file_path, savepath)  # 53143
