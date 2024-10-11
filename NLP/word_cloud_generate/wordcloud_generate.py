# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
import os
import re

def load_stopwords(stopwords_file):
    """
    从文件中加载停用词
    :param stopwords_file: 停用词文件路径，每行一个停用词
    :return: 停用词集合
    """
    stopwords = set()
    if not os.path.exists(stopwords_file):
        print(f"停用词文件 {stopwords_file} 不存在，请检查路径。")
        return stopwords

    with open(stopwords_file, 'r', encoding='utf-8') as f:
        for line in f:
            word = line.strip()
            if word:
                stopwords.add(word)
    return stopwords

def remove_punctuation(word):
    """
    移除词中的标点符号
    :param word: 单个词
    :return: 移除标点后的词
    """
    # 定义中文和英文的标点符号
    punctuation = r'，。！？：；“”‘’（）【】《》—…～￥·、-'
    # 使用正则表达式移除标点符号
    return re.sub(f'[{punctuation}]+', '', word)

def generate_wordcloud(segmented_file, output_image, font_path, stopwords=None, max_words=200, background_color='white'):
    """
    生成词云图，支持停用词和标点符号过滤

    :param segmented_file: 已经分好词的文件，每个词之间用空格隔开
    :param output_image: 输出的词云图片文件名
    :param font_path: 字体文件路径，用于显示中文
    :param stopwords: 停用词集合，用于过滤不需要的词和标点符号
    :param max_words: 词云显示的最大词数
    :param background_color: 背景颜色
    """
    # 检查分词文件是否存在
    if not os.path.exists(segmented_file):
        print(f"分词文件 {segmented_file} 不存在，请检查路径。")
        return

    # 检查字体文件是否存在
    if not os.path.exists(font_path):
        print(f"字体文件 {font_path} 不存在，请检查路径。")
        return

    # 读取分词后的文本
    with open(segmented_file, 'r', encoding='utf-8') as file:
        text = file.read()

    # 分词列表
    words = text.split()

    # 预处理：移除标点符号
    words = [remove_punctuation(word) for word in words]

    # 如果提供了停用词，进行过滤
    if stopwords:
        words = [word for word in words if word and word not in stopwords]

    # 统计词频
    word_freq = Counter(words)

    # 创建词云对象
    wc = WordCloud(
        font_path=font_path,
        width=800,
        height=600,
        max_words=max_words,
        background_color=background_color,
        collocations=False,  # 避免生成重复的短语
        # 可以根据需要添加其他参数，例如 mask, colormap 等
    )

    # 生成词云
    wc.generate_from_frequencies(word_freq)

    # 显示词云
    plt.figure(figsize=(10, 8))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout(pad=0)

    # 保存词云图片
    os.makedirs(os.path.dirname(output_image), exist_ok=True)
    wc.to_file(output_image)

    # 显示图像
    plt.show()

def main():
    """
    主函数，生成词云图
    """
    # 文件路径配置
    segmented_file = '../Result/forward_segment_result.txt'  # 已经分好词的文件路径
    stopwords_file = 'cn_stopwords.txt'  # 停用词文件路径
    output_image = 'Result/wordcloud_forward.png'           # 输出词云图片的路径
    font_path = 'C:/Windows/Fonts/simhei.ttf'        # 替换为您的中文字体路径

    # 加载停用词
    print("加载停用词...")
    stopwords = load_stopwords(stopwords_file)
    print(f"停用词的个数：{len(stopwords)}")

    # 生成词云
    print("生成词云图...")
    generate_wordcloud(segmented_file, output_image, font_path, stopwords=stopwords)
    print(f"词云图已保存到 {output_image}")

if __name__ == '__main__':
    main()
