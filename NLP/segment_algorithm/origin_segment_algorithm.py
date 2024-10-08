# -*- coding: utf-8 -*-
import re
import time
from tqdm import tqdm  # 新增：用于显示进度条
from collections import Counter

def calculate_run_time(func):
    """
    装饰器函数，用于计算函数运行时间
    """

    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"函数 '{func.__name__}' 耗时 {end_time - start_time:.5f} 秒")
        return result

    return wrapper


def remove_numbers(text):
    """
    移除字符串中的所有数字
    """
    return re.sub(r'\d+', '', text)


def is_chinese(text):
    """
    判断字符串中是否包含中文字符
    """
    return bool(re.search(r'[\u4e00-\u9fa5]', text))


def read_dictionary(file_path):
    """
    读取词典文件，返回词典集合（使用集合提高查找效率）
    """
    dictionary = set()
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            word = remove_numbers(line.strip().replace(" ", ""))
            if word:
                dictionary.add(word)
    return dictionary


def read_jieba_segments(file_path):
    """
    读取jieba分词结果文件，返回词列表
    """
    jieba_words = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            words = remove_numbers(line.strip()).split()
            # jieba_words.extend([word for word in words if is_chinese(word)])
            jieba_words.extend([word for word in words])
    return jieba_words


def read_text(file_path):
    """
    读取文本文件，返回文本列表
    """
    texts = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            text = line.replace(" ", "")
            if text:
                texts.append(text)
    return texts


@calculate_run_time
def forward_segment(texts, dictionary, disable_tqdm=False):
    """
    优化后的正向最大匹配算法，保留数字序列
    """
    word_list = []
    if not disable_tqdm:
        print("开始正向最大匹配分词...")

    max_word_length = max(len(w) for w in dictionary)  # 预先计算词典中词语的最大长度

    for text in tqdm(texts, desc="正向分词进度", disable=disable_tqdm):
        index = 0
        while index < len(text):
            # 检查是否为数字序列
            if text[index].isdigit():
                start_index = index
                while index < len(text) and (text[index].isdigit() or text[index] == '.'):
                    index += 1
                # if index < len(text) and (text[index] == '年' or text[index] == '月' or text[index] == '日'):
                #     index += 1
                index += 1
                number = text[start_index:index]
                word_list.append(number)
            elif text[index].isalpha() and text[index].isascii():
                start2_index = index
                while index < len(text) and ((text[index].isalpha() and text[index].isascii())or text[index] == '/'):
                    index += 1
                word = text[start2_index:index]
                word_list.append(word)
            else:
                max_length = min(max_word_length, len(text) - index)  # 使用预先计算的最大词长
                longest_word = text[index:index + 1]  # 初始化为单字
                for length in range(max_length, 0, -1):  # 从最大长度开始尝试匹配
                    word = text[index:index + length]
                    if word in dictionary:
                        longest_word = word
                        break  # 找到最长匹配词，退出循环
                word_list.append(longest_word)
                index += len(longest_word)
    return word_list


@calculate_run_time
def backward_segment(texts, dictionary, disable_tqdm=False):
    """
    优化后的逆向最大匹配算法，保留数字序列和英文单词不被分割。

    参数：
        texts (list of str): 需要分词的文本列表。
        dictionary (set): 分词词典，包含所有有效的中文词语。
        disable_tqdm (bool): 是否禁用进度条显示。默认为False。

    返回：
        list of str: 分词后的词语列表。
    """

    word_list = []

    if not disable_tqdm:
        print("开始逆向最大匹配分词...")

    max_word_length = max(len(w) for w in dictionary)

    for text in tqdm(texts, desc="逆向分词进度", disable=disable_tqdm):
        index = len(text)
        temp_word_list = []
        while index > 0:
            current_char = text[index - 1]

            if current_char.isdigit():
                # 处理数字序列
                end_index = index
                while index > 0 and text[index - 1].isdigit():
                    index -= 1
                number = text[index:end_index]
                temp_word_list.append(number)

            elif (current_char.isalpha() and current_char.isascii()):
                # 处理英文单词
                end_index = index
                while index > 0 and (text[index - 1]).isalpha() and (text[index - 1]).isascii():
                    index -= 1
                english_word = text[index:end_index]
                temp_word_list.append(english_word)

            else:
                # 处理中文词语
                max_length = min(max_word_length, index)  # 使用预先计算的最大长度
                longest_word = text[index - 1:index]  # 默认单字
                for length in range(max_length, 0, -1):
                    start_index = index - length
                    if start_index < 0:
                        continue
                    word = text[start_index:index]
                    if word in dictionary:
                        longest_word = word
                        index = start_index
                        break
                else:
                    # 没有匹配到词典中的词，向前移动一位
                    index -= 1
                temp_word_list.append(longest_word)

        # 由于是逆向遍历，需将临时列表反转后添加到最终结果中
        word_list.extend(reversed(temp_word_list))

    return word_list


def count_single_char_words(words):
    """
    统计单字词数量
    """
    return sum(1 for word in words if len(word) == 1)


@calculate_run_time
def bidirectional_segment(texts, dictionary):
    """
    双向最大匹配算法
    """
    print("开始双向最大匹配分词...")
    # 禁用内部的进度条，避免重复显示
    forward_result = forward_segment(texts, dictionary, disable_tqdm=True)
    backward_result = backward_segment(texts, dictionary, disable_tqdm=True)

    if len(forward_result) < len(backward_result):
        return forward_result
    elif len(forward_result) > len(backward_result):
        return backward_result
    else:
        if count_single_char_words(forward_result) < count_single_char_words(backward_result):
            return forward_result
        else:
            return backward_result


def calculate_metrics(correct_count, total_jieba_words, segmented_words):
    """
    计算准确率、召回率和F1值
    """
    precision = correct_count / len(segmented_words)
    recall = correct_count / total_jieba_words
    f1_score = 2 * precision * recall / (precision + recall)
    return precision, recall, f1_score


def print_results(correct_count, total_jieba_words, segmented_words, method_name):
    """
    输出分词结果和评估指标
    """
    precision, recall, f1_score = calculate_metrics(correct_count, total_jieba_words, segmented_words)
    print(f"{method_name}分词的个数：{len(segmented_words)}")
    print(f"准确率：{precision * 100:.2f}%")
    print(f"召回率：{recall * 100:.2f}%")
    print(f"F1值：{f1_score * 100:.2f}%\n")


def save_to_file(words, filename):
    """
    将分词结果保存到文件
    """
    with open(filename, 'w', encoding='utf-8') as file:
        for i, word in enumerate(words, 1):
            file.write(f"{word} ")
            if i % 100 == 0:
                file.write('\n')


def main():
    """
    主函数，执行分词和评估流程
    """
    # 文件路径
    raw_text_file = '../Data/RawText__xinhua.txt'  # 原始文本文件
    dictionary_file = '../Data/new_wordlist2.Dic'  # 词典文件
    jieba_segment_file = '../Result/jieba_Segment.txt'  # jieba分词结果文件

    # 读取数据
    print("读取词典文件...")
    dictionary = read_dictionary(dictionary_file)
    print("读取jieba分词结果文件...")
    jieba_words = read_jieba_segments(jieba_segment_file)
    print("读取原始文本文件...")
    texts = read_text(raw_text_file)

    print(f"词典词的个数：{len(dictionary)}")
    print(f"jieba分词的个数：{len(jieba_words)}\n")

    # 分词处理
    forward_words = forward_segment(texts, dictionary)
    backward_words = backward_segment(texts, dictionary)
    bidirectional_words = bidirectional_segment(texts, dictionary)

    # 评估结果
    print("\n开始评估分词结果...")
    # 将 jieba_words 转换为 Counter 来统计词频
    jieba_word_counter = Counter(jieba_words)

    # 计算 forward 分词的正确个数
    correct_forward = 0
    for word in forward_words:
        if word in jieba_word_counter and jieba_word_counter[word] > 0:
            correct_forward += 1
            jieba_word_counter[word] -= 1  # 减少对应词的剩余次数

    # 将 Counter 重置以进行 backward 分词的评估
    jieba_word_counter = Counter(jieba_words)
    correct_backward = 0
    for word in backward_words:
        if word in jieba_word_counter and jieba_word_counter[word] > 0:
            correct_backward += 1
            jieba_word_counter[word] -= 1  # 减少对应词的剩余次数

    # 将 Counter 重置以进行 bidirectional 分词的评估
    jieba_word_counter = Counter(jieba_words)
    correct_bidirectional = 0
    for word in bidirectional_words:
        if word in jieba_word_counter and jieba_word_counter[word] > 0:
            correct_bidirectional += 1
            jieba_word_counter[word] -= 1  # 减少对应词的剩余次数

    # 计算 jieba 分词总数
    total_jieba_words = len(jieba_words)

    # 输出结果
    print_results(correct_forward, total_jieba_words, forward_words, "正向最大匹配")
    print_results(correct_backward, total_jieba_words, backward_words, "逆向最大匹配")
    print_results(correct_bidirectional, total_jieba_words, bidirectional_words, "双向最大匹配")

    # 保存分词结果（可选）
    save_to_file(forward_words, '../Result/forward_segment_result.txt')
    save_to_file(backward_words, '../Result/backward_segment_result.txt')
    save_to_file(bidirectional_words, '../Result/bidirectional_segment_result.txt')


if __name__ == '__main__':
    main()
