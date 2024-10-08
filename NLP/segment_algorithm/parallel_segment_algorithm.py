# -*- coding: utf-8 -*-
import re
import time
from tqdm import tqdm
import concurrent.futures
from multiprocessing import Pool, Manager
import os
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


def is_chinese(word):
    """
    判断字符串中是否包含中文字符
    """
    return bool(re.search(r'[\u4e00-\u9fa5]', word))


def read_dictionary(file_path):
    """
    读取词典文件，返回词典集合和最大词长
    """
    dictionary = set()
    max_word_length = 0
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            word = remove_numbers(line.strip().replace(" ", ""))
            if word:
                dictionary.add(word)
                if len(word) > max_word_length:
                    max_word_length = len(word)
    return dictionary, max_word_length


def read_text(file_path):
    """
    读取文本文件，返回文本内容
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read().replace(" ", "").strip()
    return text


def count_single_char_words(words):
    """
    统计单字词数量
    """
    return sum(1 for word in words if len(word) == 1)


# @calculate_run_time
def forward_segment(text_block, dictionary, max_word_length):
    """
    优化后的正向最大匹配算法，保留数字序列
    """
    word_list = []
    index = 0
    while index < len(text_block):
        # 检查是否为数字序列
        if text_block[index].isdigit():
            start_index = index
            while index < len(text_block) and text_block[index].isdigit():
                index += 1
            index += 1
            number = text_block[start_index:index]
            word_list.append(number)
        elif text_block[index].isalpha() and text_block[index].isascii():
            start2_index = index
            while index < len(text_block) and ((text_block[index].isalpha() and text_block[index].isascii()) or text_block[index] == '/'):
                index += 1
            word = text_block[start2_index:index]
            word_list.append(word)
        else:
            max_length = min(max_word_length, len(text_block) - index)
            longest_word = text_block[index:index + 1]  # 默认单字
            for length in range(max_length, 0, -1):
                word = text_block[index:index + length]
                if word in dictionary:
                    longest_word = word
                    break  # 找到最长匹配词，退出循环
            word_list.append(longest_word)
            index += len(longest_word)
    return word_list


# @calculate_run_time
def backward_segment(text_block, dictionary, max_word_length):
    """
    优化后的逆向最大匹配算法，保留数字序列和英文单词不被分割
    """

    def is_english_letter(char):
        """判断字符是否为英文字母"""
        return char.isalpha() and char.isascii()

    word_list = []
    index = len(text_block)
    temp_word_list = []

    while index > 0:
        current_char = text_block[index - 1]

        if current_char.isdigit():
            # 处理数字序列
            end_index = index
            while index > 0 and text_block[index - 1].isdigit():
                index -= 1
            number = text_block[index:end_index]
            temp_word_list.append(number)

        elif is_english_letter(current_char):
            # 处理英文单词
            end_index = index
            while index > 0 and is_english_letter(text_block[index - 1]):
                index -= 1
            english_word = text_block[index:end_index]
            temp_word_list.append(english_word)

        else:
            # 处理中文词语
            max_length = min(max_word_length, index)
            longest_word = text_block[index - 1:index]  # 默认单字
            for length in range(max_length, 0, -1):
                start_index = index - length
                if start_index < 0:
                    continue
                word = text_block[start_index:index]
                if word in dictionary:
                    longest_word = word
                    index = start_index
                    break
            else:
                index -= 1  # 没有匹配到词典中的词，向前移动一位
            temp_word_list.append(longest_word)

    word_list = list(reversed(temp_word_list))
    return word_list


# @calculate_run_time
def bidirectional_segment(text_block, dictionary, max_word_length):
    """
    双向最大匹配算法
    """
    forward_words = forward_segment(text_block, dictionary, max_word_length)
    backward_words = backward_segment(text_block, dictionary, max_word_length)

    if len(forward_words) < len(backward_words):
        return forward_words
    elif len(forward_words) > len(backward_words):
        return backward_words
    else:
        if count_single_char_words(forward_words) < count_single_char_words(backward_words):
            return forward_words
        else:
            return backward_words

@calculate_run_time
def parallel_segment(text, dictionary, max_word_length, num_workers=4):
    """
    并行处理分词，支持正向、逆向和双向最大匹配
    """
    # 划分文本块
    chunk_size = len(text) // num_workers
    text_blocks = [text[i * chunk_size: (i + 1) * chunk_size] for i in range(num_workers)]
    # 处理最后一个块，确保覆盖所有文本
    if len(text) % num_workers != 0:
        text_blocks[-1] += text[num_workers * chunk_size:]

    results_forward = []
    results_backward = []
    results_bidirectional = []

    with Pool(processes=num_workers) as pool:
        # 正向分词
        forward_results = pool.starmap(forward_segment, [(block, dictionary, max_word_length) for block in text_blocks])
        # 逆向分词
        backward_results = pool.starmap(backward_segment,
                                        [(block, dictionary, max_word_length) for block in text_blocks])
        # 双向分词
        bidirectional_results = pool.starmap(bidirectional_segment,
                                             [(block, dictionary, max_word_length) for block in text_blocks])

    # 合并结果
    for fr in forward_results:
        results_forward.extend(fr)
    for br in backward_results:
        results_backward.extend(br)
    for brd in bidirectional_results:
        results_bidirectional.extend(brd)

    return results_forward, results_backward, results_bidirectional


def calculate_metrics(correct_count, total_jieba_words, segmented_words):
    """
    计算准确率、召回率和F1值
    """
    precision = correct_count / len(segmented_words) if len(segmented_words) > 0 else 0
    recall = correct_count / total_jieba_words if total_jieba_words > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1_score


def print_results(correct_count, total_jieba_words, segmented_words, method_name):
    """
    输出分词结果和评估指标
    """
    precision, recall, f1_score = calculate_metrics(correct_count, total_jieba_words, segmented_words)
    print(f"{method_name}分词的个数：{len(segmented_words)}")
    print(f"正确次数{correct_count}")
    print(f"准确率：{precision * 100:.2f}%")
    print(f"召回率：{recall * 100:.2f}%")
    print(f"F1值：{f1_score * 100:.2f}%\n")


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


def save_to_file(words, filename):
    """
    将分词结果保存到文件
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
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
    dictionary_file = '../Data/new_wordlist.Dic'  # 词典文件
    jieba_segment_file = '../Result/jieba_Segment.txt'  # jieba分词结果文件

    # 读取数据
    print("读取词典文件...")
    dictionary, max_word_length = read_dictionary(dictionary_file)
    print(f"词典词的个数：{len(dictionary)}，最大词长：{max_word_length}")

    print("读取jieba分词结果文件...")
    jieba_words = read_jieba_segments(jieba_segment_file)
    print(f"jieba分词的个数：{len(jieba_words)}\n")

    print("读取原始文本文件...")
    text = read_text(raw_text_file)
    print(f"原始文本长度：{len(text)}\n")

    # 并行分词处理
    num_workers = 4  # 使用4个进程进行并行处理
    print(f"使用 {num_workers} 个进程进行并行分词处理...")
    forward_words, backward_words, bidirectional_words = parallel_segment(text, dictionary, max_word_length,
                                                                          num_workers=num_workers)

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
    save_to_file(forward_words, '../Result/parallel_forward_segment_result.txt')
    save_to_file(backward_words, '../Result/parallel_backward_segment_result.txt')
    save_to_file(bidirectional_words, '../Result/parallel_bidirectional_segment_result.txt')


if __name__ == '__main__':
    main()
