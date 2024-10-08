# utils/data_loader.py

import torch
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict

# 定义标签
TAG_B = 'B'
TAG_I = 'I'
TAG_O = 'O'
TAG_E = 'E'
TAG_S = 'S'

ALL_TAGS = [TAG_B, TAG_I, TAG_O, TAG_E, TAG_S]

def read_data(file_path):
    """
    读取数据文件，每行一个句子，词之间用空格隔开。
    返回一个列表，每个元素是一个元组 (characters, tags)，
    其中 characters 是字符列表，tags 是对应的标签列表。
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            words = line.split()
            characters = []
            tags = []
            for word in words:
                if len(word) == 1:
                    characters.append(word)
                    tags.append(TAG_S)
                else:
                    characters.extend(list(word))
                    tags.extend([TAG_B] + [TAG_I] * (len(word) - 2) + [TAG_E])
            data.append((characters, tags))
    return data


# utils/data_loader.py (继续)

class Vocab:
    """
    字符和标签的映射。
    """

    def __init__(self, min_freq=1):
        self.char2idx = {}
        self.idx2char = {}
        self.tag2idx = {}
        self.idx2tag = {}
        self.min_freq = min_freq
        self.char_freq = defaultdict(int)

    def build_vocab(self, data):
        """
        构建字符和标签的映射表。
        """
        # 统计字符频率
        for characters, tags in data:
            for char in characters:
                self.char_freq[char] += 1

        # 建立字符映射
        self.char2idx = {'<PAD>': 0, '<UNK>': 1}
        for char, freq in self.char_freq.items():
            if freq >= self.min_freq:
                self.char2idx[char] = len(self.char2idx)

        self.idx2char = {idx: char for char, idx in self.char2idx.items()}

        # 建立标签映射
        self.tag2idx = {tag: idx for idx, tag in enumerate(ALL_TAGS)}
        self.idx2tag = {idx: tag for tag, idx in self.tag2idx.items()}

    def encode_chars(self, characters):
        return [self.char2idx.get(char, self.char2idx['<UNK>']) for char in characters]

    def encode_tags(self, tags):
        return [self.tag2idx[tag] for tag in tags]


class SegmentationDataset(Dataset):
    def __init__(self, data, vocab):
        """
        data: list of (characters, tags)
        vocab: Vocab object
        """
        self.data = data
        self.vocab = vocab
        self.char_ids = [vocab.encode_chars(chars) for chars, _ in data]
        self.tag_ids = [vocab.encode_tags(tags) for _, tags in data]

    def __len__(self):
        return len(self.char_ids)

    def __getitem__(self, idx):
        return self.char_ids[idx], self.tag_ids[idx]


# utils/data_loader.py (继续)

def collate_fn(batch):
    """
    自定义的 collate_fn，用于动态填充序列。
    batch: list of tuples (char_ids, tag_ids)
    返回:
        padded_chars: tensor of shape (batch_size, max_seq_len)
        padded_tags: tensor of shape (batch_size, max_seq_len)
        lengths: tensor of shape (batch_size)
    """
    chars, tags = zip(*batch)
    lengths = [len(c) for c in chars]
    max_len = max(lengths)

    padded_chars = [c + [0] * (max_len - len(c)) for c in chars]
    padded_tags = [t + [0] * (max_len - len(t)) for t in tags]

    return torch.tensor(padded_chars, dtype=torch.long), torch.tensor(padded_tags, dtype=torch.long), torch.tensor(
        lengths, dtype=torch.long)

