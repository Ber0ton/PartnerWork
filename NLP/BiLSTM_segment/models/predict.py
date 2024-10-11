# models/predict.py

import torch
from utils.data_loader import Vocab
from models.bilstm_segmentation import BiLSTMCRF


def load_model(model_path, vocab_path):
    """
    加载模型和词汇表
    """
    # 读取词汇表
    with open(vocab_path, 'r', encoding='utf-8') as f:
        chars = f.read().strip().split('\n')
    vocab = Vocab(min_freq=1)
    vocab.char2idx = {char: idx for idx, char in enumerate(chars)}
    vocab.idx2char = {idx: char for char, idx in vocab.char2idx.items()}
    vocab.tag2idx = {'B': 0, 'I': 1, 'O': 2, 'E': 3, 'S': 4}
    vocab.idx2tag = {idx: tag for tag, idx in vocab.tag2idx.items()}

    # 初始化模型
    model = BiLSTMCRF(vocab_size=len(vocab.char2idx), tagset_size=len(vocab.tag2idx), embedding_dim=128, hidden_dim=256)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model, vocab


def segment_sentence(model, vocab, sentence):
    """
    对单个句子进行分词
    """
    # 转换为索引
    char_ids = [vocab.char2idx.get(char, vocab.char2idx['<UNK>']) for char in sentence]
    lengths = torch.tensor([len(char_ids)], dtype=torch.long)
    char_tensor = torch.tensor([char_ids], dtype=torch.long)

    # 预测
    preds = model.predict(char_tensor, lengths)
    preds = preds[0]

    # 组装分词结果
    words = []
    word = []
    for char, tag in zip(sentence, [vocab.idx2tag[idx] for idx in preds]):
        if tag == 'B':
            if word:
                words.append(''.join(word))
                word = []
            word.append(char)
        elif tag == 'I':
            word.append(char)
        elif tag == 'E':
            word.append(char)
            words.append(''.join(word))
            word = []
        elif tag == 'S':
            if word:
                words.append(''.join(word))
                word = []
            words.append(char)
        elif tag == 'O':
            if word:
                words.append(''.join(word))
                word = []
            words.append(char)
    if word:
        words.append(''.join(word))
    return ' '.join(words)


def main():
    model_path = r'../models/bilstm_segmentation.pth'
    vocab_path = r'../data/vocab.txt'  # 已保存的词汇表
    model, vocab = load_model(model_path, vocab_path)

    # 示例句子
    sentence = "每个人都是生活的主角，今天我们关注的主角是他们。"
    segmented = segment_sentence(model, vocab, sentence)
    print(segmented)


if __name__ == '__main__':
    main()
