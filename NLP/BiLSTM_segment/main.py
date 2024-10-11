# main.py

import torch
from torch.utils.data import DataLoader
from utils.data_loader import read_data, Vocab, SegmentationDataset, collate_fn
from models.bilstm_segmentation import BiLSTMCRF
from utils.metrics import compute_metrics
from torch.utils.tensorboard import SummaryWriter
import os


def save_segmentation(preds, chars, lengths, idx2tag, filename):
    """
    将分词结果保存到文件，每个词之间用空格隔开。
    """
    with open(filename, 'w', encoding='utf-8') as f:
        for i in range(len(lengths)):
            words = []
            word = []
            for j in range(lengths[i]):
                char = chars[i][j]
                tag = idx2tag.get(preds[i][j], 'O')  # 使用 'O' 作为默认标签
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
            f.write(' '.join(words) + '\n')


def main():
    # 文件路径
    train_file = 'data/train.txt'
    test_file = 'data/test.txt'

    # 读取训练数据
    train_data = read_data(train_file)
    print(f"训练集大小: {len(train_data)}")

    # 读取测试数据
    test_data = read_data(test_file)
    print(f"测试集大小: {len(test_data)}\n")

    # 构建词汇表
    vocab = Vocab(min_freq=1)
    vocab.build_vocab(train_data)
    print(f"词典大小: {len(vocab.char2idx)}")
    print(f"标签集: {vocab.tag2idx}\n")

    # 保存词汇表
    with open('data/vocab.txt', 'w', encoding='utf-8') as f:
        for char in vocab.char2idx:
            f.write(f"{char}\n")
    print("词汇表已保存到 'data/vocab.txt'。")

    # 创建数据集
    train_dataset = SegmentationDataset(train_data, vocab)
    test_dataset = SegmentationDataset(test_data, vocab)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)

    # 模型、损失函数和优化器
    model = BiLSTMCRF(vocab_size=len(vocab.char2idx), tagset_size=len(vocab.tag2idx), embedding_dim=128, hidden_dim=256)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 创建 SummaryWriter
    writer = SummaryWriter(log_dir='runs/segmentation_experiment')

    # 训练循环
    num_epochs = 120
    global_step = 0  # 全局步数计数器
    best_f1 = 0
    patience = 10
    trigger_times = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch_chars, batch_tags, lengths in train_loader:
            batch_chars, batch_tags, lengths = batch_chars.to(device), batch_tags.to(device), lengths.to(device)
            optimizer.zero_grad()
            loss = model(batch_chars, batch_tags, lengths)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # 记录每个 batch 的损失
            writer.add_scalar('Train/Loss', loss.item(), global_step)
            global_step += 1

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

        # 在 TensorBoard 中记录每个 epoch 的平均损失
        writer.add_scalar('Train/Average_Loss', avg_loss, epoch + 1)

        # 评估
        model.eval()
        with torch.no_grad():
            all_preds = []
            all_labels = []
            all_chars = []
            all_lengths = []
            for batch_chars, batch_tags, lengths in test_loader:
                batch_chars, batch_tags, lengths = batch_chars.to(device), batch_tags.to(device), lengths.to(device)
                preds = model.predict(batch_chars, lengths)
                all_preds.extend(preds)
                all_labels.extend(batch_tags.cpu().numpy())
                all_lengths.extend(lengths.cpu().numpy())

            # 调试信息
            total_true = 0
            total_pred = 0
            for pred, label, length in zip(all_preds, all_labels, all_lengths):
                total_true += length
                total_pred += len(pred)
                if len(pred) != length:
                    print(f"长度不匹配 - 预测长度: {len(pred)}, 真实长度: {length}")

            print(f"总真实标签数量: {total_true}, 总预测标签数量: {total_pred}")

            precision, recall, f1 = compute_metrics(all_preds, all_labels, vocab.tag2idx)
            print(f"验证集 - 精确率: {precision:.4f}, 召回率: {recall:.4f}, F1值: {f1:.4f}\n")

            # 在 TensorBoard 中记录验证指标
            writer.add_scalar('Validation/Precision', precision, epoch + 1)
            writer.add_scalar('Validation/Recall', recall, epoch + 1)
            writer.add_scalar('Validation/F1', f1, epoch + 1)

            # 早停机制
            if f1 > best_f1:
                best_f1 = f1
                trigger_times = 0
                os.makedirs('models', exist_ok=True)
                torch.save(model.state_dict(), 'models/bilstm_segmentation_best.pth')
                print("保存了最佳模型。")
            else:
                trigger_times += 1
                if trigger_times >= patience:
                    print("早停机制触发，停止训练。")
                    break

    # 关闭 SummaryWriter
    writer.close()

    # 保存模型
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), 'models/bilstm_segmentation.pth')
    print("模型已保存。")

    # 保存分词结果
    print("保存分词结果...")
    with torch.no_grad():
        all_preds = []
        all_chars = []
        all_lengths = []
        for batch_chars, batch_tags, lengths in test_loader:
            batch_chars, lengths = batch_chars.to(device), lengths.to(device)
            preds = model.predict(batch_chars, lengths)
            all_preds.extend(preds)
            all_chars.extend(batch_chars.cpu().numpy())
            all_lengths.extend(lengths.cpu().numpy())

    # 转换为字符列表
    all_chars_list = []
    for chars in all_chars:
        word = []
        for char_idx in chars:
            if char_idx == 0:
                word.append('<PAD>')
            else:
                word.append(vocab.idx2char.get(char_idx, '<UNK>'))
        all_chars_list.append(word)

    # 保存
    save_segmentation(all_preds, all_chars_list, all_lengths, vocab.idx2tag, 'results/predicted_segmentation.txt')
    print("分词结果已保存到 'results/predicted_segmentation.txt'。")


if __name__ == '__main__':
    main()
