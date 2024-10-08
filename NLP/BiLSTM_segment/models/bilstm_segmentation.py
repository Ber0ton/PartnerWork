# models/bilstm_segmentation.py

import torch
import torch.nn as nn


class BiLSTMCRF(nn.Module):
    def __init__(self, vocab_size, tagset_size, embedding_dim=128, hidden_dim=256):
        super(BiLSTMCRF, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=1, bidirectional=True, batch_first=True)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

        # Transition matrix for CRF
        self.transitions = nn.Parameter(torch.randn(tagset_size, tagset_size))
        # 初始化 transition 的限制，比如禁止某些标签转移
        self.transitions.data[:, 0] = -10000  # 禁止转移到 <PAD>
        self.transitions.data[0, :] = -10000  # 禁止从 <PAD> 转移

    def forward(self, sentences, tags, lengths):
        """
        计算损失
        sentences: tensor of shape (batch_size, seq_len)
        tags: tensor of shape (batch_size, seq_len)
        lengths: tensor of shape (batch_size)
        """
        emissions = self._get_emissions(sentences, lengths)
        loss = self._compute_loss(emissions, tags, lengths)
        return loss

    def _get_emissions(self, sentences, lengths):
        embeds = self.embedding(sentences)  # (batch_size, seq_len, embedding_dim)
        packed_embeds = nn.utils.rnn.pack_padded_sequence(embeds, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_lstm_out, _ = self.lstm(packed_embeds)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(packed_lstm_out, batch_first=True)
        emissions = self.hidden2tag(lstm_out)  # (batch_size, seq_len, tagset_size)
        return emissions

    def _compute_loss(self, emissions, tags, lengths):
        # 计算真实路径的得分
        gold_score = self._score_sentence(emissions, tags, lengths)
        # 计算所有可能路径的得分
        forward_score = self._forward_alg(emissions, lengths)
        # 负对数似然
        loss = forward_score - gold_score
        return loss.mean()

    def _score_sentence(self, emissions, tags, lengths):
        """
        计算给定标签序列的得分
        """
        batch_size, seq_len, tagset_size = emissions.size()
        score = torch.zeros(batch_size).to(emissions.device)
        tags = torch.cat([torch.full((batch_size, 1), 0, dtype=torch.long).to(emissions.device), tags], dim=1)  # 添加开始符
        for i in range(seq_len):
            emit_score = emissions[range(batch_size), i, tags[:, i + 1]]
            trans_score = self.transitions[tags[:, i + 1], tags[:, i]]
            score += emit_score + trans_score
        return score

    def _forward_alg(self, emissions, lengths):
        """
        计算所有路径的总得分（分母）
        使用前向算法
        """
        batch_size, seq_len, tagset_size = emissions.size()
        alpha = torch.full((batch_size, tagset_size), -10000.).to(emissions.device)
        alpha[:, 0] = 0  # 开始符

        for i in range(seq_len):
            emit_score = emissions[:, i].unsqueeze(2)  # (batch_size, tagset_size, 1)
            trans_score = self.transitions.unsqueeze(0)  # (1, tagset_size, tagset_size)
            alpha_t = alpha.unsqueeze(1) + trans_score + emit_score  # (batch_size, tagset_size, tagset_size)
            alpha = torch.logsumexp(alpha_t, dim=2)  # (batch_size, tagset_size)
            # Mask padding
            mask = (i < lengths).float().unsqueeze(1)
            alpha = alpha * mask + alpha * (1 - mask)
        return torch.logsumexp(alpha, dim=1)

    def predict(self, sentences, lengths):
        """
        使用 Viterbi 算法进行预测
        """
        emissions = self._get_emissions(sentences, lengths)
        return self._viterbi_decode(emissions, lengths)

    def _viterbi_decode(self, emissions, lengths):
        batch_size, seq_len, tagset_size = emissions.size()
        backpointers = []

        # 初始化
        viterbi_scores = torch.full((batch_size, tagset_size), -10000.).to(emissions.device)
        viterbi_scores[:, 0] = 0  # 开始符

        for i in range(seq_len):
            emit_score = emissions[:, i].unsqueeze(2)  # (batch_size, tagset_size, 1)
            trans_score = self.transitions.unsqueeze(0)  # (1, tagset_size, tagset_size)
            scores = viterbi_scores.unsqueeze(1) + trans_score + emit_score  # (batch_size, tagset_size, tagset_size)
            best_scores, best_tags = torch.max(scores, dim=2)  # (batch_size, tagset_size)
            backpointers.append(best_tags)
            # 更新 scores
            mask = (i < lengths).float().unsqueeze(1)
            viterbi_scores = best_scores * mask + viterbi_scores * (1 - mask)

        # 最后一步
        best_scores, best_last_tags = torch.max(viterbi_scores, dim=1)  # (batch_size)

        # 回溯
        best_paths = []
        for b in range(batch_size):
            best_tag = best_last_tags[b].item()
            path = [best_tag]
            for i in range(seq_len - 1, -1, -1):
                if i >= lengths[b]:
                    continue
                best_tag = backpointers[i][b][best_tag].item()
                path.append(best_tag)
            # 去除开始符
            path = path[-lengths[b]:]
            path = path[::-1]
            best_paths.append(path)
        return best_paths
