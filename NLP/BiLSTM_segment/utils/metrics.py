# utils/metrics.py

from sklearn.metrics import precision_score, recall_score, f1_score

def compute_metrics(all_preds, all_labels, tag2idx):
    """
    计算精确率、召回率和 F1 值。
    """
    # 需要将标签索引转换为标签名称，忽略 <START> 和 <STOP>
    idx2tag = {idx: tag for tag, idx in tag2idx.items()}
    valid_tags = set(tag2idx.keys())

    y_true = []
    y_pred = []

    for preds, labels in zip(all_preds, all_labels):
        for p, l in zip(preds, labels):
            tag_p = idx2tag.get(p, 'O')
            tag_l = idx2tag.get(l, 'O')
            if tag_p in valid_tags:
                y_pred.append(tag_p)
            if tag_l in valid_tags:
                y_true.append(tag_l)

    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    return precision, recall, f1
