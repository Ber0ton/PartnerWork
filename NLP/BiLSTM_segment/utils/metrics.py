# utils/metrics.py

from collections import defaultdict, Counter


def compute_metrics(preds, labels, tag2idx):
    """
    计算精确率、召回率和 F1 值。
    使用 BIOES 标注方式计算。
    """
    # 将标签索引转换为标签
    idx2tag = {idx: tag for tag, idx in tag2idx.items()}

    def get_entities(tags):
        """
        从标签序列中提取实体
        返回实体的起始和结束索引及其类型
        """
        entities = []
        entity = []
        for i, tag in enumerate(tags):
            # 不需要再次转换标签
            if tag == 'B':
                if entity:
                    entities.append(tuple(entity))
                    entity = []
                entity = [i, 'B']
            elif tag == 'I':
                if entity and entity[-1] in ['B', 'I']:
                    entity[-1] = 'I'
                else:
                    entity = [i, 'I']
            elif tag == 'E':
                if entity and entity[-1] in ['B', 'I']:
                    entity[-1] = 'E'
                    entities.append(tuple(entity))
                    entity = []
                else:
                    entity = [i, 'E']
            elif tag == 'S':
                entities.append((i, 'S'))
            else:
                if entity:
                    entities.append(tuple(entity))
                    entity = []
        if entity:
            entities.append(tuple(entity))
        return entities

    # 统计真实实体和预测实体
    true_entities = []
    pred_entities = []

    for label_seq, pred_seq in zip(labels, preds):
        true_tags = [idx2tag[idx] for idx in label_seq]
        pred_tags = [idx2tag[idx] for idx in pred_seq]
        true_entities.extend(get_entities(true_tags))
        pred_entities.extend(get_entities(pred_tags))

    # 使用 Counter 统计真实实体和预测实体的频次
    true_counter = Counter(true_entities)
    pred_counter = Counter(pred_entities)

    # 计算正确预测的实体数
    correct = 0
    for ent, cnt in pred_counter.items():
        correct += min(cnt, true_counter.get(ent, 0))

    precision = correct / sum(pred_counter.values()) if sum(pred_counter.values()) > 0 else 0
    recall = correct / sum(true_counter.values()) if sum(true_counter.values()) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1
