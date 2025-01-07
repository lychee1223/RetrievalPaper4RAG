import numpy as np

def recall_at_k(preds: list[int], GT: list[int], k: int) -> float:
    """
    Recall@kを計算する
    :param preds: 予測結果
    :param GT: 正解
    :param k: k]
    :return: Recall@k
    """
    relevant = len(set(preds[:k]) & set(GT))
    return relevant / min(len(GT), k) if len(GT) > 0 else 0

def ndcg_at_k(preds: list[int], GT: list[int], k: int) -> float:
    """
    nDCG@kを計算する
    :param preds: 予測結果
    :param GT: 正解
    :param k: k
    :return: nDCG@k
    """
    dcg = sum([
        1.0 / np.log2(i + 2) for i, pred in enumerate(preds[:k]) if pred in GT
    ])
    idcg = sum([
        1.0 / np.log2(i + 2) for i in range(min(len(GT), k))
    ])
    return dcg / idcg if idcg > 0 else 0

def map(preds: list[int], GT: list[int]) -> float:
    """
    mAPを計算する
    :param preds: 予測結果
    :param GT: 正解
    :return: mAP
    """
    relevant = 0
    ap_sum = 0
    for i, pred in enumerate(preds):
        if pred in GT:
            relevant += 1
            ap_sum += relevant / (i + 1)
    return ap_sum / len(GT) if len(GT) > 0 else 0

def mrr(preds: list[int], GT: list[int]) -> float:
    """
    MRRを計算する
    :param preds: 予測結果
    :param GT: 正解
    :return: MRR
    """
    for i, pred in enumerate(preds):
        if pred in GT:
            return 1 / (i + 1)
    return 0