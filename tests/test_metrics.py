import numpy as np
from src.evaluation.metrics import recall_at_k, ndcg_at_k, mrr_at_k


def test_recall_ndcg_mrr_simple():
    preds = [[1, 2, 3], [4, 5, 6]]
    gt = [2, 7]
    assert abs(recall_at_k(preds, gt, 3) - 0.5) < 1e-9
    assert ndcg_at_k(preds, gt, 3) > 0.0
    assert mrr_at_k(preds, gt, 3) > 0.0


