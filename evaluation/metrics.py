"""医学指标评估实现。"""

from __future__ import annotations

from typing import Dict

import numpy as np


class MedicalMetrics:
    """医学分类指标（支持二分类与多分类宏平均）。

    输入:
        y_true: ndarray, shape [N], dtype int
        y_pred: ndarray, shape [N], dtype int
        y_prob: ndarray, shape [N] or [N, C], dtype float
    """

    @staticmethod
    def _binary_confusion(y_true: np.ndarray, y_pred: np.ndarray, pos_label: int = 1) -> tuple[int, int, int, int]:
        tp = int(np.sum((y_true == pos_label) & (y_pred == pos_label)))
        tn = int(np.sum((y_true != pos_label) & (y_pred != pos_label)))
        fp = int(np.sum((y_true != pos_label) & (y_pred == pos_label)))
        fn = int(np.sum((y_true == pos_label) & (y_pred != pos_label)))
        return tp, tn, fp, fn

    @staticmethod
    def _safe_div(a: float, b: float) -> float:
        return float(a / b) if b != 0 else 0.0

    @staticmethod
    def _binary_auc_roc(y_true_bin: np.ndarray, score: np.ndarray) -> float:
        order = np.argsort(score)
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(1, len(score) + 1)
        pos = y_true_bin == 1
        n_pos = int(pos.sum())
        n_neg = int((~pos).sum())
        if n_pos == 0 or n_neg == 0:
            return 0.5
        sum_ranks_pos = float(ranks[pos].sum())
        auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
        return float(max(0.0, min(1.0, auc)))

    @staticmethod
    def _binary_auc_pr(y_true_bin: np.ndarray, score: np.ndarray) -> float:
        desc = np.argsort(-score)
        y = y_true_bin[desc]
        tp = np.cumsum(y == 1)
        fp = np.cumsum(y == 0)
        precision = tp / np.maximum(tp + fp, 1)
        recall = tp / max(int(np.sum(y == 1)), 1)
        precision = np.concatenate([[1.0], precision])
        recall = np.concatenate([[0.0], recall])
        return float(np.trapz(precision, recall))

    @staticmethod
    def compute_all(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict:
        """计算 Accuracy/Precision/Recall/F1/AUC/Sensitivity/Specificity。"""
        y_true = np.asarray(y_true).astype(int)
        y_pred = np.asarray(y_pred).astype(int)
        y_prob = np.asarray(y_prob)

        acc = float(np.mean(y_true == y_pred))
        classes = np.unique(y_true)

        if classes.size <= 2:
            tp, tn, fp, fn = MedicalMetrics._binary_confusion(y_true, y_pred, pos_label=1)
            precision = MedicalMetrics._safe_div(tp, tp + fp)
            recall = MedicalMetrics._safe_div(tp, tp + fn)
            f1 = MedicalMetrics._safe_div(2 * precision * recall, precision + recall)
            sensitivity = recall
            specificity = MedicalMetrics._safe_div(tn, tn + fp)
            if y_prob.ndim == 2:
                score = y_prob[:, 1] if y_prob.shape[1] > 1 else y_prob[:, 0]
            else:
                score = y_prob
            y_bin = (y_true == 1).astype(int)
            n_pos = int(np.sum(y_bin == 1))
            n_neg = int(np.sum(y_bin == 0))
            auc_valid = bool(n_pos > 0 and n_neg > 0)
            auc_note = "ok" if auc_valid else "single_class_in_y_true"
            auc_roc = MedicalMetrics._binary_auc_roc(y_bin, score.astype(float))
            auc_pr = MedicalMetrics._binary_auc_pr(y_bin, score.astype(float))
        else:
            auc_valid = True
            auc_note = "ok"
            precisions, recalls, f1s, auc_rocs, auc_prs = [], [], [], [], []
            for c in classes:
                y_true_c = (y_true == c).astype(int)
                y_pred_c = (y_pred == c).astype(int)
                tp, tn, fp, fn = MedicalMetrics._binary_confusion(y_true_c, y_pred_c, pos_label=1)
                p = MedicalMetrics._safe_div(tp, tp + fp)
                r = MedicalMetrics._safe_div(tp, tp + fn)
                f = MedicalMetrics._safe_div(2 * p * r, p + r)
                if y_prob.ndim == 2 and y_prob.shape[1] > int(c):
                    score_c = y_prob[:, int(c)].astype(float)
                else:
                    score_c = y_pred_c.astype(float)
                auc_rocs.append(MedicalMetrics._binary_auc_roc(y_true_c, score_c))
                auc_prs.append(MedicalMetrics._binary_auc_pr(y_true_c, score_c))
                precisions.append(p)
                recalls.append(r)
                f1s.append(f)
            precision = float(np.mean(precisions))
            recall = float(np.mean(recalls))
            f1 = float(np.mean(f1s))
            sensitivity = recall
            specificity = 0.0
            auc_roc = float(np.mean(auc_rocs)) if auc_rocs else 0.5
            auc_pr = float(np.mean(auc_prs)) if auc_prs else 0.5

        return {
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "auc_roc": auc_roc,
            "auc_pr": auc_pr,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "support": int(len(y_true)),
            "auc_valid": bool(auc_valid),
            "auc_note": str(auc_note),
        }
