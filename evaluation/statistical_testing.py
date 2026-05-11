"""统计检验实现（轻量版，不依赖 scipy）。"""

from __future__ import annotations

import math
from typing import Dict, List

import numpy as np


class StatisticalTesting:
    """统计检验工具集。

    提供: 配对 t 检验、Cohen's d、单因素 ANOVA。
    """

    @staticmethod
    def paired_ttest(results_a: List[float], results_b: List[float]) -> Dict:
        """配对 t 检验。

        Returns:
            dict: {"t_stat", "p_value_approx", "df", "mean_diff"}
        """
        a = np.asarray(results_a, dtype=float)
        b = np.asarray(results_b, dtype=float)
        if a.shape != b.shape:
            raise ValueError("results_a and results_b must have the same length")
        n = int(a.size)
        if n < 2:
            return {"t_stat": 0.0, "p_value_approx": 1.0, "df": max(0, n - 1), "mean_diff": float((a - b).mean() if n else 0.0)}

        d = a - b
        mean_d = float(np.mean(d))
        std_d = float(np.std(d, ddof=1))
        if std_d == 0.0:
            t_stat = 0.0
        else:
            t_stat = mean_d / (std_d / math.sqrt(n))

        # 正态近似双尾 p 值（轻量近似，避免重依赖）
        p_approx = float(math.erfc(abs(t_stat) / math.sqrt(2.0)))
        return {"t_stat": float(t_stat), "p_value_approx": p_approx, "df": n - 1, "mean_diff": mean_d}

    @staticmethod
    def cohens_d(results_a: List[float], results_b: List[float]) -> float:
        """Cohen's d（独立样本 pooled std 版本）。"""
        a = np.asarray(results_a, dtype=float)
        b = np.asarray(results_b, dtype=float)
        if a.size < 2 or b.size < 2:
            return 0.0
        mean_a = float(np.mean(a))
        mean_b = float(np.mean(b))
        var_a = float(np.var(a, ddof=1))
        var_b = float(np.var(b, ddof=1))
        denom = (a.size + b.size - 2)
        if denom <= 0:
            return 0.0
        pooled = ((a.size - 1) * var_a + (b.size - 1) * var_b) / denom
        if pooled <= 0:
            return 0.0
        return float((mean_a - mean_b) / math.sqrt(pooled))

    @staticmethod
    def anova_oneway(groups: List[List[float]]) -> Dict:
        """单因素 ANOVA。

        Returns:
            dict: {"f_stat", "df_between", "df_within", "ss_between", "ss_within"}
        """
        clean_groups = [np.asarray(g, dtype=float) for g in groups if len(g) > 0]
        k = len(clean_groups)
        if k < 2:
            return {"f_stat": 0.0, "df_between": 0, "df_within": 0, "ss_between": 0.0, "ss_within": 0.0}

        n_total = int(sum(g.size for g in clean_groups))
        if n_total <= k:
            return {"f_stat": 0.0, "df_between": k - 1, "df_within": max(0, n_total - k), "ss_between": 0.0, "ss_within": 0.0}

        grand_mean = float(np.mean(np.concatenate(clean_groups)))
        ss_between = float(sum(g.size * (float(np.mean(g)) - grand_mean) ** 2 for g in clean_groups))
        ss_within = float(sum(np.sum((g - float(np.mean(g))) ** 2) for g in clean_groups))

        df_between = k - 1
        df_within = n_total - k
        ms_between = ss_between / df_between if df_between > 0 else 0.0
        ms_within = ss_within / df_within if df_within > 0 else 0.0
        f_stat = ms_between / ms_within if ms_within > 0 else 0.0

        return {
            "f_stat": float(f_stat),
            "df_between": int(df_between),
            "df_within": int(df_within),
            "ss_between": ss_between,
            "ss_within": ss_within,
        }
