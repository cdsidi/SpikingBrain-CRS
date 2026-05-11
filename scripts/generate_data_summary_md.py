import json
import os
import statistics

BASE = "results"
MODELS = ["t_base", "t_crs", "s_base", "s_crs"]
DATASETS = ["lc25000", "physionet", "iu_xray"]


def load_global(model: str):
    p = os.path.join(BASE, f"full_protocol_{model}", "global_summary.json")
    if not os.path.exists(p):
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def mean_of(rows, key_path, default=0.0):
    out = []
    for r in rows:
        cur = r
        for k in key_path:
            if isinstance(cur, dict):
                cur = cur.get(k)
            else:
                cur = None
                break
        out.append(float(cur if cur is not None else default))
    return statistics.mean(out) if out else 0.0


md = []
md.append("# 绘图数据汇总（先不画图）")
md.append("")
md.append("## 1) 四模型×三数据集×5折训练完成性检查")
md.append("")
md.append("| 模型 | lc25000 fold_results | physionet fold_results | iu_xray fold_results | 结论 |")
md.append("|---|---:|---:|---:|---|")
all_ok = True

for m in MODELS:
    j = load_global(m)
    if j is None:
        md.append(f"| {m} | 0 | 0 | 0 | 缺失 global_summary.json |")
        all_ok = False
        continue
    counts = []
    row_ok = True
    for d in DATASETS:
        n = len(j.get(d, {}).get("fold_results", []))
        counts.append(n)
        if n < 25:
            row_ok = False
    if not row_ok:
        all_ok = False
    status = "完成(每数据集25=5折×5重复)" if row_ok else "未完成"
    md.append(f"| {m} | {counts[0]} | {counts[1]} | {counts[2]} | {status} |")

md.append("")
md.append(f"**总体结论：{'已完成' if all_ok else '未完成'}**")
md.append("")

md.append("## 2) Table.1 四模型完整性能对比（汇总均值）")
md.append("")
md.append("| 模型 | 数据集 | Accuracy(mean) | F1(mean) | AUC(mean) | PeakMem(MB) | Throughput(samples/s) |")
md.append("|---|---|---:|---:|---:|---:|---:|")

for m in MODELS:
    j = load_global(m)
    if j is None:
        continue
    for d in DATASETS:
        folds = j.get(d, {}).get("fold_results", [])
        if not folds:
            continue
        acc = mean_of(folds, ["eval", "accuracy"])
        f1 = mean_of(folds, ["eval", "f1"])
        auc = mean_of(folds, ["eval", "auc_roc"])
        mem = mean_of(folds, ["performance", "peak_memory_MB"])
        thr = mean_of(folds, ["performance", "throughput_samples_per_s"])
        md.append(f"| {m} | {d} | {acc:.4f} | {f1:.4f} | {auc:.4f} | {mem:.2f} | {thr:.2f} |")

md.append("")
md.append("## 3) Fig/Table 数据可获得性清单")
md.append("")
md.append("| 项目 | 数据状态 | 数据文件/来源 | 备注 |")
md.append("|---|---|---|---|")
md.append("| Fig.1 架构图 | 可获得（概念图） | `fangan.md`, `sheji/01~12` | 非训练数值图 |")
md.append("| Fig.2 三数据集性能对比 | 可获得 | `results/plot_data/fig2_dataset_comparison.csv` + 各 `global_summary.json` | 建议按数据集分别作图 |")
md.append("| Fig.3 长期记忆保持曲线(10周期) | 可获得（真实训练评估驱动） | `results/long_term_memory_real/long_term_memory_curve.json|csv|png` | 每周期真实调用 `run_train` |")
md.append("| Fig.4 消融瀑布图 | 可获得 | `results/ablation_framework_final/ablation_results.json`, `results/plot_data/fig4_ablation_waterfall.csv` | 已跑通 |")
md.append("| Fig.5 帕累托前沿 | 可获得 | `results/plot_data/fig5_pareto.csv` | 稀疏度+准确率 |")
md.append("| Fig.6 硬件效率雷达图 | 部分可获得 | `results/plot_data/fig6_hardware_radar.csv` | `power_w` 目前缺失 |")
md.append("| Fig.7 案例注意力热力图 | 暂缺 | 暂无统一导出产物 | 需新增可视化导出脚本 |")
md.append("| Table.1 四模型完整性能对比 | 可获得 | `results/plot_data/table1_model_comparison.csv` + 本文汇总表 | 可直接整理三线表 |")
md.append("| Table.2 与SOTA对比 | 暂缺 | 项目内无SOTA基线文件 | 需外部基线录入 |")

fig3_path = "results/long_term_memory_real/long_term_memory_curve.json"
if os.path.exists(fig3_path):
    with open(fig3_path, "r", encoding="utf-8") as f:
        j3 = json.load(f)
    md.append("")
    md.append("## 4) Fig.3 样例数据（真实驱动）")
    md.append("")
    md.append(f"- driver: `{j3.get('driver')}`")
    md.append(f"- model: `{j3.get('model')}`, dataset_a: `{j3.get('dataset_a')}`, dataset_b: `{j3.get('dataset_b')}`")
    md.append(f"- cycles: `{j3.get('cycles')}`, epochs_per_cycle: `{j3.get('epochs_per_cycle')}`, quick_smoke: `{j3.get('quick_smoke')}`")
    md.append(f"- acc_a_curve: `{j3.get('acc_a_curve')}`")
    md.append(f"- acc_b_curve: `{j3.get('acc_b_curve')}`")

out = "results/plot_data/DATA_SUMMARY.md"
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w", encoding="utf-8") as f:
    f.write("\n".join(md) + "\n")

print("WROTE", out)

