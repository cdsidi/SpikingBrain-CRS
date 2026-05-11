import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List


def read_json(p: Path) -> Dict[str, Any]:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def find_model_summaries(results_dir: Path) -> Dict[str, Dict[str, Any]]:
    out = {}
    for m in ["t_base", "t_crs", "s_base", "s_crs"]:
        p = results_dir / f"model_check_{m}" / "train_summary.json"
        if p.exists():
            out[m] = read_json(p)
    return out


def write_csv(path: Path, rows: List[Dict[str, Any]], headers: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in headers})


def main() -> None:
    ap = argparse.ArgumentParser(description="Collect data for figures/tables.")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--out-dir", default="results/plot_data")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    missing: Dict[str, Any] = {}

    model_summaries = find_model_summaries(results_dir)

    # Fig2 + Table1 + Fig5 + Fig6 from model_check summaries
    fig2_rows, table1_rows, fig5_rows, fig6_rows = [], [], [], []
    for model, s in model_summaries.items():
        ev = s.get("eval", {})
        pf = s.get("performance", {})
        lat = pf.get("inference_latency_ms", {}) if isinstance(pf.get("inference_latency_ms", {}), dict) else {}

        fig2_rows.append({
            "model": model,
            "accuracy": ev.get("accuracy"),
            "f1": ev.get("f1"),
            "auc_roc": ev.get("auc_roc"),
            "accuracy_std": "", "f1_std": "", "auc_roc_std": "",
        })
        table1_rows.append({
            "model": model,
            "dataset": s.get("config", {}).get("dataset"),
            "accuracy": ev.get("accuracy"), "f1": ev.get("f1"), "auc_roc": ev.get("auc_roc"),
            "peak_memory_MB": pf.get("peak_memory_MB"),
            "throughput_samples_per_s": pf.get("throughput_samples_per_s"),
            "latency_p50_ms": lat.get("p50"), "latency_p95_ms": lat.get("p95"),
        })
        fig5_rows.append({"model": model, "sparsity": pf.get("sparsity"), "accuracy": ev.get("accuracy")})
        fig6_rows.append({
            "model": model,
            "memory_mb": pf.get("peak_memory_MB"),
            "latency_p50_ms": lat.get("p50"),
            "throughput_sps": pf.get("throughput_samples_per_s"),
            "power_w": "",  # not found in current summaries
        })

    write_csv(out_dir / "fig2_dataset_comparison.csv", fig2_rows,
              ["model", "accuracy", "f1", "auc_roc", "accuracy_std", "f1_std", "auc_roc_std"])
    write_csv(out_dir / "table1_model_comparison.csv", table1_rows,
              ["model", "dataset", "accuracy", "f1", "auc_roc", "peak_memory_MB", "throughput_samples_per_s", "latency_p50_ms", "latency_p95_ms"])
    write_csv(out_dir / "fig5_pareto.csv", fig5_rows, ["model", "sparsity", "accuracy"])
    write_csv(out_dir / "fig6_hardware_radar.csv", fig6_rows,
              ["model", "memory_mb", "latency_p50_ms", "throughput_sps", "power_w"])

    # Fig4 from ablation results
    ablation_path = results_dir / "ablation_framework_test" / "ablation_results.json"
    if not ablation_path.exists():
        ablation_path = results_dir / "ablation_framework_smoke" / "ablation_results.json"
    fig4_rows = []
    if ablation_path.exists():
        ar = read_json(ablation_path)
        for v in ar.get("variants", []):
            fig4_rows.append({
                "variant_id": v.get("variant_id"),
                "variant_name": v.get("variant_name"),
                "delta_vs_baseline": v.get("delta_vs_baseline"),
                "avg_score": v.get("avg_score"),
            })
        write_csv(out_dir / "fig4_ablation_waterfall.csv", fig4_rows,
                  ["variant_id", "variant_name", "delta_vs_baseline", "avg_score"])
    else:
        missing["Fig4"] = "ablation_results.json not found"

    # Missing items report
    if not model_summaries:
        missing["Fig2/Table1/Fig5/Fig6"] = "results/model_check_{t_base,t_crs,s_base,s_crs}/train_summary.json missing"
    missing["Fig3"] = "10-cycle long-term memory curve data missing (experiments/long_term_memory_test.py is TODO)"
    missing["Fig7"] = "CRS stage attention heatmaps not found in results"
    missing["Table2"] = "SOTA benchmark comparison source file missing"

    (out_dir / "missing_items.json").write_text(json.dumps(missing, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"DONE. files saved to: {out_dir}")


if __name__ == "__main__":
    main()

