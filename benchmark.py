import os
import json
from collections import defaultdict
from typing import Dict, Any, List

from core.diagnostics import diagnose_failure
from core.reporting import save_json

BENCHMARK_DIR = "data/synthetic_benchmark"
OUTPUT_PATH = "outputs/benchmark_summary.json"

def run_benchmark() -> Dict[str, Any]:
    if not os.path.exists(BENCHMARK_DIR):
        from data.synthetic_generator import generate_synthetic_benchmark
        generate_synthetic_benchmark()

    fnames = sorted([f for f in os.listdir(BENCHMARK_DIR) if f.endswith(".json")])
    
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)

    all_failure_types = {
        "position_or_translation_mismatch",
        "orientation_or_reflection_mismatch",
        "color_mismatch",
        "missing_object",
        "spurious_object"
    }

    results = []

    for fname in fnames:
        filepath = os.path.join(BENCHMARK_DIR, fname)
        with open(filepath, "r", encoding="utf-8") as f:
            case = json.load(f)

        report = diagnose_failure(case["input"], case["pred"], case["true"])
        
        detected_failures = set(report.get("global_failures", []))
        for pd in report.get("pair_diagnostics", []):
            for fail in pd.get("pair_failures", []):
                if fail.get("type") != "no_pair_level_failure":
                    detected_failures.add(fail.get("type"))

        expected_failures = set(case.get("expected_failures", []))

        for ftype in all_failure_types:
            if ftype in expected_failures and ftype in detected_failures:
                tp[ftype] += 1
            elif ftype in detected_failures and ftype not in expected_failures:
                fp[ftype] += 1
            elif ftype in expected_failures and ftype not in detected_failures:
                fn[ftype] += 1

        results.append({
            "case": case["name"],
            "expected": list(expected_failures),
            "detected": list(detected_failures),
            "match": expected_failures.issubset(detected_failures) or (not expected_failures and "prediction_matches_target_under_current_diagnostics" in detected_failures)
        })

    metrics = {}
    total_tp = sum(tp.values())
    total_fp = sum(fp.values())
    total_fn = sum(fn.values())

    for ftype in sorted(all_failure_types):
        p = tp[ftype] / (tp[ftype] + fp[ftype]) if (tp[ftype] + fp[ftype]) > 0 else 1.0
        r = tp[ftype] / (tp[ftype] + fn[ftype]) if (tp[ftype] + fn[ftype]) > 0 else 1.0
        f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0
        metrics[ftype] = {
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1_score": round(f1, 4),
            "tp": tp[ftype],
            "fp": fp[ftype],
            "fn": fn[ftype]
        }

    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 1.0
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 1.0
    overall_f1 = (2 * overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0.0

    summary = {
        "total_benchmark_cases": len(results),
        "overall_metrics": {
            "precision": round(overall_precision, 4),
            "recall": round(overall_recall, 4),
            "f1_score": round(overall_f1, 4)
        },
        "per_category_metrics": metrics,
        "case_results": results
    }

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    save_json(summary, OUTPUT_PATH)

    print("==========================================")
    print("      SYNTHETIC BENCHMARK EVALUATION      ")
    print("==========================================")
    print(f"Total Test Cases: {len(results)}")
    print(f"Overall Precision: {overall_precision:.2%}")
    print(f"Overall Recall   : {overall_recall:.2%}")
    print(f"Overall F1 Score : {overall_f1:.2%}")
    print("------------------------------------------")
    print("Per-Category Metrics:")
    for ftype, m in metrics.items():
        print(f"  * {ftype:<35}: F1={m['f1_score']:.2%} (P={m['precision']:.2%}, R={m['recall']:.2%})")
    print("==========================================")

    return summary

if __name__ == "__main__":
    run_benchmark()
