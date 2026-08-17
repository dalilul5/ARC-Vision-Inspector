import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any

from core.diagnostics import diagnose_failure
from core.reporting import save_json

ROOT = Path(__file__).resolve().parent
BENCHMARK_DIR = ROOT / "data" / "synthetic_benchmark"
OUTPUT_PATH = ROOT / "outputs" / "benchmark_summary.json"
FAILURE_TYPES = {
    "position_or_translation_mismatch",
    "orientation_or_reflection_mismatch",
    "color_mismatch",
    "missing_object",
    "spurious_object",
}


def _metric(tp, fp, fn):
    predicted_support = tp + fp
    positive_support = tp + fn
    precision = tp / predicted_support if predicted_support else None
    recall = tp / positive_support if positive_support else None
    f1 = 2 * precision * recall / (precision + recall) if precision is not None and recall is not None and precision + recall else None
    return {
        "precision": None if precision is None else round(precision, 4),
        "recall": None if recall is None else round(recall, 4),
        "f1_score": None if f1 is None else round(f1, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "positive_support": positive_support,
        "predicted_support": predicted_support,
    }


def run_benchmark() -> Dict[str, Any]:
    if not BENCHMARK_DIR.exists():
        from data.synthetic_generator import generate_synthetic_benchmark
        generate_synthetic_benchmark()
    tp, fp, fn, results = defaultdict(int), defaultdict(int), defaultdict(int), []
    for path in sorted(BENCHMARK_DIR.glob("*.json")):
        case = json.loads(path.read_text(encoding="utf-8"))
        report = diagnose_failure(case["input"], case["pred"], case["true"])
        detected = set(report.get("global_failures", []))
        for pair in report.get("pair_diagnostics", []):
            detected.update(item["type"] for item in pair.get("pair_failures", []) if item.get("type") != "no_pair_level_failure")
        detected &= FAILURE_TYPES
        expected = set(case.get("expected_failures", []))
        for label in FAILURE_TYPES:
            tp[label] += int(label in expected and label in detected)
            fp[label] += int(label not in expected and label in detected)
            fn[label] += int(label in expected and label not in detected)
        results.append({
            "case": case["name"],
            "expected": sorted(expected),
            "detected": sorted(detected),
            "exact_label_match": expected == detected,
            "all_expected_detected": expected.issubset(detected),
        })
    per_category = {label: _metric(tp[label], fp[label], fn[label]) for label in sorted(FAILURE_TYPES)}
    total_tp, total_fp, total_fn = sum(tp.values()), sum(fp.values()), sum(fn.values())
    valid_f1 = [m["f1_score"] for m in per_category.values() if m["f1_score"] is not None]
    summary = {
        "total_benchmark_cases": len(results),
        "exact_case_accuracy": round(sum(r["exact_label_match"] for r in results) / len(results), 4) if results else 0.0,
        "micro_metrics": _metric(total_tp, total_fp, total_fn),
        "macro_f1": round(sum(valid_f1) / len(valid_f1), 4) if valid_f1 else None,
        "per_category_metrics": per_category,
        "case_results": results,
    }
    save_json(summary, str(OUTPUT_PATH))
    return summary


if __name__ == "__main__":
    print(json.dumps(run_benchmark(), indent=2))
