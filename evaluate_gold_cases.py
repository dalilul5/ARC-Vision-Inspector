import os
import json
from typing import Dict, Any, List

from core.diagnostics import diagnose_failure
from core.reporting import save_json, save_csv

GOLD_DIR = "data/gold_cases"
OUTPUT_DIR = "outputs/gold_eval"

def load_gold_cases(directory: str) -> List[Dict[str, Any]]:
    cases = []
    if not os.path.exists(directory):
        return cases

    for fname in sorted(os.listdir(directory)):
        if fname.endswith(".json"):
            with open(os.path.join(directory, fname), "r", encoding="utf-8") as f:
                case = json.load(f)
                case["_file"] = fname
                cases.append(case)
    return cases

def flatten_pair_failures(report: Dict[str, Any]) -> List[str]:
    labels = []
    for pd in report.get("pair_diagnostics", []):
        labels.extend(pd.get("pair_failures", []))
    return sorted(set(labels))

def evaluate_case(case: Dict[str, Any]) -> Dict[str, Any]:
    report = diagnose_failure(case["input"], case["pred"], case["true"])

    actual_global = sorted(report.get("global_failures", []))
    actual_pair = flatten_pair_failures(report)

    expected_global = sorted(case.get("expected_global_failures", []))
    expected_pair = sorted(case.get("expected_pair_failures", []))

    global_exact_match = actual_global == expected_global

    pair_contains_expected = all(label in actual_pair for label in expected_pair)
    pair_exact_match = actual_pair == expected_pair

    return {
        "name": case["name"],
        "file": case["_file"],
        "description": case.get("description", ""),
        "expected_global_failures": expected_global,
        "actual_global_failures": actual_global,
        "global_exact_match": global_exact_match,
        "expected_pair_failures": expected_pair,
        "actual_pair_failures": actual_pair,
        "pair_contains_expected": pair_contains_expected,
        "pair_exact_match": pair_exact_match,
        "pass_strict": global_exact_match and pair_exact_match,
        "pass_relaxed": global_exact_match and pair_contains_expected,
        "notes": case.get("notes", "")
    }

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    cases = load_gold_cases(GOLD_DIR)
    results = [evaluate_case(case) for case in cases]

    summary_rows = []
    strict_pass = 0
    relaxed_pass = 0

    for r in results:
        summary_rows.append({
            "name": r["name"],
            "file": r["file"],
            "global_exact_match": r["global_exact_match"],
            "pair_contains_expected": r["pair_contains_expected"],
            "pair_exact_match": r["pair_exact_match"],
            "pass_strict": r["pass_strict"],
            "pass_relaxed": r["pass_relaxed"],
        })
        strict_pass += int(r["pass_strict"])
        relaxed_pass += int(r["pass_relaxed"])

    summary = {
        "num_cases": len(results),
        "strict_pass_count": strict_pass,
        "relaxed_pass_count": relaxed_pass,
        "strict_accuracy": (strict_pass / len(results)) if results else 0.0,
        "relaxed_accuracy": (relaxed_pass / len(results)) if results else 0.0,
        "results": results,
    }

    save_json(summary, os.path.join(OUTPUT_DIR, "gold_eval_summary.json"))
    save_csv(summary_rows, os.path.join(OUTPUT_DIR, "gold_eval_summary.csv"))

    print(f"Gold case evaluation completed on {len(results)} cases.")
    print(f"Strict accuracy : {summary['strict_accuracy']:.3f}")
    print(f"Relaxed accuracy: {summary['relaxed_accuracy']:.3f}")
    print(f"Saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
