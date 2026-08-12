import os
import json
from typing import Dict, Any, List
import itertools

from core.diagnostics import diagnose_failure
from core.objects import extract_objects
from core.matching import object_match_score, greedy_match_objects
import core.matching as matching_module

BENCHMARK_DIR = "data/synthetic_benchmark"
CONFIG_PATH = "weights.json"

def load_benchmark_cases() -> List[Dict[str, Any]]:
    cases = []
    if not os.path.exists(BENCHMARK_DIR):
        from data.synthetic_generator import generate_synthetic_benchmark
        return generate_synthetic_benchmark()

    for fname in sorted(os.listdir(BENCHMARK_DIR)):
        if fname.endswith(".json"):
            with open(os.path.join(BENCHMARK_DIR, fname), "r", encoding="utf-8") as f:
                cases.append(json.load(f))
    return cases

def evaluate_weights_on_cases(cases: List[Dict[str, Any]], w_color, w_area, w_shape, w_centroid) -> float:
    # Custom match score monkey patch or wrapper
    def custom_match_score(pred_obj, true_obj) -> float:
        score = 0.0
        if pred_obj.color == true_obj.color:
            score += w_color
        if pred_obj.area == true_obj.area:
            score += w_area
        if pred_obj.canonical_shape == true_obj.canonical_shape:
            score += w_shape
        pr, pc = pred_obj.centroid
        tr, tc = true_obj.centroid
        dist = abs(pr - tr) + abs(pc - tc)
        score += max(0.0, w_centroid - 0.2 * dist)
        return round(score, 3)

    orig_score_fn = matching_module.object_match_score
    matching_module.object_match_score = custom_match_score

    correct_matches = 0
    total_expected = 0

    try:
        for case in cases:
            report = diagnose_failure(case["input"], case["pred"], case["true"])
            actual_failures = report.get("global_failures", [])
            for pd in report.get("pair_diagnostics", []):
                for f in pd.get("pair_failures", []):
                    actual_failures.append(f.get("type"))
            
            exp_failures = case.get("expected_failures", [])
            total_expected += len(exp_failures)
            for exp in exp_failures:
                if exp in actual_failures or (exp == "exact" and len(actual_failures) == 0):
                    correct_matches += 1
    finally:
        matching_module.object_match_score = orig_score_fn

    return (correct_matches / total_expected) if total_expected > 0 else 0.0

def optimize_matching_weights() -> Dict[str, float]:
    cases = load_benchmark_cases()
    
    color_grid = [2.0, 3.0, 4.0]
    area_grid = [1.0, 2.0, 3.0]
    shape_grid = [3.0, 4.0, 5.0]
    centroid_grid = [1.0, 2.0, 3.0]

    best_score = -1.0
    best_weights = {
        "weight_color": 3.0,
        "weight_area": 2.0,
        "weight_shape": 4.0,
        "weight_centroid": 2.0
    }

    print("Running Grid Search optimization over matching weights...")
    for wc, wa, ws, wct in itertools.product(color_grid, area_grid, shape_grid, centroid_grid):
        score = evaluate_weights_on_cases(cases, wc, wa, ws, wct)
        if score > best_score:
            best_score = score
            best_weights = {
                "weight_color": wc,
                "weight_area": wa,
                "weight_shape": ws,
                "weight_centroid": wct,
                "benchmark_accuracy": round(score, 4)
            }

    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(best_weights, f, indent=2)

    print(f"Optimal weights found (Benchmark Acc: {best_weights['benchmark_accuracy']:.2%}):")
    print(json.dumps(best_weights, indent=2))
    print(f"Saved to {CONFIG_PATH}")
    return best_weights

if __name__ == "__main__":
    optimize_matching_weights()
