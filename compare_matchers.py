import json
from pathlib import Path

from core.objects import extract_objects
from core.matching import match_objects_greedy, match_objects_hungarian

ROOT = Path(__file__).resolve().parent
BENCHMARK_DIR = ROOT / "data" / "synthetic_benchmark"
OUTPUT_PATH = ROOT / "outputs" / "matcher_comparison.json"


def _total_score(result):
    return round(sum(pair["score"] for pair in result["matches"]), 3)


def compare_matchers():
    rows = []
    for path in sorted(BENCHMARK_DIR.glob("*.json")):
        case = json.loads(path.read_text(encoding="utf-8"))
        pred = extract_objects(case["pred"])
        true = extract_objects(case["true"])
        greedy = match_objects_greedy(pred, true)
        hungarian = match_objects_hungarian(pred, true)
        rows.append({
            "case": case["name"],
            "greedy_total_score": _total_score(greedy),
            "hungarian_total_score": _total_score(hungarian),
            "score_gain": round(_total_score(hungarian) - _total_score(greedy), 3),
            "greedy_matches": [(m["pred_obj_id"], m["true_obj_id"]) for m in greedy["matches"]],
            "hungarian_matches": [(m["pred_obj_id"], m["true_obj_id"]) for m in hungarian["matches"]],
        })
    summary = {
        "cases": len(rows),
        "hungarian_higher_score_cases": sum(r["score_gain"] > 0 for r in rows),
        "results": rows,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps(compare_matchers(), indent=2))
