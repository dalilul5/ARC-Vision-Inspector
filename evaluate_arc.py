import os
import json

from core.arc_loader import load_arc_tasks_from_dir, iter_all_pairs
from core.baseline_predictor import simple_predict
from core.diagnostics import diagnose_failure, extract_transformations
from core.visualization import save_render
from core.reporting import (
    save_json,
    save_csv,
    summarize_global_failures,
    flatten_report_metadata,
    aggregate_hierarchical_failures,
    print_executive_summary
)

DATA_DIR = "data/sample_tasks"
OUTPUT_DIR = "outputs/batch_run"
PREDICTOR_MODE = "identity"   # options: identity / shift_right

def check_rule_consistency(train_transformations):
    """Checks if the observed transformations are consistent across all train pairs."""
    if not train_transformations:
        return "no_train_data"
        
    # Simplified consistency check: collect unique non-empty transformations
    all_rules = []
    for trans_data in train_transformations:
        pair_rules = set()
        for t in trans_data["matched_transformations"]:
            if t["transformations"] != "unchanged":
                # Convert dict to a sorted tuple of items to make it hashable
                rule_sig = tuple(sorted(t["transformations"].items()))
                pair_rules.add(rule_sig)
        all_rules.append(pair_rules)
    
    if not all_rules:
        return "no_rules_found"
        
    # Check if the set of transformations is exactly the same across all train examples
    first_rules = all_rules[0]
    is_consistent = all(rules == first_rules for rules in all_rules)
    
    return "consistent" if is_consistent else "inconsistent"

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    tasks = load_arc_tasks_from_dir(DATA_DIR)
    sample_rows = []
    sample_reports = []

    for task in tasks:
        task_name = task.get("_task_name", "unknown").replace(".json", "")
        print(f"Analyzing Task: {task_name}")
        
        # 1. Multi-shot Rule Extraction (Train Pairs)
        train_pairs = task.get("train", [])
        train_transformations = []
        for idx, pair in enumerate(train_pairs):
            trans = extract_transformations(pair["input"], pair["output"])
            train_transformations.append(trans)
            
        consistency_status = check_rule_consistency(train_transformations)
        print(f"  -> Rule consistency across train pairs: {consistency_status}")
        
        # 2. Evaluation (All Pairs)
        for pair in iter_all_pairs(task):
            split = pair["split"]
            idx = pair["index"]

            input_grid = pair["input"]
            true_grid = pair["output"]
            pred_grid = simple_predict(input_grid, mode=PREDICTOR_MODE)

            report = diagnose_failure(input_grid, pred_grid, true_grid)
            # Inject multi-shot context into the test report
            if split == "test":
                report["multi_shot_context"] = {
                    "train_consistency": consistency_status,
                    "train_transformations_summary": [
                        {"pair": i, "rules": len([t for t in tr["matched_transformations"] if t["transformations"] != "unchanged"])}
                        for i, tr in enumerate(train_transformations)
                    ]
                }
            
            sample_reports.append(report)
            sample_rows.append(flatten_report_metadata(pair, report))

            sample_dir = os.path.join(OUTPUT_DIR, f"{task_name}_{split}_{idx}")
            os.makedirs(sample_dir, exist_ok=True)

            save_json(report, os.path.join(sample_dir, "diagnostic_report.json"))
            save_render(input_grid, os.path.join(sample_dir, "input.png"), with_overlay=True)
            save_render(pred_grid, os.path.join(sample_dir, "pred.png"), with_overlay=True)
            save_render(true_grid, os.path.join(sample_dir, "true.png"), with_overlay=True)

    # --- Executive Summary V2.3 ---
    hierarchy_stats = aggregate_hierarchical_failures(sample_reports, sample_rows)
    print_executive_summary(len(sample_reports), hierarchy_stats)
    
    # Save the new hierarchical summary
    save_json(hierarchy_stats, os.path.join(OUTPUT_DIR, "batch_executive_summary.json"))

    global_summary = {
        "predictor_mode": PREDICTOR_MODE,
        "num_tasks": len(tasks),
        "num_samples": len(sample_reports),
        "global_failure_counts": summarize_global_failures(sample_reports)
    }

    save_json(global_summary, os.path.join(OUTPUT_DIR, "summary.json"))
    save_csv(sample_rows, os.path.join(OUTPUT_DIR, "summary.csv"))

    print(f"Saved batch outputs and executive summary to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
