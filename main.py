import os
import json

from demo_data import EXAMPLE_INPUT, EXAMPLE_TRUE, EXAMPLE_PRED
from core.relations import build_relation_graph
from core.objects import extract_objects
from core.diagnostics import diagnose_failure
from core.visualization import save_render

OUTPUT_DIR = "outputs/single_run"

def save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def print_failure_tree(report):
    print("\nRoot Cause Analysis (Failure Hierarchy Tree):")
    global_fails = report.get("global_failures", [])
    matching = report.get("matching", {})
    color_dist = report.get("color_distribution", {})
    mapping = color_dist.get("color_shift_mapping", {})
    
    tree_empty = True
    
    # 1. Structural Failures (Primary)
    if "missing_object" in global_fails or "spurious_object" in global_fails:
        tree_empty = False
        if "missing_object" in global_fails:
            print(f"  ├── Primary Structural Failure: missing_object (Target objects missing: {matching.get('unmatched_true', [])})")
        if "spurious_object" in global_fails:
            print(f"  ├── Primary Structural Failure: spurious_object (Hallucinated objects: {matching.get('unmatched_pred', [])})")
            
        # Symptoms of structural failures
        if "global_color_distribution_shift" in global_fails:
            print(f"  │   └── Secondary Symptom: global_color_distribution_shift")
            if not mapping:
                print(f"  │       └── (No color_shift_mapping: color shift is merely a symptom of missing/spurious objects)")
            else:
                print(f"  │       └── (Also accompanied by explicit color mapping errors: {mapping})")
                
    # 2. Pure Color Mapping Failures (If no structural failures but colors are wrong)
    elif "global_color_distribution_shift" in global_fails:
        tree_empty = False
        print(f"  ├── Primary Rule Failure: incorrect_color_mapping")
        print(f"  │   └── Secondary Symptom: global_color_distribution_shift (Shift mapping: {mapping})")
        
    # 3. Pair-level Transformation Failures
    pair_diags = report.get("pair_diagnostics", [])
    for pd in pair_diags:
        failures = [f for f in pd.get("pair_failures", []) if f.get("type") != "no_pair_level_failure"]
        if failures:
            tree_empty = False
            print(f"  ├── Transformation Failure: pred#{pd['pred_obj_id']} vs true#{pd['true_obj_id']}")
            for i, f in enumerate(failures):
                connector = "└──" if i == len(failures) - 1 else "├──"
                mag_str = f" (magnitude: {f.get('magnitude'):.1f} {f.get('unit', '')})".strip() if "magnitude" in f else ""
                print(f"  │   {connector} {f['type']}{mag_str}")
                
    if tree_empty:
        print("  └── No failures detected (Perfect match!)")

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    input_objects = extract_objects(EXAMPLE_INPUT)
    true_objects = extract_objects(EXAMPLE_TRUE)
    pred_objects = extract_objects(EXAMPLE_PRED)

    input_relations = build_relation_graph(input_objects)
    report = diagnose_failure(EXAMPLE_INPUT, EXAMPLE_PRED, EXAMPLE_TRUE)

    save_json([o.to_dict() for o in input_objects], os.path.join(OUTPUT_DIR, "input_objects.json"))
    save_json([o.to_dict() for o in true_objects], os.path.join(OUTPUT_DIR, "true_objects.json"))
    save_json([o.to_dict() for o in pred_objects], os.path.join(OUTPUT_DIR, "pred_objects.json"))
    save_json(input_relations, os.path.join(OUTPUT_DIR, "input_relations.json"))
    save_json(report, os.path.join(OUTPUT_DIR, "diagnostic_report.json"))

    save_render(EXAMPLE_INPUT, os.path.join(OUTPUT_DIR, "input_grid.png"), with_overlay=True)
    save_render(EXAMPLE_TRUE, os.path.join(OUTPUT_DIR, "true_grid.png"), with_overlay=True)
    save_render(EXAMPLE_PRED, os.path.join(OUTPUT_DIR, "pred_grid.png"), with_overlay=True)

    print("Saved outputs:")
    print(f"- {OUTPUT_DIR}/input_objects.json")
    print(f"- {OUTPUT_DIR}/true_objects.json")
    print(f"- {OUTPUT_DIR}/pred_objects.json")
    print(f"- {OUTPUT_DIR}/input_relations.json")
    print(f"- {OUTPUT_DIR}/diagnostic_report.json")
    print(f"- {OUTPUT_DIR}/input_grid.png")
    print(f"- {OUTPUT_DIR}/true_grid.png")
    print(f"- {OUTPUT_DIR}/pred_grid.png")

    print_failure_tree(report)

if __name__ == "__main__":
    main()
