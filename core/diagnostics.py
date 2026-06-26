from typing import Dict, Any, List
from collections import Counter

from core.objects import extract_objects, ArcObject
from core.matching import greedy_match_objects

def extract_transformations(input_grid: List[List[int]], output_grid: List[List[int]]) -> Dict[str, Any]:
    """Extracts transformation rules by matching input objects directly to output objects."""
    input_objs = extract_objects(input_grid)
    output_objs = extract_objects(output_grid)
    
    matching = greedy_match_objects(input_objs, output_objs)
    transformations = []
    
    for m in matching["matches"]:
        trans = {}
        if m["pred_color"] != m["true_color"]:
            trans["color_change"] = f"{m['pred_color']} -> {m['true_color']}"
        if m["pred_area"] != m["true_area"]:
            trans["area_change"] = f"{m['pred_area']} -> {m['true_area']}"
        if m["shape_relation"] != "exact_shape":
            trans["shape_change"] = m["shape_relation"]
        
        pr, pc = m["pred_centroid"]
        tr, tc = m["true_centroid"]
        if (pr, pc) != (tr, tc):
            trans["translation"] = (round(tr - pr, 2), round(tc - pc, 2))
            
        transformations.append({
            "input_obj_id": m["pred_obj_id"],
            "output_obj_id": m["true_obj_id"],
            "transformations": trans if trans else "unchanged"
        })
        
    return {
        "matched_transformations": transformations,
        "unmatched_input_objects": matching["unmatched_pred"],
        "unmatched_output_objects": matching["unmatched_true"]
    }

def compare_color_distributions(pred_objs: List[ArcObject], true_objs: List[ArcObject], matching: Dict[str, Any] = None) -> Dict[str, Any]:
    pred_counter = Counter([o.color for o in pred_objs])
    true_counter = Counter([o.color for o in true_objs])
    
    color_shift_mapping = {}
    if matching and pred_counter != true_counter:
        # Extract explicit color mapping from matched objects
        for m in matching.get("matches", []):
            if m["pred_color"] != m["true_color"]:
                # Map true target color to what the model predicted
                key = str(m["true_color"])
                if key not in color_shift_mapping:
                    color_shift_mapping[key] = []
                color_shift_mapping[key].append(m["pred_color"])
                
        # Condense the lists into the most common predicted color for each true color
        for k, v in color_shift_mapping.items():
            color_shift_mapping[k] = Counter(v).most_common(1)[0][0]

    return {
        "pred_colors": dict(pred_counter),
        "true_colors": dict(true_counter),
        "match": pred_counter == true_counter,
        "color_shift_mapping": color_shift_mapping
    }

def classify_pair_failure(match_info: Dict[str, Any]) -> List[Dict[str, Any]]:
    failures = []

    if match_info["pred_color"] != match_info["true_color"]:
        failures.append({"type": "color_mismatch", "magnitude": 1.0})

    if match_info["pred_area"] != match_info["true_area"]:
        failures.append({
            "type": "area_mismatch",
            "magnitude": abs(match_info["pred_area"] - match_info["true_area"]),
            "expected_area": match_info["true_area"]
        })

    shape_relation = match_info["shape_relation"]
    if shape_relation == "different_shape":
        failures.append({"type": "shape_mismatch", "magnitude": 1.0})
    elif shape_relation.startswith("transformed_via_") or shape_relation == "same_canonical_shape_different_variant":
        failures.append({"type": "orientation_or_reflection_mismatch", "magnitude": 1.0})

    pr, pc = match_info["pred_centroid"]
    tr, tc = match_info["true_centroid"]
    if (pr, pc) != (tr, tc):
        dist = abs(pr - tr) + abs(pc - tc)
        failures.append({
            "type": "position_or_translation_mismatch",
            "magnitude": dist,
            "unit": "pixels"
        })

    return failures

def diagnose_failure(input_grid, pred_grid, true_grid) -> Dict[str, Any]:
    input_objs = extract_objects(input_grid)
    pred_objs = extract_objects(pred_grid)
    true_objs = extract_objects(true_grid)

    matching = greedy_match_objects(pred_objs, true_objs)
    color_dist = compare_color_distributions(pred_objs, true_objs, matching)

    pair_diagnostics = []
    global_failures = []

    for m in matching["matches"]:
        failures = classify_pair_failure(m)
        pair_diagnostics.append({
            **m,
            "pair_failures": failures if failures else [{"type": "no_pair_level_failure"}]
        })

    if matching["unmatched_true"]:
        global_failures.append("missing_object")
    if matching["unmatched_pred"]:
        global_failures.append("spurious_object")
    if not color_dist["match"]:
        global_failures.append("global_color_distribution_shift")

    if not global_failures and all(
        len(pd["pair_failures"]) == 1 and pd["pair_failures"][0]["type"] == "no_pair_level_failure" for pd in pair_diagnostics
    ):
        global_failures.append("prediction_matches_target_under_current_diagnostics")

    return {
        "object_counts": {
            "input": len(input_objs),
            "pred": len(pred_objs),
            "true": len(true_objs)
        },
        "color_distribution": color_dist,
        "matching": matching,
        "pair_diagnostics": pair_diagnostics,
        "global_failures": global_failures,
        "input_objects": [o.to_dict() for o in input_objs],
        "pred_objects": [o.to_dict() for o in pred_objs],
        "true_objects": [o.to_dict() for o in true_objs],
    }
