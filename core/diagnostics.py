from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from collections import Counter

from core.objects import extract_objects, ArcObject
from core.matching import greedy_match_objects
from core.formatters import to_markdown_context, to_action_vector, to_json_api

class BaseFailureClassifier(ABC):
    @abstractmethod
    def classify(self, match_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        pass

class ColorMismatchClassifier(BaseFailureClassifier):
    def classify(self, match_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        if match_info["pred_color"] != match_info["true_color"]:
            return [{
                "type": "color_mismatch",
                "error_category": "perceptual",
                "magnitude": 1.0,
                "expected_color": match_info["true_color"],
                "predicted_color": match_info["pred_color"]
            }]
        return []

class AreaMismatchClassifier(BaseFailureClassifier):
    def classify(self, match_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        if match_info["pred_area"] != match_info["true_area"]:
            return [{
                "type": "area_mismatch",
                "error_category": "transformation",
                "magnitude": abs(match_info["pred_area"] - match_info["true_area"]),
                "expected_area": match_info["true_area"],
                "predicted_area": match_info["pred_area"]
            }]
        return []

class ShapeMismatchClassifier(BaseFailureClassifier):
    def classify(self, match_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        shape_relation = match_info["shape_relation"]
        if shape_relation == "different_shape":
            return [{
                "type": "shape_mismatch",
                "error_category": "perceptual",
                "magnitude": 1.0
            }]
        return []

class OrientationMismatchClassifier(BaseFailureClassifier):
    def classify(self, match_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        shape_relation = match_info["shape_relation"]
        if shape_relation.startswith("transformed_via_") or shape_relation == "same_canonical_shape_different_variant":
            variant_transform = shape_relation.replace("transformed_via_", "") if shape_relation.startswith("transformed_via_") else "variant_mismatch"
            return [{
                "type": "orientation_or_reflection_mismatch",
                "error_category": "transformation",
                "magnitude": 1.0,
                "shape_relation": shape_relation,
                "suggested_correction": f"apply_inverse_transform_{variant_transform}"
            }]
        return []

class PositionMismatchClassifier(BaseFailureClassifier):
    def classify(self, match_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        pr, pc = match_info["pred_centroid"]
        tr, tc = match_info["true_centroid"]
        if (pr, pc) != (tr, tc):
            delta_row = round(tr - pr, 3)
            delta_col = round(tc - pc, 3)
            dist = abs(pr - tr) + abs(pc - tc)
            return [{
                "type": "position_or_translation_mismatch",
                "error_category": "transformation",
                "magnitude": round(dist, 3),
                "delta_row": delta_row,
                "delta_col": delta_col,
                "unit": "pixels"
            }]
        return []

DEFAULT_CLASSIFIERS: List[BaseFailureClassifier] = [
    ColorMismatchClassifier(),
    AreaMismatchClassifier(),
    ShapeMismatchClassifier(),
    OrientationMismatchClassifier(),
    PositionMismatchClassifier(),
]

def classify_pair_failure(match_info: Dict[str, Any], classifiers: Optional[List[BaseFailureClassifier]] = None) -> List[Dict[str, Any]]:
    if classifiers is None:
        classifiers = DEFAULT_CLASSIFIERS
    failures = []
    for clf in classifiers:
        failures.extend(clf.classify(match_info))
    return failures

def extract_transformations(input_grid: List[List[int]], output_grid: List[List[int]]) -> Dict[str, Any]:
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
        for m in matching.get("matches", []):
            if m["pred_color"] != m["true_color"]:
                key = str(m["true_color"])
                if key not in color_shift_mapping:
                    color_shift_mapping[key] = []
                color_shift_mapping[key].append(m["pred_color"])
                
        for k, v in color_shift_mapping.items():
            color_shift_mapping[k] = Counter(v).most_common(1)[0][0]

    return {
        "pred_colors": dict(pred_counter),
        "true_colors": dict(true_counter),
        "match": pred_counter == true_counter,
        "color_shift_mapping": color_shift_mapping
    }

def generate_textual_diagnosis(report: Dict[str, Any]) -> str:
    lines = []
    lines.append("=== ARC Diagnostic Card ===")
    lines.append(f"Global Failures: {', '.join(report.get('global_failures', [])) or 'None'}")
    
    pair_diags = report.get("pair_diagnostics", [])
    if not pair_diags:
        lines.append("No matched object pairs.")
    else:
        for pd in pair_diags:
            pred_id = pd.get("pred_obj_id")
            true_id = pd.get("true_obj_id")
            failures = pd.get("pair_failures", [])
            lines.append(f"\n- Object Pair (Pred {pred_id} <-> True {true_id}):")
            for f in failures:
                ftype = f.get("type")
                if ftype == "no_pair_level_failure":
                    lines.append("  * Exact match (No failure)")
                elif ftype == "position_or_translation_mismatch":
                    lines.append(f"  * Position Error: Shift by delta_row={f.get('delta_row')}, delta_col={f.get('delta_col')} (distance={f.get('magnitude')} px)")
                elif ftype == "orientation_or_reflection_mismatch":
                    lines.append(f"  * Orientation Error: Shape transformed via {f.get('shape_relation')}. Fix: {f.get('suggested_correction')}")
                elif ftype == "color_mismatch":
                    lines.append(f"  * Color Error: Predicted color {f.get('predicted_color')}, expected {f.get('expected_color')}")
                elif ftype == "area_mismatch":
                    lines.append(f"  * Area Error: Predicted area {f.get('predicted_area')}, expected {f.get('expected_area')}")
    return "\n".join(lines)

class DiagnosticEngine:
    def __init__(self, classifiers: Optional[List[BaseFailureClassifier]] = None):
        self.classifiers = classifiers or DEFAULT_CLASSIFIERS

    def run(self, input_grid: List[List[int]], pred_grid: List[List[int]], true_grid: List[List[int]]) -> Dict[str, Any]:
        report = diagnose_failure(input_grid, pred_grid, true_grid, classifiers=self.classifiers)
        return report

def diagnose_failure(input_grid, pred_grid, true_grid, classifiers: Optional[List[BaseFailureClassifier]] = None) -> Dict[str, Any]:
    input_objs = extract_objects(input_grid)
    pred_objs = extract_objects(pred_grid)
    true_objs = extract_objects(true_grid)

    matching = greedy_match_objects(pred_objs, true_objs)
    color_dist = compare_color_distributions(pred_objs, true_objs, matching)

    pair_diagnostics = []
    global_failures = []

    for m in matching["matches"]:
        failures = classify_pair_failure(m, classifiers=classifiers)
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

    report = {
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
    report["textual_diagnosis"] = generate_textual_diagnosis(report)
    report["markdown_card"] = to_markdown_context(report)
    report["action_vector"] = to_action_vector(report)
    report["json_api"] = to_json_api(report)
    return report
