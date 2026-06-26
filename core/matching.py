from typing import List, Dict, Any, Tuple
import numpy as np
from scipy.optimize import linear_sum_assignment
from core.objects import ArcObject

def detect_shape_relation(pred_obj: ArcObject, true_obj: ArcObject) -> str:
    if pred_obj.shape_mask == true_obj.shape_mask:
        return "exact_shape"
    
    # Check if any variant of the prediction matches the true shape mask
    for variant_name, variant_sig in pred_obj.variant_signatures.items():
        if variant_sig == true_obj.shape_mask:
            return f"transformed_via_{variant_name}"
            
    if pred_obj.canonical_shape == true_obj.canonical_shape:
        return "same_canonical_shape_different_variant"
        
    return "different_shape"

def object_match_score(pred_obj: ArcObject, true_obj: ArcObject) -> float:
    score = 0.0

    if pred_obj.color == true_obj.color:
        score += 3.0

    if pred_obj.area == true_obj.area:
        score += 2.0

    if pred_obj.canonical_shape == true_obj.canonical_shape:
        score += 4.0

    pr, pc = pred_obj.centroid
    tr, tc = true_obj.centroid
    dist = abs(pr - tr) + abs(pc - tc)
    score += max(0.0, 2.0 - 0.2 * dist)

    return round(score, 3)

def greedy_match_objects(pred_objs: List[ArcObject], true_objs: List[ArcObject]) -> Dict[str, Any]:
    # Keeping the original function name for backwards compatibility, but implementing optimal matching
    pairs = []
    
    if not pred_objs or not true_objs:
        return {
            "matches": [],
            "unmatched_pred": [po.obj_id for po in pred_objs],
            "unmatched_true": [to.obj_id for to in true_objs],
        }

    cost_matrix = np.zeros((len(pred_objs), len(true_objs)))
    for i, po in enumerate(pred_objs):
        for j, to in enumerate(true_objs):
            # We negate the score because linear_sum_assignment finds the minimum cost
            cost_matrix[i, j] = -object_match_score(po, to)

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    used_pred = set()
    used_true = set()

    for i, j in zip(row_ind, col_ind):
        score = -cost_matrix[i, j]
        # Only accept matches with a positive score (e.g., some baseline similarity)
        # If score is 0, they are completely unrelated.
        if score > 0:
            used_pred.add(i)
            used_true.add(j)
            po = pred_objs[i]
            to = true_objs[j]

            pairs.append({
                "pred_obj_id": po.obj_id,
                "true_obj_id": to.obj_id,
                "score": score,
                "pred_color": po.color,
                "true_color": to.color,
                "pred_area": po.area,
                "true_area": to.area,
                "pred_centroid": po.centroid,
                "true_centroid": to.centroid,
                "shape_relation": detect_shape_relation(po, to),
            })

    unmatched_pred = [pred_objs[i].obj_id for i in range(len(pred_objs)) if i not in used_pred]
    unmatched_true = [true_objs[j].obj_id for j in range(len(true_objs)) if j not in used_true]

    return {
        "matches": pairs,
        "unmatched_pred": unmatched_pred,
        "unmatched_true": unmatched_true,
    }
