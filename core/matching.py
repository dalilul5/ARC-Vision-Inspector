import os
import json
import math
from typing import List, Dict, Any, Tuple
import numpy as np
from scipy.optimize import linear_sum_assignment
from core.objects import ArcObject

CONFIG_PATH = "weights.json"
DEFAULT_MIN_MATCH_THRESHOLD = 3.5

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

def get_active_weights():
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                return (
                    data.get("weight_color", 3.0),
                    data.get("weight_area", 2.0),
                    data.get("weight_shape", 4.0),
                    data.get("weight_centroid", 2.0),
                    data.get("min_match_threshold", DEFAULT_MIN_MATCH_THRESHOLD),
                )
        except Exception:
            pass
    return 3.0, 2.0, 4.0, 2.0, DEFAULT_MIN_MATCH_THRESHOLD

def object_match_score(pred_obj: ArcObject, true_obj: ArcObject) -> float:
    w_color, w_area, w_shape, w_centroid, _ = get_active_weights()
    score = 0.0

    # 1. Color matching (discrete)
    if pred_obj.color == true_obj.color:
        score += w_color

    # 2. Area matching (continuous similarity ratio)
    if pred_obj.area > 0 and true_obj.area > 0:
        area_ratio = min(pred_obj.area, true_obj.area) / max(pred_obj.area, true_obj.area)
        score += w_area * area_ratio

    # 3. Shape matching (canonical credit)
    if pred_obj.shape_mask == true_obj.shape_mask:
        score += w_shape
    elif pred_obj.canonical_shape == true_obj.canonical_shape:
        score += w_shape * 0.85

    # 4. Centroid proximity (normalized Euclidean distance)
    pr, pc = pred_obj.centroid
    tr, tc = true_obj.centroid
    dist = math.sqrt((pr - tr) ** 2 + (pc - tc) ** 2)
    
    # Estimate grid diagonal scale dynamically
    grid_diag = max(30.0, math.sqrt(max(pr, tr, 1) ** 2 + max(pc, tc, 1) ** 2) * 1.5)
    norm_dist = min(1.0, dist / grid_diag)
    score += max(0.0, w_centroid * (1.0 - norm_dist))

    return round(score, 3)

def greedy_match_objects(pred_objs: List[ArcObject], true_objs: List[ArcObject], min_threshold: float = None) -> Dict[str, Any]:
    # Optimal bipartite matching via Hungarian algorithm (linear_sum_assignment)
    pairs = []
    
    if min_threshold is None:
        _, _, _, _, min_threshold = get_active_weights()
    
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
        # Only accept matches meeting or exceeding the minimum match threshold
        if score >= min_threshold:
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

