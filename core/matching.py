import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from core.objects import ArcObject

CONFIG_PATH = Path(__file__).resolve().parent.parent / "weights.json"
DEFAULT_MIN_MATCH_THRESHOLD = 3.5


@dataclass(frozen=True)
class MatchingConfig:
    weight_color: float = 3.0
    weight_area: float = 2.0
    weight_shape: float = 4.0
    weight_centroid: float = 2.0
    min_match_threshold: float = DEFAULT_MIN_MATCH_THRESHOLD


def load_matching_config(path: Path = CONFIG_PATH) -> MatchingConfig:
    defaults = MatchingConfig()
    if not path.exists():
        warnings.warn(f"Matching config not found at {path}; using defaults.", RuntimeWarning)
        return defaults
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return MatchingConfig(
            weight_color=float(data.get("weight_color", defaults.weight_color)),
            weight_area=float(data.get("weight_area", defaults.weight_area)),
            weight_shape=float(data.get("weight_shape", defaults.weight_shape)),
            weight_centroid=float(data.get("weight_centroid", defaults.weight_centroid)),
            min_match_threshold=float(data.get("min_match_threshold", defaults.min_match_threshold)),
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        warnings.warn(f"Invalid matching config at {path}: {exc}; using defaults.", RuntimeWarning)
        return defaults


def get_active_weights():
    config = load_matching_config()
    return (
        config.weight_color,
        config.weight_area,
        config.weight_shape,
        config.weight_centroid,
        config.min_match_threshold,
    )


def detect_shape_relation(pred_obj: ArcObject, true_obj: ArcObject) -> str:
    if pred_obj.shape_mask == true_obj.shape_mask:
        return "exact_shape"
    for variant_name, variant_sig in pred_obj.variant_signatures.items():
        if variant_sig == true_obj.shape_mask:
            return f"transformed_via_{variant_name}"
    if pred_obj.canonical_shape == true_obj.canonical_shape:
        return "same_canonical_shape_different_variant"
    return "different_shape"


def object_match_score(pred_obj: ArcObject, true_obj: ArcObject, config: MatchingConfig = None) -> float:
    config = config or load_matching_config()
    score = 0.0
    if pred_obj.color == true_obj.color:
        score += config.weight_color
    if pred_obj.area > 0 and true_obj.area > 0:
        area_ratio = min(pred_obj.area, true_obj.area) / max(pred_obj.area, true_obj.area)
        score += config.weight_area * area_ratio
    if pred_obj.shape_mask == true_obj.shape_mask:
        score += config.weight_shape
    elif pred_obj.canonical_shape == true_obj.canonical_shape:
        score += config.weight_shape * 0.85
    pr, pc = pred_obj.centroid
    tr, tc = true_obj.centroid
    dist = math.hypot(pr - tr, pc - tc)
    grid_diag = max(30.0, math.hypot(max(pr, tr, 1), max(pc, tc, 1)) * 1.5)
    score += max(0.0, config.weight_centroid * (1.0 - min(1.0, dist / grid_diag)))
    return round(score, 3)


def _format_match(pred_obj: ArcObject, true_obj: ArcObject, score: float) -> Dict[str, Any]:
    return {
        "pred_obj_id": pred_obj.obj_id,
        "true_obj_id": true_obj.obj_id,
        "score": float(score),
        "pred_color": pred_obj.color,
        "true_color": true_obj.color,
        "pred_area": pred_obj.area,
        "true_area": true_obj.area,
        "pred_centroid": pred_obj.centroid,
        "true_centroid": true_obj.centroid,
        "shape_relation": detect_shape_relation(pred_obj, true_obj),
    }


def match_objects_hungarian(pred_objs: List[ArcObject], true_objs: List[ArcObject], config: MatchingConfig = None, min_threshold: float = None) -> Dict[str, Any]:
    config = config or load_matching_config()
    threshold = config.min_match_threshold if min_threshold is None else min_threshold
    if not pred_objs or not true_objs:
        return {"matches": [], "unmatched_pred": [o.obj_id for o in pred_objs], "unmatched_true": [o.obj_id for o in true_objs]}
    scores = np.array([[object_match_score(p, t, config) for t in true_objs] for p in pred_objs], dtype=float)
    rows, cols = linear_sum_assignment(-scores)
    used_pred, used_true, matches = set(), set(), []
    for i, j in zip(rows, cols):
        score = scores[i, j]
        if score >= threshold:
            used_pred.add(i)
            used_true.add(j)
            matches.append(_format_match(pred_objs[i], true_objs[j], score))
    return {
        "matches": matches,
        "unmatched_pred": [o.obj_id for i, o in enumerate(pred_objs) if i not in used_pred],
        "unmatched_true": [o.obj_id for i, o in enumerate(true_objs) if i not in used_true],
    }


def match_objects_greedy(pred_objs: List[ArcObject], true_objs: List[ArcObject], config: MatchingConfig = None, min_threshold: float = None) -> Dict[str, Any]:
    config = config or load_matching_config()
    threshold = config.min_match_threshold if min_threshold is None else min_threshold
    candidates = sorted(
        ((object_match_score(p, t, config), i, j) for i, p in enumerate(pred_objs) for j, t in enumerate(true_objs)),
        key=lambda item: (-item[0], item[1], item[2]),
    )
    used_pred, used_true, matches = set(), set(), []
    for score, i, j in candidates:
        if score < threshold or i in used_pred or j in used_true:
            continue
        used_pred.add(i)
        used_true.add(j)
        matches.append(_format_match(pred_objs[i], true_objs[j], score))
    return {
        "matches": matches,
        "unmatched_pred": [o.obj_id for i, o in enumerate(pred_objs) if i not in used_pred],
        "unmatched_true": [o.obj_id for i, o in enumerate(true_objs) if i not in used_true],
    }


def greedy_match_objects(pred_objs, true_objs, min_threshold=None):
    """Deprecated compatibility alias. Use match_objects_hungarian."""
    warnings.warn("greedy_match_objects is deprecated; use match_objects_hungarian", DeprecationWarning, stacklevel=2)
    return match_objects_hungarian(pred_objs, true_objs, min_threshold=min_threshold)
