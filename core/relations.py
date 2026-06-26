from typing import List, Dict
from math import sqrt
from core.objects import ArcObject

def touches(obj_a: ArcObject, obj_b: ArcObject) -> bool:
    pixels_a = set(obj_a.pixels)
    pixels_b = set(obj_b.pixels)
    for ra, ca in pixels_a:
        for rb, cb in pixels_b:
            if abs(ra - rb) + abs(ca - cb) == 1:
                return True
    return False

def is_contained(obj_a: ArcObject, obj_b: ArcObject) -> bool:
    """Checks if obj_a is strictly contained within obj_b's bounding box."""
    # Ensure all pixels of obj_a are inside the min/max bounds of obj_b
    b_min_r, b_min_c, b_max_r, b_max_c = obj_b.bbox
    for r, c in obj_a.pixels:
        if not (b_min_r <= r <= b_max_r and b_min_c <= c <= b_max_c):
            return False
    return True

def bbox_contains(obj_a: ArcObject, obj_b: ArcObject) -> bool:
    a_min_r, a_min_c, a_max_r, a_max_c = obj_a.bbox
    b_min_r, b_min_c, b_max_r, b_max_c = obj_b.bbox
    return (
        a_min_r <= b_min_r and
        a_min_c <= b_min_c and
        a_max_r >= b_max_r and
        a_max_c >= b_max_c
    )

def centroid_distance(obj_a: ArcObject, obj_b: ArcObject) -> float:
    ar, ac = obj_a.centroid
    br, bc = obj_b.centroid
    return round(sqrt((ar - br) ** 2 + (ac - bc) ** 2), 3)

def relative_position(obj_a: ArcObject, obj_b: ArcObject) -> str:
    ar, ac = obj_a.centroid
    br, bc = obj_b.centroid
    vertical = "same_row_band"
    horizontal = "same_col_band"

    if ar < br:
        vertical = "above"
    elif ar > br:
        vertical = "below"

    if ac < bc:
        horizontal = "left_of"
    elif ac > bc:
        horizontal = "right_of"

    return f"{vertical}|{horizontal}"

def build_relation_graph(objects: List[ArcObject]) -> List[Dict]:
    rels = []
    for i in range(len(objects)):
        for j in range(i + 1, len(objects)):
            a, b = objects[i], objects[j]
            rels.append({
                "obj_a": a.obj_id,
                "obj_b": b.obj_id,
                "same_color": a.color == b.color,
                "touches": touches(a, b),
                "contains_bbox": bbox_contains(a, b) or bbox_contains(b, a),
                "centroid_distance": centroid_distance(a, b),
                "relative_position": relative_position(a, b),
            })
    return rels
