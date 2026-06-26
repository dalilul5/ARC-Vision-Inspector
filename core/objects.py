from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Any
import numpy as np

from core.grid_utils import (
    to_numpy,
    infer_background_color,
    neighbors4,
    crop_binary_mask_from_pixels,
    mask_to_tuple,
    all_shape_variants,
)

@dataclass
class ArcObject:
    obj_id: int
    color: int
    pixels: List[Tuple[int, int]]
    area: int
    bbox: Tuple[int, int, int, int]
    height: int
    width: int
    centroid: Tuple[float, float]
    shape_mask: Any
    canonical_shape: Any
    variant_signatures: Dict[str, Any]

    def to_dict(self):
        data = asdict(self)
        data["shape_mask"] = [list(row) for row in self.shape_mask]
        data["canonical_shape"] = [list(row) for row in self.canonical_shape]
        data["variant_signatures"] = {
            k: [list(row) for row in v] for k, v in self.variant_signatures.items()
        }
        return data

def compute_bbox(pixels: List[Tuple[int, int]]) -> Tuple[int, int, int, int]:
    rs = [p[0] for p in pixels]
    cs = [p[1] for p in pixels]
    return min(rs), min(cs), max(rs), max(cs)

def compute_centroid(pixels: List[Tuple[int, int]]) -> Tuple[float, float]:
    rs = [p[0] for p in pixels]
    cs = [p[1] for p in pixels]
    return (round(sum(rs) / len(rs), 3), round(sum(cs) / len(cs), 3))

def canonicalize_shape(mask: np.ndarray):
    variants = all_shape_variants(mask)
    variant_tuples = {name: mask_to_tuple(v) for name, v in variants.items()}
    canonical_name = min(variant_tuples, key=lambda k: str(variant_tuples[k]))
    canonical_shape = variant_tuples[canonical_name]
    return canonical_shape, variant_tuples

def extract_objects(grid: List[List[int]], background_color: int = None) -> List[ArcObject]:
    arr = to_numpy(grid)
    rows, cols = arr.shape

    if background_color is None:
        background_color = infer_background_color(grid)

    visited = np.zeros((rows, cols), dtype=bool)
    objects = []
    obj_id = 0

    for r in range(rows):
        for c in range(cols):
            if visited[r, c]:
                continue
            if arr[r, c] == background_color:
                continue

            color = int(arr[r, c])
            stack = [(r, c)]
            visited[r, c] = True
            pixels = []

            while stack:
                cr, cc = stack.pop()
                pixels.append((cr, cc))
                for nr, nc in neighbors4(cr, cc, rows, cols):
                    if not visited[nr, nc] and arr[nr, nc] == color:
                        visited[nr, nc] = True
                        stack.append((nr, nc))

            bbox = compute_bbox(pixels)
            min_r, min_c, max_r, max_c = bbox
            centroid = compute_centroid(pixels)
            shape_mask_np = crop_binary_mask_from_pixels(pixels)
            canonical_shape, variant_signatures = canonicalize_shape(shape_mask_np)

            objects.append(
                ArcObject(
                    obj_id=obj_id,
                    color=color,
                    pixels=sorted(pixels),
                    area=len(pixels),
                    bbox=bbox,
                    height=max_r - min_r + 1,
                    width=max_c - min_c + 1,
                    centroid=centroid,
                    shape_mask=mask_to_tuple(shape_mask_np),
                    canonical_shape=canonical_shape,
                    variant_signatures=variant_signatures,
                )
            )
            obj_id += 1

    return objects
