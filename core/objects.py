from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from scipy.ndimage import binary_fill_holes

from core.grid_utils import (
    to_numpy,
    infer_background_color,
    neighbors4,
    neighbors8,
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
    parent_id: Optional[int] = None
    children: List[int] = field(default_factory=list)
    holes: List[Tuple[int, int]] = field(default_factory=list)
    color_map: Dict[int, int] = field(default_factory=dict)
    composite_shape_mask: Any = None
    composite_canonical_shape: Any = None

    def to_dict(self):
        data = asdict(self)
        data["shape_mask"] = [list(row) for row in self.shape_mask]
        data["canonical_shape"] = [list(row) for row in self.canonical_shape]
        data["variant_signatures"] = {
            k: [list(row) for row in v] for k, v in self.variant_signatures.items()
        }
        if self.composite_shape_mask is not None:
            data["composite_shape_mask"] = [list(row) for row in self.composite_shape_mask]
        if self.composite_canonical_shape is not None:
            data["composite_canonical_shape"] = [list(row) for row in self.composite_canonical_shape]
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

def bbox_strictly_contains(bbox_parent: Tuple[int, int, int, int], bbox_child: Tuple[int, int, int, int]) -> bool:
    pr1, pc1, pr2, pc2 = bbox_parent
    cr1, cc1, cr2, cc2 = bbox_child
    return pr1 <= cr1 and pc1 <= cc1 and pr2 >= cr2 and pc2 >= cc2 and (bbox_parent != bbox_child)

def build_object_hierarchy(objects: List[ArcObject]) -> List[ArcObject]:
    """Establishes parent-child relationships, computes color_map fingerprints and composite shape masks."""
    obj_dict = {o.obj_id: o for o in objects}
    for i, obj_a in enumerate(objects):
        for j, obj_b in enumerate(objects):
            if i == j:
                continue
            if bbox_strictly_contains(obj_a.bbox, obj_b.bbox):
                if obj_b.parent_id is None or (obj_dict[obj_b.parent_id].area > obj_a.area):
                    obj_b.parent_id = obj_a.obj_id
                    if obj_b.obj_id not in obj_a.children:
                        obj_a.children.append(obj_b.obj_id)

    # Compute color_map and composite_shape_mask for each object
    for obj in objects:
        c_map = {obj.color: obj.area}
        all_pixels = list(obj.pixels)
        for child_id in obj.children:
            child = obj_dict[child_id]
            c_map[child.color] = c_map.get(child.color, 0) + child.area
            all_pixels.extend(child.pixels)
        obj.color_map = c_map

        if obj.children:
            comp_mask_np = crop_binary_mask_from_pixels(all_pixels)
            comp_canonical, _ = canonicalize_shape(comp_mask_np)
            obj.composite_shape_mask = mask_to_tuple(comp_mask_np)
            obj.composite_canonical_shape = comp_canonical
        else:
            obj.composite_shape_mask = obj.shape_mask
            obj.composite_canonical_shape = obj.canonical_shape

    return objects

def detect_holes(grid: List[List[int]], objects: List[ArcObject], background_color: int = None) -> None:
    """Detects background pixels completely enclosed within an object using binary flood-fill."""
    arr = to_numpy(grid)
    if background_color is None:
        background_color = infer_background_color(grid)

    for obj in objects:
        min_r, min_c, max_r, max_c = obj.bbox
        shape_mask_np = crop_binary_mask_from_pixels(obj.pixels)
        filled = binary_fill_holes(shape_mask_np)
        hole_mask = filled & (~(shape_mask_np.astype(bool)))
        
        obj_holes = []
        hole_coords = np.argwhere(hole_mask)
        for hr_local, hc_local in hole_coords:
            gr, gc = int(min_r + hr_local), int(min_c + hc_local)
            if arr[gr, gc] == background_color:
                obj_holes.append((gr, gc))
        obj.holes = sorted(obj_holes)

class ObjectExtractor(ABC):
    @abstractmethod
    def extract(self, grid: List[List[int]], background_color: int = None, connectivity: int = 4) -> List[ArcObject]:
        pass

class SingleColorConnectedComponentExtractor(ObjectExtractor):
    def __init__(self, connectivity: int = 4):
        self.connectivity = connectivity

    def extract(self, grid: List[List[int]], background_color: int = None, connectivity: int = None) -> List[ArcObject]:
        conn = connectivity if connectivity is not None else self.connectivity
        neighbor_fn = neighbors8 if conn == 8 else neighbors4

        arr = to_numpy(grid)
        rows, cols = arr.shape

        if background_color is None:
            background_color = infer_background_color(grid)

        visited = np.zeros((rows, cols), dtype=bool)
        objects = []
        obj_id = 0

        for r in range(rows):
            for c in range(cols):
                if visited[r, c] or arr[r, c] == background_color:
                    continue

                color = int(arr[r, c])
                stack = [(r, c)]
                visited[r, c] = True
                pixels = []

                while stack:
                    cr, cc = stack.pop()
                    pixels.append((cr, cc))
                    for nr, nc in neighbor_fn(cr, cc, rows, cols):
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

        build_object_hierarchy(objects)
        detect_holes(grid, objects, background_color)
        return objects

def extract_objects(grid: List[List[int]], background_color: int = None, connectivity: int = 4) -> List[ArcObject]:
    extractor = SingleColorConnectedComponentExtractor(connectivity=connectivity)
    return extractor.extract(grid, background_color)

