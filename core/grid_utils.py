from typing import List, Tuple
import numpy as np

Grid = List[List[int]]

def to_numpy(grid: Grid) -> np.ndarray:
    return np.array(grid, dtype=int)

def grid_shape(grid: Grid) -> Tuple[int, int]:
    arr = to_numpy(grid)
    return arr.shape

def infer_background_color(grid: Grid) -> int:
    arr = to_numpy(grid)
    values, counts = np.unique(arr, return_counts=True)
    return int(values[np.argmax(counts)])

def neighbors4(r: int, c: int, rows: int, cols: int):
    for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            yield nr, nc

def neighbors8(r: int, c: int, rows: int, cols: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                yield nr, nc

def validate_grid(grid: Grid) -> Grid:
    """Validates that grid is a non-empty 2D list of integers with uniform row length and valid ARC colors (0-9)."""
    if not isinstance(grid, list) or len(grid) == 0:
        raise ValueError("Grid must be a non-empty 2D list.")
    
    first_row = grid[0]
    if not isinstance(first_row, list) or len(first_row) == 0:
        raise ValueError("Grid rows must be non-empty lists.")
        
    num_cols = len(first_row)
    for r_idx, row in enumerate(grid):
        if not isinstance(row, list):
            raise ValueError(f"Row {r_idx} is not a list.")
        if len(row) != num_cols:
            raise ValueError(f"Row {r_idx} length ({len(row)}) does not match first row length ({num_cols}).")
        for c_idx, val in enumerate(row):
            if not isinstance(val, (int, np.integer)):
                raise ValueError(f"Cell at ({r_idx}, {c_idx}) is not an integer: {val}")
            if not (0 <= int(val) <= 9):
                raise ValueError(f"Cell at ({r_idx}, {c_idx}) has invalid color value {val} (must be 0-9).")
    return grid


def crop_binary_mask_from_pixels(pixels: List[Tuple[int, int]]) -> np.ndarray:
    rs = [r for r, _ in pixels]
    cs = [c for _, c in pixels]
    min_r, max_r = min(rs), max(rs)
    min_c, max_c = min(cs), max(cs)

    h = max_r - min_r + 1
    w = max_c - min_c + 1
    mask = np.zeros((h, w), dtype=int)

    for r, c in pixels:
        mask[r - min_r, c - min_c] = 1
    return mask

def normalize_coords(pixels: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    min_r = min(r for r, _ in pixels)
    min_c = min(c for _, c in pixels)
    return sorted([(r - min_r, c - min_c) for r, c in pixels])

def mask_to_tuple(mask: np.ndarray):
    return tuple(tuple(int(v) for v in row) for row in mask.tolist())

def rotate_mask(mask: np.ndarray, k: int) -> np.ndarray:
    return np.rot90(mask, k=k)

def flip_mask_h(mask: np.ndarray) -> np.ndarray:
    return np.fliplr(mask)

def flip_mask_v(mask: np.ndarray) -> np.ndarray:
    return np.flipud(mask)

def all_shape_variants(mask: np.ndarray):
    variants = {}
    for k in range(4):
        rot = rotate_mask(mask, k)
        variants[f"rot{k*90}"] = rot
        variants[f"rot{k*90}_flip_h"] = flip_mask_h(rot)
        variants[f"rot{k*90}_flip_v"] = flip_mask_v(rot)
    return variants
