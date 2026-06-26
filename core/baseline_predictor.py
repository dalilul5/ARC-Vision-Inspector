from typing import List
import copy

Grid = List[List[int]]

def identity_predictor(input_grid: Grid) -> Grid:
    return copy.deepcopy(input_grid)

def horizontal_shift_predictor(input_grid: Grid, shift: int = 1) -> Grid:
    rows = len(input_grid)
    cols = len(input_grid[0])
    out = [[0 for _ in range(cols)] for _ in range(rows)]

    for r in range(rows):
        for c in range(cols):
            val = input_grid[r][c]
            if val != 0:
                nc = c + shift
                if 0 <= nc < cols:
                    out[r][nc] = val
    return out

def simple_predict(input_grid: Grid, mode: str = "identity") -> Grid:
    if mode == "identity":
        return identity_predictor(input_grid)
    elif mode == "shift_right":
        return horizontal_shift_predictor(input_grid, shift=1)
    else:
        return identity_predictor(input_grid)
