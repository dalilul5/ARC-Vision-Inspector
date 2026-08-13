import pytest
from core.grid_utils import validate_grid

def test_validate_grid_valid():
    grid = [[0, 1], [2, 3]]
    res = validate_grid(grid)
    assert res == grid

def test_validate_grid_empty():
    with pytest.raises(ValueError, match="non-empty 2D list"):
        validate_grid([])

def test_validate_grid_ragged_rows():
    with pytest.raises(ValueError, match="does not match first row length"):
        validate_grid([[0, 1], [2]])

def test_validate_grid_invalid_color_range():
    with pytest.raises(ValueError, match="invalid color value"):
        validate_grid([[0, 10]])
    with pytest.raises(ValueError, match="invalid color value"):
        validate_grid([[-1, 0]])

def test_validate_grid_non_integer():
    with pytest.raises(ValueError, match="not an integer"):
        validate_grid([[0, "a"]])
