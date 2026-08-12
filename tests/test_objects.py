import pytest
from core.objects import extract_objects

def test_extract_objects_basic():
    grid = [
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0]
    ]
    objs = extract_objects(grid, background_color=0)
    assert len(objs) == 1
    assert objs[0].color == 1
    assert objs[0].area == 1
    assert objs[0].centroid == (1.0, 1.0)

def test_object_hierarchy_nesting():
    # Large canvas of background 0
    # Outer box (red=2) enclosing inner box (blue=1)
    grid = [
        [0, 0, 0, 0, 0, 0, 0],
        [0, 2, 2, 2, 2, 2, 0],
        [0, 2, 1, 1, 1, 2, 0],
        [0, 2, 1, 0, 1, 2, 0],
        [0, 2, 1, 1, 1, 2, 0],
        [0, 2, 2, 2, 2, 2, 0],
        [0, 0, 0, 0, 0, 0, 0]
    ]
    objs = extract_objects(grid, background_color=0)
    assert len(objs) == 2
    parent = [o for o in objs if o.color == 2][0]
    child = [o for o in objs if o.color == 1][0]
    assert child.parent_id == parent.obj_id
    assert child.obj_id in parent.children

def test_hole_detection():
    # A hollow square of green (color 3) surrounding background (0) on a large grid
    grid = [
        [0, 0, 0, 0, 0],
        [0, 3, 3, 3, 0],
        [0, 3, 0, 3, 0],
        [0, 3, 3, 3, 0],
        [0, 0, 0, 0, 0]
    ]
    objs = extract_objects(grid, background_color=0)
    assert len(objs) == 1
    assert len(objs[0].holes) == 1
    assert objs[0].holes[0] == (2, 2)
