import pytest
from core.objects import extract_objects
from core.matching import object_match_score, greedy_match_objects

def test_object_match_score_exact():
    grid_a = [[0, 0], [0, 1]]
    grid_b = [[0, 0], [0, 1]]
    obj_a = extract_objects(grid_a)[0]
    obj_b = extract_objects(grid_b)[0]

    score = object_match_score(obj_a, obj_b)
    from core.matching import get_active_weights
    w_color, w_area, w_shape, w_centroid = get_active_weights()
    expected = w_color + w_area + w_shape + w_centroid
    assert score == expected

def test_matching_unmatched():
    grid_pred = [[0, 0], [0, 1]]
    grid_true = [[0, 0], [0, 2]]
    pred_objs = extract_objects(grid_pred)
    true_objs = extract_objects(grid_true)

    result = greedy_match_objects(pred_objs, true_objs)
    assert len(result["matches"]) == 1
    # Different colors but same area, shape, centroid
    assert result["matches"][0]["pred_color"] == 1
    assert result["matches"][0]["true_color"] == 2
