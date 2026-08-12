import pytest
from core.objects import extract_objects
from core.diagnostics import diagnose_failure
from core.formatters import to_markdown_context, to_action_vector, to_json_api

def test_composite_object_color_map():
    # Outer box red (2), inner box blue (1)
    grid = [
        [0, 0, 0, 0, 0],
        [0, 2, 2, 2, 0],
        [0, 2, 1, 2, 0],
        [0, 2, 2, 2, 0],
        [0, 0, 0, 0, 0]
    ]
    objs = extract_objects(grid, background_color=0)
    assert len(objs) == 2
    parent = [o for o in objs if o.color == 2][0]
    assert parent.color_map.get(2) == 8
    assert parent.color_map.get(1) == 1

def test_formatters_action_vector():
    input_grid = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
    pred_grid  = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
    true_grid  = [[0, 0, 0], [0, 0, 1], [0, 0, 0]]

    report = diagnose_failure(input_grid, pred_grid, true_grid)
    actions = to_action_vector(report)
    assert len(actions) > 0
    translate_action = [a for a in actions if a["action"] == "translate"][0]
    assert translate_action["delta_col"] == 1.0

def test_json_api_payload():
    input_grid = [[0, 0], [1, 2]]
    pred_grid  = [[0, 0], [1, 0]]
    true_grid  = [[0, 0], [1, 2]]

    report = diagnose_failure(input_grid, pred_grid, true_grid)
    payload = to_json_api(report)
    assert "summary" in payload
    assert "action_vector" in payload
    assert "markdown_card" in payload
