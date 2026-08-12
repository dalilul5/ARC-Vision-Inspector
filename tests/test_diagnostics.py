import pytest
from core.diagnostics import DiagnosticEngine, diagnose_failure, PositionMismatchClassifier

def test_position_mismatch_vector():
    # Grid where true object is shifted 1 column to the right
    input_grid = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
    pred_grid  = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
    true_grid  = [[0, 0, 0], [0, 0, 1], [0, 0, 0]]

    engine = DiagnosticEngine()
    report = engine.run(input_grid, pred_grid, true_grid)

    assert len(report["pair_diagnostics"]) == 1
    pair = report["pair_diagnostics"][0]
    pos_failures = [f for f in pair["pair_failures"] if f["type"] == "position_or_translation_mismatch"]
    assert len(pos_failures) == 1
    assert pos_failures[0]["delta_row"] == 0.0
    assert pos_failures[0]["delta_col"] == 1.0
    assert "delta_col=1.0" in report["textual_diagnosis"]

def test_missing_object_global_failure():
    input_grid = [[0, 0], [1, 2]]
    pred_grid  = [[0, 0], [1, 0]]
    true_grid  = [[0, 0], [1, 2]]

    report = diagnose_failure(input_grid, pred_grid, true_grid)
    assert "missing_object" in report["global_failures"]
