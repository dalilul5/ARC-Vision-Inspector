from pathlib import Path

from core.objects import extract_objects
from core.matching import MatchingConfig, load_matching_config, object_match_score, match_objects_greedy, match_objects_hungarian


def test_object_match_score_exact():
    obj_a = extract_objects([[0, 0], [0, 1]])[0]
    obj_b = extract_objects([[0, 0], [0, 1]])[0]
    config = MatchingConfig()
    assert object_match_score(obj_a, obj_b, config) == round(config.weight_color + config.weight_area + config.weight_shape + config.weight_centroid, 3)


def test_different_color_same_object_can_still_match():
    pred = extract_objects([[0, 0], [0, 1]])
    true = extract_objects([[0, 0], [0, 2]])
    result = match_objects_hungarian(pred, true)
    assert len(result["matches"]) == 1
    assert result["matches"][0]["pred_color"] == 1
    assert result["matches"][0]["true_color"] == 2


def test_matching_discards_false_pairs_below_threshold():
    pred = extract_objects([[1,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]])
    true = extract_objects([[0,0,0,0,0],[0,0,0,0,0],[0,0,2,2,2],[0,0,2,2,2],[0,0,2,2,2]])
    result = match_objects_hungarian(pred, true, min_threshold=3.5)
    assert not result["matches"]
    assert len(result["unmatched_pred"]) == len(result["unmatched_true"]) == 1


def test_invalid_config_falls_back_to_defaults(tmp_path: Path):
    path = tmp_path / "weights.json"
    path.write_text("not-json", encoding="utf-8")
    assert load_matching_config(path) == MatchingConfig()


def test_hungarian_never_scores_below_greedy():
    pred = extract_objects([[1,0,2],[1,0,2],[0,0,0]], background_color=0)
    true = extract_objects([[1,0,2],[0,0,2],[0,0,2]], background_color=0)
    greedy = match_objects_greedy(pred, true, min_threshold=0)
    hungarian = match_objects_hungarian(pred, true, min_threshold=0)
    assert sum(m["score"] for m in hungarian["matches"]) >= sum(m["score"] for m in greedy["matches"])
