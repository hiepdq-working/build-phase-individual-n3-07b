import pandas as pd
import pytest

from src.data.make_splits import pick_rare
from src.select.tool_select import select_top
from src.train.datasets import parse_run_id


def test_select_top_ties_by_score_then_image_id():
    cand = pd.DataFrame({"image_id": ["c", "b", "a", "d"], "score": [0.9, 0.8, 0.8, 0.7],
                         "uncertainty": [0.5, 0.5, 0.5, 0.9]})
    out = select_top(cand, 3)
    assert list(out[out.selected].image_id) == ["d", "c", "a"]
    sel, rest = out[out.selected], out[~out.selected]
    assert sel.uncertainty.min() >= rest.uncertainty.max()   # FR-SEL-04


def test_parse_run_id():
    assert parse_run_id("base") == ("base", 0)
    assert parse_run_id("tool_s2") == ("tool", 2)
    with pytest.raises(ValueError):
        parse_run_id("foo_s1")


def test_pick_rare_fallback():
    val = pd.DataFrame({"image_id": list("abcd"), "timeofday": ["night"] * 4,
                        "weather": ["rainy", "foggy", "clear", "clear"], "n_ped": [1, 1, 1, 0]})
    rs = {"min_pedestrians": 1, "min_images": 2, "levels": [
        {"timeofday": ["night"], "weather": ["rainy", "snowy"]},
        {"timeofday": ["night"], "weather": ["rainy", "snowy", "foggy"]},
        {"timeofday": ["night"], "weather": None}]}
    rare, lvl = pick_rare(val, rs)
    assert lvl == 1 and set(rare.image_id) == {"a", "b"}
