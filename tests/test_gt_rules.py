import pandas as pd

from src.eval.gt_rules import RULES, build_table, is_rare, positives


def table():
    imgs = pd.DataFrame({
        "image_id": ["train/a", "train/b", "train/c"],
        "timeofday": ["night", "night", "daytime"],
        "weather": ["rainy", "clear", "snowy"],
        "scene": ["city street", "tunnel", "parking lot"],
    })
    bx = pd.DataFrame({
        "image_id": ["train/a", "train/a", "train/b"] + ["train/c"] * 15,
        "category": ["pedestrian", "pedestrian", "car"] + ["car"] * 15,
        "y1": [0, 0, 0] + [0] * 15,
        "y2": [30, 100, 50] + [50] * 15,
    })
    return build_table(imgs, bx)


def test_build_table_counts():
    t = table()
    assert t.loc["train/a", "pedestrian"] == 2 and t.loc["train/c", "car"] == 15
    assert t.loc["train/a", "small_ped"] and not t.loc["train/b", "small_ped"]


def test_rules_run_without_error():
    t = table()
    for name, fn in RULES.items():
        assert len(fn(t)) == len(t), name


def test_positives_and_within():
    t = table()
    assert positives("night", t) == {"train/a", "train/b"}
    assert positives("night", t, within={"train/b"}) == {"train/b"}
    assert positives("cars15", t) == {"train/c"}
    assert positives("night_ped", t) == {"train/a"}


def test_is_rare():
    t = table()
    assert set(t.index[is_rare(t)]) == {"train/a"}
