"""Kiểm cấu trúc decisions.yaml và queries.yaml (FR-EVL-01, 05; FR-SRC-10)."""
from src.common.config import load_decisions, load_queries
from src.eval.gt_rules import RULES


def test_decisions_keys():
    d = load_decisions()
    for k in ["seeds", "sizes", "class_map", "rare_slice", "clip", "uncertainty", "train", "eval",
              "timing", "conclusion_rule"]:
        assert k in d, k
    assert sorted(d["class_map"].values()) == list(range(7))
    assert d["clip"]["preprocess"] == "pad_square"
    assert d["seeds"]["train"] == [0, 1, 2]


def test_queries_structure():
    q = load_queries()
    assert 3 <= len(q["tool"]) <= 5
    auto = q["eval_auto"]
    assert len(auto) == 20
    assert all(a["rule"] in RULES for a in auto)
    human = q["eval_human"]
    assert all("rule" not in h and h.get("definition") for h in human)
    ids = [x["id"] for x in auto + human]
    assert len(ids) == len(set(ids))
