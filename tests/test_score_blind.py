import pandas as pd

from src.eval.score_blind import cohen_kappa, final_labels


def L(rows):
    return pd.DataFrame(rows, columns=["pair_id", "judge_id", "relevant"])


def test_primary_judge_and_disagreement():
    labels = L([[1, "j1", 1], [2, "j1", 0], [3, "j1", 1], [1, "j2", 1], [2, "j2", 1]])
    rel, kappa, dis = final_labels(labels)
    assert rel.to_dict() == {1: 1, 2: 0, 3: 1}             # nhãn người chấm chính
    assert list(dis.pair_id) == [2]
    assert kappa["n_pairs"] == 2 and kappa["agree_pct"] == 50.0


def test_adjudication_overrides():
    labels = L([[1, "j1", 0], [1, "j2", 1]])
    adj = pd.DataFrame({"pair_id": [1], "relevant": [1]})
    rel, _, dis = final_labels(labels, adj)
    assert rel[1] == 1 and dis.empty


def test_last_label_wins():
    rel, kappa, _ = final_labels(L([[1, "j1", 0], [1, "j1", 1]]))
    assert rel[1] == 1 and kappa is None


def test_kappa_values():
    assert cohen_kappa([1, 0, 1, 0], [1, 0, 1, 0]) == 1.0
    assert abs(cohen_kappa([1, 1, 0, 0], [1, 0, 1, 0])) < 1e-9
