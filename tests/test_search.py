import numpy as np
import pytest

from src.search.engine import SearchIndex, rank_hits

V = np.array([[1, 0], [0, 1], [0.6, 0.8], [0.6, 0.8]], dtype=np.float32)
IDS = np.array(["a", "a", "b", "c"])
TILES = np.array([0, 1, 0, 0])


def test_max_over_tiles_and_prompts():
    hits = rank_hits(V, IDS, TILES, np.array([[0, 1], [1, 0]], np.float32), n=3)
    assert hits.iloc[0].image_id == "a" and hits.iloc[0].score == 1.0
    assert hits.image_id.is_unique
    assert list(hits["rank"]) == [1, 2, 3]
    assert hits.score.is_monotonic_decreasing


def test_best_tile():
    hits = rank_hits(V, IDS, TILES, np.array([[0, 1]], np.float32), n=3)
    assert hits.set_index("image_id").loc["a", "best_tile"] == 1


def test_tie_break_by_image_id_and_deterministic():
    q = np.array([[0.6, 0.8]], np.float32)
    h1 = rank_hits(V, IDS, TILES, q, n=3)
    h2 = rank_hits(V, IDS, TILES, q, n=3)
    assert list(h1.image_id[:2]) == ["b", "c"]
    assert h1.equals(h2)


def test_filter_returns_min_n_allowed():
    hits = rank_hits(V, IDS, TILES, np.array([[1, 0]], np.float32), n=3, allowed={"c"})
    assert list(hits.image_id) == ["c"]


def _fake_index():
    s = SearchIndex.__new__(SearchIndex)
    s.V, s.image_ids, s.tiles = V, IDS, TILES
    return s


def test_model_mismatch_rejected():
    with pytest.raises(ValueError):
        _fake_index().search(np.ones((1, 3), np.float32))


def test_prompt_count_rejected():
    s = _fake_index()
    with pytest.raises(ValueError):
        s.encode_text([])
    with pytest.raises(ValueError):
        s.encode_text(["a"] * 6)
