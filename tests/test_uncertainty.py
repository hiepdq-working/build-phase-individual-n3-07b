from src.uncertainty.score import uncertainty


def test_edges():
    assert uncertainty([]) == 0.0
    assert uncertainty([0.2, 0.5]) == 1.0          # hai biên đều tính là phân vân
    assert uncertainty([0.19, 0.9]) == 0.0         # 0.19 bị loại, còn 1 box chắc chắn
    assert uncertainty([0.3, 0.9]) == 0.5
    assert uncertainty([0.1, 0.15]) == 0.0         # không có box ≥ 0.2 -> empty_value
    assert uncertainty([0.1], empty=-1) == -1
    assert 0 <= uncertainty([0.21, 0.51, 0.7]) <= 1
