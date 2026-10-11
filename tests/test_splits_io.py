import pandas as pd

from src.data import splits_io


def test_write_split_is_deterministic(tmp_path, monkeypatch):
    monkeypatch.setattr(splits_io, "SPLITS", tmp_path)
    sha1 = splits_io.write_split("x", ["train/b", "train/a", "train/a"], seed=1)
    sha2 = splits_io.write_split("x", ["train/a", "train/b"], seed=1)
    assert sha1 == sha2
    assert splits_io.read_split("x") == ["train/a", "train/b"]
    reg = pd.read_csv(tmp_path / "registry.csv")
    assert len(reg) == 1 and reg.iloc[0].n_images == 2
