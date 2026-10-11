"""G9 · arm_random và arm_meta (FR-SEL-06, 07). Chạy trên Mac.

    python -m src.select.make_arms && python -m src.data.check_splits
"""
import numpy as np
import pandas as pd

from src.common.config import load_decisions
from src.common.paths import META
from src.data.splits_io import read_split, write_split


def main():
    cfg = load_decisions()
    k, sd = cfg["sizes"]["arm_k"], cfg["seeds"]
    pool = sorted(read_split("pool"))
    write_split("arm_random", np.random.default_rng(sd["arm_random"]).choice(pool, k, replace=False),
                sd["arm_random"], "ngẫu nhiên đều từ pool")
    imgs = pd.read_parquet(META / "images.parquet")
    hard = imgs[imgs.image_id.isin(set(pool)) & (imgs.timeofday == "night") & imgs.weather.isin(["rainy", "snowy"])]
    meta_ids = sorted(hard.image_id)
    if len(meta_ids) < k:
        raise SystemExit(f"Pool chỉ có {len(meta_ids)} ảnh night ∧ (rainy ∨ snowy), cần ≥ {k}")
    write_split("arm_meta", np.random.default_rng(sd["arm_meta"]).choice(meta_ids, k, replace=False),
                sd["arm_meta"], f"ngẫu nhiên trong {len(meta_ids)} ảnh night ∧ (rainy ∨ snowy) của pool")


if __name__ == "__main__":
    main()
