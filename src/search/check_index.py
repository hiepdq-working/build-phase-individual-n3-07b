"""Kiểm một chỉ mục sau khi tải về Mac (FR-IDX-05, 06, 08; NFR-17).

    python -m src.search.check_index vitb32_whole_tiles
    python -m src.search.check_index vitb32_whole

- meta.json đủ trường, 4 file đủ, số dòng khớp nhau, vector chuẩn hóa L2;
- 1.000 hàng ngẫu nhiên tự truy vấn: điểm top-1 = điểm của chính nó và ≥ 0.999;
- mã hóa lại 20 crop trên Mac (CPU) và so cosine với vector tính trên Colab.
"""
import json
import sys

import numpy as np
import pandas as pd
from PIL import Image

from src.common.config import load_decisions
from src.common.paths import INDEX_ROOT, img_path
from src.search.tiling import pad_square, tile_boxes

META_KEYS = ["index_id", "model", "pretrained", "preprocess", "tiling", "backend", "precision",
             "n_vectors", "dim", "torch", "open_clip", "faiss"]


def main(index_id, n_reencode=20):
    import faiss
    import open_clip
    import torch

    d = INDEX_ROOT / index_id
    for f in ["vectors.npy", "index.faiss", "lookup.parquet", "meta.json"]:
        assert (d / f).exists(), f"thiếu {f}"
    meta = json.loads((d / "meta.json").read_text())
    miss = [k for k in META_KEYS if k not in meta]
    assert not miss, f"meta.json thiếu {miss}"
    V = np.load(d / "vectors.npy")
    idx = faiss.read_index(str(d / "index.faiss"))
    lk = pd.read_parquet(d / "lookup.parquet")
    assert idx.ntotal == len(V) == len(lk) == meta["n_vectors"]
    assert np.abs(np.linalg.norm(V, axis=1) - 1).max() < 1e-3
    rng = np.random.default_rng(0)
    q = rng.choice(len(V), min(1000, len(V)), replace=False)
    D, _ = idx.search(V[q], 1)
    self_score = (V[q] * V[q]).sum(1)
    assert np.allclose(D[:, 0], self_score, atol=1e-5) and D[:, 0].min() >= 0.999
    print(f"{index_id}: {len(V)} vector, self-query OK")

    model, _, pp = open_clip.create_model_and_transforms(meta["model"], pretrained=meta["pretrained"])
    model.eval()
    t = load_decisions()["clip"]["tiling"]
    cos = []
    for r in rng.choice(len(lk), n_reencode, replace=False):
        row = lk.iloc[r]
        im = Image.open(img_path(row.image_id)).convert("RGB")
        x = pp(pad_square(im.crop(tile_boxes(*im.size, **t)[row.tile]))).unsqueeze(0)
        with torch.no_grad():
            f = torch.nn.functional.normalize(model.encode_image(x).float(), dim=-1).numpy()[0]
        cos.append(float(f @ V[r]))
    print(f"OK. Cosine Mac (CPU) vs {meta['backend']}/{meta['precision']}: "
          f"min {min(cos):.5f}, mean {np.mean(cos):.5f}  (ghi vào report — OI-09)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Cách dùng: python -m src.search.check_index <index_id>")
    main(sys.argv[1])
