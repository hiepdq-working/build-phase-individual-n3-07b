"""G4–G5 · Bộ máy tìm kiếm (FR-SRC-01 → 11).

Chỉ mục phẳng (IndexFlatIP) là tìm chính xác, nên lấy toàn bộ vector ra rồi nhân ma trận:
chấm hết 280.000 vector (k_vectors = n_vectors, FR-SRC-09), mất vài chục mili giây.

torch / open_clip / faiss chỉ được import khi cần, để test và CI không phải cài chúng.
"""
import json
import time
import uuid

import numpy as np
import pandas as pd

from src.common.paths import INDEX_ROOT, RESULTS
from src.search.tiling import pad_square

META_KEYS = ["model", "pretrained", "preprocess", "tiling", "backend", "precision", "n_vectors"]


def rank_hits(V, image_ids, tiles, q, n=50, allowed=None):
    """Điểm ảnh = max qua mọi tile và mọi prompt (FR-SRC-04); image_id không lặp (FR-SRC-05);
    trả best_tile (FR-SRC-06); hòa điểm thì image_id tăng dần (FR-SRC-07); lọc theo `allowed` (FR-SRC-08)."""
    s = V @ q.T                                            # (n_vectors, n_prompt)
    df = pd.DataFrame({"image_id": image_ids, "tile": tiles, "score": s.max(axis=1)})
    if allowed is not None:
        df = df[df.image_id.isin(allowed)]
    df = df.sort_values(["score", "tile"], ascending=[False, True]).drop_duplicates("image_id")
    df = df.sort_values(["score", "image_id"], ascending=[False, True]).head(n)
    df = df.rename(columns={"tile": "best_tile"}).reset_index(drop=True)
    df.insert(0, "rank", np.arange(1, len(df) + 1))
    return df


class SearchIndex:
    def __init__(self, index_id):
        import faiss

        d = INDEX_ROOT / index_id
        if not (d / "meta.json").exists():
            raise ValueError(f"{index_id}: không có meta.json")
        self.meta = json.loads((d / "meta.json").read_text())
        miss = [k for k in META_KEYS if k not in self.meta]
        if miss:
            raise ValueError(f"{index_id}: meta.json thiếu {miss}")   # FR-APP-04
        idx = faiss.read_index(str(d / "index.faiss"))
        self.V = idx.reconstruct_n(0, idx.ntotal)
        lk = pd.read_parquet(d / "lookup.parquet")
        if len(lk) != idx.ntotal:
            raise ValueError(f"{index_id}: lookup.parquet {len(lk)} dòng ≠ {idx.ntotal} vector")
        self.image_ids, self.tiles = lk.image_id.values, lk.tile.values
        self.index_id = index_id
        self._model = None

    def _load(self):
        """Luôn dùng đúng model + pretrained ghi trong meta.json của chỉ mục (FR-SRC-01, NFR-07)."""
        if self._model is None:
            import open_clip
            m, _, pp = open_clip.create_model_and_transforms(self.meta["model"], pretrained=self.meta["pretrained"])
            self._model, self._pp = m.eval(), pp
            self._tok = open_clip.get_tokenizer(self.meta["model"])

    def encode_text(self, prompts):
        prompts = [p.strip() for p in prompts if p and p.strip()]
        if not 1 <= len(prompts) <= 5:
            raise ValueError(f"Cần 1–5 prompt, đang có {len(prompts)}")   # FR-SRC-03
        import torch
        self._load()
        with torch.no_grad():
            f = self._model.encode_text(self._tok(prompts)).float()
        return torch.nn.functional.normalize(f, dim=-1).numpy()

    def encode_image(self, im):
        """OI-17: mã hóa ảnh truy vấn nguyên ảnh, cùng preprocess pad_square với chỉ mục (FR-SRC-02)."""
        import torch
        if self.meta["preprocess"] != "pad_square":
            raise ValueError(f"preprocess của chỉ mục là {self.meta['preprocess']}, code chỉ hỗ trợ pad_square")
        self._load()
        x = self._pp(pad_square(im.convert("RGB"))).unsqueeze(0)
        with torch.no_grad():
            f = self._model.encode_image(x).float()
        return torch.nn.functional.normalize(f, dim=-1).numpy()

    def search(self, q, n=50, allowed=None, prompt_agg="max"):
        """prompt_agg="max" là cách đo chính thức. "mean" (trung bình vector prompt, cách của repo cũ)
        chỉ để thử, không dùng khi đo."""
        if q.shape[1] != self.V.shape[1]:
            raise ValueError("Truy vấn được mã hóa bằng model khác chỉ mục")   # NFR-07
        if prompt_agg == "mean":
            q = q.mean(axis=0, keepdims=True)
            q = q / np.linalg.norm(q)
        return rank_hits(self.V, self.image_ids, self.tiles, q, n=n, allowed=allowed)


def log_run(index, kind, query_id, prompts, hits, elapsed):
    """Ghi results/search/runs.csv + hits/<id>.csv (FR-SRC-11). Dùng CSV để commit được."""
    rid = f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    out = RESULTS / "search"
    (out / "hits").mkdir(parents=True, exist_ok=True)
    hits.assign(search_run_id=rid).to_csv(out / "hits" / f"{rid}.csv", index=False)
    row = dict(search_run_id=rid, kind=kind, index_id=index.index_id, query_id=query_id,
               prompts=" | ".join(prompts), k_vectors=len(index.V), n_hits=len(hits),
               seconds=round(elapsed, 3), timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"))
    p = out / "runs.csv"
    pd.DataFrame([row]).to_csv(p, mode="a", header=not p.exists(), index=False)
    return rid
