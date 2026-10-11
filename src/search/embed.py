"""G2–G3 · Embedding CLIP theo shard + dựng chỉ mục FAISS (FR-IDX-02 → 09). Chạy trên Colab GPU.

    python -m src.search.embed --model ViT-B-32 --limit 200 --per-shard 100   # đo tốc độ (NFR-23)
    python -m src.search.embed --model ViT-B-32                               # 40.000 ảnh pool
    python -m src.search.embed --model ViT-B-16-SigLIP-384 --amp              # model thứ hai (nếu kịp)

Shard ≈ 1.428 ảnh (≈ 10.000 crop); shard đã xong được bỏ qua khi chạy lại, nên mất phiên Colab
chỉ mất tối đa 1 shard (FR-IDX-04, NFR-27). Chỉ mục whole lấy đúng các vector tile 0 của chỉ mục
whole_tiles nên trùng từng bit (FR-IDX-07).
"""
import argparse
import json
import time

import faiss
import numpy as np
import open_clip
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.common.config import load_decisions
from src.common.paths import INDEX_ROOT, img_path
from src.data.splits_io import read_split
from src.search.tiling import pad_square, tile_boxes

SLUG = {"ViT-B-32": "vitb32", "ViT-B-16-SigLIP-384": "siglip384"}


class CropDS(Dataset):
    def __init__(self, ids, preprocess, tiling):
        self.ids, self.pp, self.t = ids, preprocess, tiling

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        im = Image.open(img_path(self.ids[i])).convert("RGB")
        return torch.stack([self.pp(pad_square(im.crop(b))) for b in tile_boxes(*im.size, **self.t)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ViT-B-32", choices=list(SLUG))
    ap.add_argument("--amp", action="store_true", help="fp16 trên GPU, dùng cho SigLIP")
    ap.add_argument("--per-shard", type=int, default=1428)
    ap.add_argument("--batch", type=int, default=32, help="số ảnh mỗi batch (× 7 crop)")
    ap.add_argument("--limit", type=int, help="chạy thử N ảnh để đo tốc độ")
    a = ap.parse_args()

    cfg = load_decisions()["clip"]
    if cfg.get("preprocess") != "pad_square":                       # FR-IDX-02
        raise SystemExit("decisions.yaml: thiếu hoặc sai khóa clip.preprocess (cần pad_square)")
    tiling = cfg["tiling"]
    pre = {m["name"]: m["pretrained"] for m in cfg["models"]}[a.model]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(a.model, pretrained=pre)
    model = model.to(dev).eval()
    for p in model.parameters():                                     # NFR-08: không bao giờ cập nhật CLIP
        p.requires_grad_(False)
    ids = read_split("pool")[: a.limit]
    slug = SLUG[a.model] + ("_test" if a.limit else "")
    tmp = INDEX_ROOT / f"_shards_{slug}"
    tmp.mkdir(parents=True, exist_ok=True)
    nt = 1 + tiling["cols"] * tiling["rows"]
    use_amp = a.amp and dev == "cuda"

    t0, done = time.time(), 0
    for s in range(0, len(ids), a.per_shard):
        out = tmp / f"shard_{s // a.per_shard:04d}.npy"
        if out.exists():
            continue
        chunk = ids[s:s + a.per_shard]
        dl = DataLoader(CropDS(chunk, preprocess, tiling), batch_size=a.batch, num_workers=2)
        vecs = []
        with torch.no_grad(), torch.autocast("cuda", enabled=use_amp):
            for x in dl:
                f = model.encode_image(x.flatten(0, 1).to(dev)).float()
                vecs.append(torch.nn.functional.normalize(f, dim=-1).cpu().numpy())
        np.save(out, np.concatenate(vecs).astype(np.float32))
        done += len(chunk) * nt
        rate = done / (time.time() - t0)
        print(f"{out.name}: {time.time() - t0:.0f}s, {rate:.0f} crop/s, "
              f"ước tính còn {(len(ids) * nt - done) / rate / 60:.0f} phút", flush=True)

    V = np.concatenate([np.load(p) for p in sorted(tmp.glob("shard_*.npy"))])
    assert len(V) == len(ids) * nt, f"thiếu shard: {len(V)} vector, cần {len(ids) * nt}"
    lookup = pd.DataFrame({"image_id": np.repeat(ids, nt), "tile": np.tile(np.arange(nt), len(ids))})
    common = dict(model=a.model, pretrained=pre, preprocess=cfg["preprocess"], tiling_grid=tiling,
                  backend=f"torch-{dev}", precision="fp16-autocast" if use_amp else "fp32",
                  torch=torch.__version__, open_clip=open_clip.__version__, faiss=faiss.__version__,
                  crops_per_second=round(done / (time.time() - t0), 1) if done else None,
                  created_at=time.strftime("%Y-%m-%dT%H:%M:%S"))
    for name, mask in [("whole_tiles", np.ones(len(V), bool)), ("whole", lookup.tile.values == 0)]:
        d = INDEX_ROOT / f"{slug}_{name}"
        d.mkdir(parents=True, exist_ok=True)
        v = np.ascontiguousarray(V[mask])
        lk = lookup[mask].reset_index(drop=True)
        lk.insert(0, "row", np.arange(len(lk)))
        np.save(d / "vectors.npy", v)
        idx = faiss.IndexFlatIP(v.shape[1])
        idx.add(v)
        faiss.write_index(idx, str(d / "index.faiss"))
        lk.to_parquet(d / "lookup.parquet", index=False)
        meta = dict(index_id=d.name, tiling=name, n_vectors=int(len(v)), dim=int(v.shape[1]), **common)
        (d / "meta.json").write_text(json.dumps(meta, indent=2))
        print("Đã lưu", d.name, len(v), "vector")


if __name__ == "__main__":
    main()
