"""G8 · Chạy detector gốc trên 40.000 ảnh pool, mỗi ảnh một điểm uncertainty (FR-SEL-01, 02). Chạy trên Colab.

    python -m src.uncertainty.run_pool

Kết quả ghi lên RUNS_ROOT/uncertainty/ (Drive). Tải base_pool.csv + params.json về results/uncertainty/.
"""
import json

import pandas as pd

from src.common.config import load_decisions
from src.common.paths import RUNS_ROOT, img_path
from src.data.splits_io import read_split
from src.uncertainty.score import uncertainty


def main():
    from ultralytics import YOLO

    u = load_decisions()["uncertainty"]
    weights = RUNS_ROOT / "base" / "weights" / "last.pt"
    assert (RUNS_ROOT / "base" / "DONE").exists(), "detector gốc chưa train xong"
    model = YOLO(str(weights))
    ids = read_split("pool")
    rows, dets = [], []
    for s in range(0, len(ids), 512):
        chunk = ids[s:s + 512]
        res = model.predict([str(img_path(i)) for i in chunk], conf=u["conf_min"], iou=u["iou"],
                            imgsz=u["imgsz"], stream=True, verbose=False)
        for iid, r in zip(chunk, res):
            conf = r.boxes.conf.cpu().numpy()
            rows.append(dict(image_id=iid, n_box=len(conf),
                             n_mid=int(((conf >= u["conf_min"]) & (conf <= u["conf_max"])).sum()),
                             uncertainty=uncertainty(conf, u["conf_min"], u["conf_max"], u["empty_value"])))
            for b, c, p in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy(), conf):
                dets.append(dict(image_id=iid, cls=int(c), conf=float(p),
                                 x1=float(b[0]), y1=float(b[1]), x2=float(b[2]), y2=float(b[3])))
        print(f"{s + len(chunk)}/{len(ids)}", flush=True)
    out = RUNS_ROOT / "uncertainty"
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    assert len(df) == len(ids) and df.image_id.is_unique and df.uncertainty.between(0, 1).all()
    df.to_csv(out / "base_pool.csv", index=False)
    pd.DataFrame(dets).to_parquet(out / "base_pool_detections.parquet", index=False)
    (out / "params.json").write_text(json.dumps(dict(weights="runs/base/weights/last.pt", backend="torch-cuda",
                                                     precision="fp32", **u), indent=2))   # NFR-18
    print("->", out)


if __name__ == "__main__":
    main()
