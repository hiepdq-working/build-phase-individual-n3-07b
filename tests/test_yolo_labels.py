from types import SimpleNamespace

import pandas as pd
import pytest

from src.common.paths import META, YOLO_LABELS
from src.data.bdd_to_yolo import to_line


def test_to_line_clips_and_normalizes():
    r = SimpleNamespace(yolo_class=0, x1=-10.0, y1=0.0, x2=640.0, y2=360.0)
    c, cx, cy, w, h = map(float, to_line(r).split())
    assert c == 0 and (cx, cy, w, h) == (0.25, 0.25, 0.5, 0.5)


@pytest.mark.skipif(not (META / "images.parquet").exists(), reason="chưa có dữ liệu BDD100K")
def test_roundtrip_50_images():
    """FR-DAT-03: với 50 ảnh ngẫu nhiên, số dòng = số box, khôi phục box2d lệch ≤ 1 px."""
    imgs = pd.read_parquet(META / "images.parquet").sample(50, random_state=0)
    bx = pd.read_parquet(META / "boxes.parquet")
    bx = bx[bx.yolo_class >= 0]
    for iid in imgs.image_id:
        split, name = iid.split("/", 1)
        lines = [ln for ln in (YOLO_LABELS / split / name.replace(".jpg", ".txt")).read_text().splitlines() if ln]
        g = bx[bx.image_id == iid]
        assert len(lines) == len(g)
        rec = []
        for ln in lines:
            c, cx, cy, w, h = map(float, ln.split())
            assert 0 <= c <= 6
            rec.append(((cx - w / 2) * 1280, (cy - h / 2) * 720))
        exp = sorted((max(0.0, r.x1), max(0.0, r.y1)) for r in g.itertuples())
        for (x1, y1), (ex, ey) in zip(sorted(rec), exp):
            assert abs(x1 - ex) <= 1 and abs(y1 - ey) <= 1
