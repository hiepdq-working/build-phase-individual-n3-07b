"""Trang chấm mù cho người không cần code (Bước 8.2, NFR-09, NFR-31).

    streamlit run app/judge.py

Đọc results/relevance/blind_set.csv (không có tên chỉ mục hay thứ hạng), ghi labels_<alias>.csv.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
from PIL import Image  # noqa: E402

from src.common.paths import RESULTS, img_path  # noqa: E402

st.set_page_config(page_title="Chấm mù", layout="centered")
src = RESULTS / "relevance" / "blind_set.csv"
if not src.exists():
    st.error("Chưa có results/relevance/blind_set.csv — chạy python -m src.eval.export_blind trước.")
    st.stop()
bs = pd.read_csv(src)
judge = st.text_input("Alias người chấm (ví dụ j1) — không nhập tên thật hay email")
if not judge:
    st.info("Mỗi ảnh: đọc mô tả và định nghĩa, bấm Liên quan hoặc Không liên quan. "
            "**Không chắc thì chọn Không liên quan.**")
    st.stop()
out = RESULTS / "relevance" / f"labels_{judge}.csv"
done = set(pd.read_csv(out).pair_id) if out.exists() else set()
todo = bs[~bs.pair_id.isin(done)]
st.progress(len(done) / len(bs), text=f"Đã chấm {len(done)}/{len(bs)}")
if todo.empty:
    st.success("Xong, cảm ơn bạn!")
    st.stop()
r = todo.iloc[0]
st.markdown(f"### Ảnh này có khớp mô tả?\n**{r.query_text}**")
if isinstance(r.get("definition"), str) and r.definition:
    st.caption(f"Định nghĩa: {r.definition}")
try:
    st.image(Image.open(img_path(r.image_id)))
except Exception as e:
    st.warning(f"Không mở được ảnh ({e}) — chọn Không liên quan.")


def save(label):
    pd.DataFrame([dict(pair_id=int(r.pair_id), judge_id=judge, relevant=label,
                       timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"))]) \
        .to_csv(out, mode="a", header=not out.exists(), index=False)


c1, c2 = st.columns(2)
c1.button("✅ Liên quan", on_click=save, args=(1,), width="stretch")
c2.button("❌ Không liên quan", on_click=save, args=(0,), width="stretch")
