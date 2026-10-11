"""Demo N3-07B Data Engine — chạy offline trên Mac (Bước 7).

    streamlit run app/app.py

4 chế độ:
  * Tìm kiếm            — văn bản (1–5 prompt) hoặc ảnh, lọc metadata, uncertainty, xuất CSV (FR-APP-01 → 12)
  * So sánh whole/tiles — hai cột cùng truy vấn, đánh dấu ảnh chung (FR-APP-10)
  * Timing — duyệt tay  — thứ tự ngẫu nhiên cố định, không ô truy vấn (FR-APP-14)
  * Timing — dùng tool  — chỉ gõ câu, ẩn metadata, khóa ghi claim sau 15 phút (FR-APP-13, 15, 16)
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
from PIL import Image  # noqa: E402

from src.common.config import load_decisions  # noqa: E402
from src.common.paths import INDEX_ROOT, META, RESULTS, img_path  # noqa: E402
from src.data.splits_io import read_split  # noqa: E402
from src.search.engine import META_KEYS, SearchIndex  # noqa: E402

st.set_page_config(page_title="N3-07B Data Engine", layout="wide")
CFG = load_decisions()
LIMIT = CFG["timing"]["minutes"] * 60
PAGE = CFG["timing"]["page_size"]
ss = st.session_state


@st.cache_resource(show_spinner="Đang nạp chỉ mục…")
def get_index(index_id):
    return SearchIndex(index_id)


@st.cache_data
def list_indexes():
    """Chỉ cho chọn chỉ mục có meta.json hợp lệ và đủ file (FR-APP-04)."""
    ok, bad = [], []
    if not INDEX_ROOT.exists():
        return ok, bad
    for d in sorted(p for p in INDEX_ROOT.iterdir() if p.is_dir() and not p.name.startswith("_")):
        try:
            meta = json.loads((d / "meta.json").read_text())
        except Exception:
            bad.append((d.name, "không đọc được meta.json"))
            continue
        miss = [k for k in META_KEYS if k not in meta]
        miss += [f for f in ["vectors.npy", "index.faiss", "lookup.parquet"] if not (d / f).exists()]
        (bad.append((d.name, f"thiếu {miss}")) if miss else ok.append(d.name))
    return ok, bad


@st.cache_data
def tables():
    imgs = pd.read_parquet(META / "images.parquet").set_index("image_id")
    p = RESULTS / "uncertainty" / "base_pool.csv"
    unc = pd.read_csv(p).set_index("image_id").uncertainty if p.exists() else pd.Series(dtype=float)
    return imgs, unc, read_split("pool")


@st.cache_data(max_entries=3000)
def thumb(image_id, w=360):
    try:
        im = Image.open(img_path(image_id)).convert("RGB")
    except Exception:                                          # NFR-33: ảnh mất không làm sập app
        return None
    im.thumbnail((w, w))
    return im


def fmt(x, f="{:.2f}"):
    return "—" if x is None or pd.isna(x) else f.format(x)


# ---------------------------------------------------------------- timing
def log_row(name, row):
    p = RESULTS / "timing" / name
    p.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(p, mode="a", header=not p.exists(), index=False)


def claim(image_id):
    if "t0" not in ss:
        return
    el = time.time() - ss.t0
    if el > LIMIT:                                             # FR-APP-16
        return
    log_row("claims.csv", dict(session_id=ss.sid, participant=ss.who, mode=ss.tmode, image_id=image_id,
                               t_seconds=round(el, 1), timestamp=time.strftime("%Y-%m-%dT%H:%M:%S")))
    ss.claimed.add(image_id)


def timing_header(tmode):
    if ss.get("tmode") != tmode:
        ss.pop("t0", None)
    if "t0" not in ss:
        st.markdown("**Ca hiếm** = ảnh ban đêm, trời mưa hoặc tuyết, có ít nhất một người đi bộ. "
                    "Bấm **Hiếm ✓** dưới mỗi ảnh bạn cho là đúng. Phiên dài 15 phút.")
        who = st.text_input("Alias người tham gia (ví dụ p1) — không nhập tên thật hay email")
        if st.button("Bắt đầu 15 phút", type="primary", disabled=not who):
            ss.t0, ss.who, ss.tmode, ss.claimed = time.time(), who, tmode, set()
            ss.sid = f"{tmode}-{who}-{int(ss.t0)}"
            ss.pop("q", None)                                  # không mang truy vấn cũ vào phiên
            log_row("sessions.csv", dict(session_id=ss.sid, participant=who, mode=tmode, event="start",
                                         timestamp=time.strftime("%Y-%m-%dT%H:%M:%S")))
            st.rerun()
        st.stop()
    el = time.time() - ss.t0
    c1, c2 = st.columns([4, 1])
    left = max(0, LIMIT - el)
    c1.info(f"Đã qua {int(el // 60)}:{int(el % 60):02d} / {LIMIT // 60}:00 — còn {int(left // 60)}:"
            f"{int(left % 60):02d} — đã đánh dấu {len(ss.claimed)} ảnh "
            "(đồng hồ cập nhật khi bạn thao tác)")
    if el > LIMIT:
        c1.warning("Hết giờ — không ghi thêm được. Bấm Kết thúc phiên.")
    if c2.button("Kết thúc phiên"):
        log_row("sessions.csv", dict(session_id=ss.sid, participant=ss.who, mode=tmode, event="end",
                                     timestamp=time.strftime("%Y-%m-%dT%H:%M:%S")))
        ss.pop("t0")
        st.rerun()
    return el <= LIMIT


# ---------------------------------------------------------------- tìm kiếm
def query_ui(ix, allow_image=True):
    tabs = st.tabs(["Tìm bằng văn bản", "Tìm bằng ảnh"] if allow_image else ["Tìm bằng văn bản"])
    with tabs[0]:
        txt = st.text_area("1–5 prompt, mỗi dòng một prompt (tiếng Anh)", key="txt",
                           placeholder="a pedestrian at night in the rain")
        if st.button("Tìm", key="go_t", type="primary"):
            try:
                ss.q, ss.qlabel = ix.encode_text(txt.splitlines()), txt.strip().replace("\n", " | ")
            except Exception as e:
                st.error(f"Truy vấn không hợp lệ: {e}")
    if allow_image:
        with tabs[1]:
            up = st.file_uploader("Tải ảnh lên", type=["jpg", "jpeg", "png"])
            iid = st.text_input("hoặc nhập image_id trong pool (ví dụ train/xxxx.jpg)")
            if st.button("Tìm bằng ảnh", key="go_i"):
                try:
                    im = Image.open(up) if up else Image.open(img_path(iid.strip()))
                    ss.q, ss.qlabel = ix.encode_image(im), f"image:{up.name if up else iid.strip()}"
                except Exception as e:
                    st.error(f"Không đọc được ảnh: {e}")


def grid(hits, unc, timing=False, enabled=True, mark=None, ncol=5):
    cols = st.columns(ncol)
    for i, r in enumerate(hits.itertuples()):
        with cols[i % ncol]:
            im = thumb(r.image_id)
            if im is None:
                st.warning(f"Không mở được ảnh {r.image_id}")
            else:
                st.image(im)
            if timing:
                done = r.image_id in ss.claimed
                st.button("Đã đánh dấu" if done else "Hiếm ✓", key=f"c_{r.image_id}",
                          disabled=done or not enabled, on_click=claim, args=(r.image_id,))
            else:
                star = "★ " if mark and r.image_id in mark else ""
                st.caption(f"{star}#{r.rank} · {r.image_id} · score {r.score:.3f} · "
                           f"unc {fmt(unc.get(r.image_id))}")


def main():
    ok, bad = list_indexes()
    for name, why in bad:
        st.sidebar.warning(f"Bỏ qua {name}: {why}")          # FR-APP-04
    if not META.joinpath("images.parquet").exists():
        st.error("Thiếu data/meta/images.parquet — chạy Bước 1 trong README.")
        st.stop()
    if not ok:
        st.error(f"Không có chỉ mục hợp lệ trong {INDEX_ROOT} — xem README Bước 2.")
        st.stop()

    mode = st.sidebar.radio("Chế độ", ["Tìm kiếm", "So sánh whole / tiles", "Timing — duyệt tay",
                                       "Timing — dùng tool"])
    timing = mode.startswith("Timing")
    imgs, unc, pool = tables()

    if mode == "Timing — duyệt tay":                           # FR-APP-14
        st.title("Timing — duyệt tay")
        enabled = timing_header("browse")
        order = np.random.default_rng(CFG["seeds"]["timing_browse"]).permutation(sorted(pool))
        page = st.number_input("Trang", 1, len(order) // PAGE, 1)
        grid(pd.DataFrame({"image_id": order[(page - 1) * PAGE: page * PAGE], "rank": 0, "score": 0.0}),
             unc, timing=True, enabled=enabled)
        return

    index_id = st.sidebar.selectbox("Chỉ mục", ok)
    try:
        ix = get_index(index_id)
    except Exception as e:
        st.error(f"Không nạp được chỉ mục {index_id}: {e}")
        st.stop()
    st.sidebar.json({k: ix.meta[k] for k in META_KEYS})      # FR-APP-05
    n = st.sidebar.slider("Số ảnh N", 10, 100, 50, 10)
    allowed = set(pool)
    filters = {}
    if not timing:                                           # FR-APP-13: timing ẩn metadata
        for col in ["weather", "scene", "timeofday"]:        # FR-APP-06
            v = st.sidebar.multiselect(col, sorted(imgs[col].unique()))
            if v:
                filters[col] = v
        sub = imgs.loc[imgs.index.isin(allowed)]
        for col, v in filters.items():
            sub = sub[sub[col].isin(v)]
        allowed = set(sub.index)
    by_unc = (not timing) and st.sidebar.toggle("Sắp top-N theo uncertainty")

    st.title(mode)
    enabled = timing_header("tool") if timing else True
    query_ui(ix, allow_image=not timing)
    if "q" not in ss:
        st.stop()
    if ss.q.shape[1] != ix.V.shape[1]:
        st.warning("Truy vấn được mã hóa bằng model khác — bấm Tìm lại.")
        st.stop()

    t0 = time.time()
    hits = ix.search(ss.q, n=n, allowed=allowed)
    st.caption(f"Truy vấn: {ss.qlabel} · {len(hits)} ảnh · {time.time() - t0:.2f} s"
               + ("" if timing else f" · {len(allowed)} ảnh thỏa bộ lọc"))
    hits["uncertainty"] = hits.image_id.map(unc)
    if by_unc:                                               # FR-APP-08: cùng tập ảnh, đổi thứ tự
        hits = hits.sort_values(["uncertainty", "rank"], ascending=[False, True])

    if mode == "So sánh whole / tiles":                      # FR-APP-10
        prefix = index_id.split("_whole")[0]
        pair = [f"{prefix}_whole", f"{prefix}_whole_tiles"]
        if not all(p in ok for p in pair):
            st.error(f"Cần cả hai chỉ mục {pair}")
            st.stop()
        res = [get_index(p).search(ss.q, n=n, allowed=allowed) for p in pair]
        common = set(res[0].image_id) & set(res[1].image_id)
        st.write(f"**{len(common)} / {n} ảnh chung** (đánh dấu ★)")
        c1, c2 = st.columns(2)
        for c, p, h in zip([c1, c2], pair, res):
            with c:
                st.subheader(p)
                grid(h, unc, mark=common, ncol=2)
        return

    grid(hits, unc, timing=timing, enabled=enabled)
    if not timing:                                           # FR-APP-11
        out = hits.assign(index_id=index_id, query=ss.qlabel, filters=json.dumps(filters, ensure_ascii=False))
        cols = ["image_id", "rank", "score", "best_tile", "uncertainty", "index_id", "query", "filters"]
        st.download_button("Xuất CSV", out[cols].to_csv(index=False), "ket_qua.csv", "text/csv")
        pick = st.selectbox("Xem chi tiết ảnh", hits.image_id)
        if pick:
            try:
                st.image(Image.open(img_path(pick)))
            except Exception as e:
                st.warning(f"Không mở được ảnh: {e}")
            st.json({**imgs.loc[pick][["weather", "scene", "timeofday"]].to_dict(),
                     "best_tile": int(hits.set_index("image_id").loc[pick, "best_tile"]),
                     "uncertainty": fmt(unc.get(pick), "{:.3f}")})


main()
