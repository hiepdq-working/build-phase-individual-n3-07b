"""G6 · Luật đáp án tự động cho 20 truy vấn eval_auto (FR-EVL-01) và hàm is_rare (FR-EVL-18).

Commit file này cùng configs/queries.yaml TRƯỚC khi chạy eval_search (OI-24).
Trước khi commit, kiểm tên giá trị thật trong dữ liệu (ví dụ 'parking lot', 'dawn/dusk'):
    python -c "import pandas as pd; t = pd.read_parquet('data/meta/images.parquet'); \
               print(t.scene.value_counts(), t.timeofday.value_counts())"
"""
import pandas as pd

from src.common.paths import META

CLASSES = ["pedestrian", "rider", "car", "truck", "bus", "bicycle", "motorcycle"]
SMALL_PED_PX = 40   # "người ở xa" = box pedestrian cao dưới 40 px (Kế hoạch chi tiết)


def build_table(imgs, bx):
    """Mỗi ảnh một dòng: metadata + số box theo lớp + có người nhỏ hay không."""
    t = imgs.set_index("image_id")
    cnt = bx.groupby(["image_id", "category"]).size().unstack(fill_value=0)
    t = t.join(cnt)
    for c in CLASSES:
        if c not in t:
            t[c] = 0
    t[CLASSES] = t[CLASSES].fillna(0).astype(int)
    small = bx[(bx.category == "pedestrian") & ((bx.y2 - bx.y1) < SMALL_PED_PX)].image_id.unique()
    t["small_ped"] = t.index.isin(small)
    return t


def load_table():
    return build_table(pd.read_parquet(META / "images.parquet"), pd.read_parquet(META / "boxes.parquet"))


RULES = {
    "night": lambda t: t.timeofday == "night",
    "rainy": lambda t: t.weather == "rainy",
    "snowy": lambda t: t.weather == "snowy",
    "foggy": lambda t: t.weather == "foggy",
    "overcast": lambda t: t.weather == "overcast",
    "tunnel": lambda t: t.scene == "tunnel",
    "highway": lambda t: t.scene == "highway",
    "residential": lambda t: t.scene == "residential",
    "parking_lot": lambda t: t.scene == "parking lot",
    "dawn_dusk": lambda t: t.timeofday == "dawn/dusk",
    "bus": lambda t: t.bus >= 1,
    "trucks3": lambda t: t.truck >= 3,
    "cars15": lambda t: t.car >= 15,
    "no_vehicle": lambda t: (t.car + t.truck + t.bus) == 0,
    "small_ped": lambda t: t.small_ped,
    "peds5": lambda t: t.pedestrian >= 5,
    "cyclist": lambda t: (t.rider >= 1) & (t.bicycle >= 1),
    "motorcycle": lambda t: t.motorcycle >= 1,
    "night_ped": lambda t: (t.timeofday == "night") & (t.pedestrian >= 1),
    "night_rain": lambda t: (t.timeofday == "night") & (t.weather == "rainy"),
}


def positives(rule, table, within=None):
    t = table if within is None else table.loc[table.index.isin(within)]
    return set(t.index[RULES[rule](t)])


def is_rare(table):
    """Định nghĩa ca hiếm dùng chung cho cả hai chế độ đo thời gian (FR-EVL-18, OI-16):
    ban đêm, trời mưa hoặc tuyết, có ít nhất một người đi bộ."""
    return (table.timeofday == "night") & table.weather.isin(["rainy", "snowy"]) & (table.pedestrian >= 1)
