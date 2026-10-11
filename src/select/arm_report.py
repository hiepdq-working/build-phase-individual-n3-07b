"""G9 · Bảng thành phần và độ trùng của 3 arm (FR-SEL-09).

    python -m src.select.arm_report

Lưu ý: không được sửa prompt tool sau khi xem bảng này (NFR-04).
"""
import pandas as pd

from src.common.paths import RESULTS
from src.data.splits_io import read_split
from src.eval.gt_rules import load_table


def main():
    t = load_table()
    arms = {a: set(read_split(f"arm_{a}")) for a in ["random", "meta", "tool"]}
    rows = []
    for a, s in arms.items():
        x = t.loc[sorted(s)]
        rows.append(dict(arm=a, n=len(x), pct_night=(x.timeofday == "night").mean(),
                         pct_rain_snow=x.weather.isin(["rainy", "snowy"]).mean(),
                         n_pedestrian_boxes=int(x.pedestrian.sum()),
                         pct_with_pedestrian=(x.pedestrian >= 1).mean(),
                         overlap_with_tool=len(s & arms["tool"]),
                         overlap_with_random=len(s & arms["random"])))
    df = pd.DataFrame(rows)
    (RESULTS / "select").mkdir(exist_ok=True)
    df.to_csv(RESULTS / "select" / "arm_composition.csv", index=False)
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
