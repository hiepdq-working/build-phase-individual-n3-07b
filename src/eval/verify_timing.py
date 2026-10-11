"""Timing · Xác minh mọi claim bằng cùng một hàm is_rare (FR-EVL-16 → 18) — con số 2.

    python -m src.eval.verify_timing
"""
import pandas as pd

from src.common.config import load_decisions
from src.common.paths import RESULTS

TIMING = RESULTS / "timing"


def summarize(claims, sessions, rare, limit_s):
    c = claims[claims.t_seconds <= limit_s].copy()
    c["verified"] = c.image_id.isin(rare)
    c["verified_id"] = c.image_id.where(c.verified)          # nunique bỏ qua NaN
    s = c.groupby(["session_id", "participant", "mode"]).agg(
        n_claims=("image_id", "size"),
        n_verified_unique=("verified_id", "nunique"),
        n_wrong=("verified", lambda v: int((~v).sum()))).reset_index()
    if sessions is not None and len(sessions):
        ev = sessions.pivot_table(index="session_id", columns="event", values="timestamp", aggfunc="first")
        if {"start", "end"} <= set(ev.columns):
            dur = (pd.to_datetime(ev["end"]) - pd.to_datetime(ev["start"])).dt.total_seconds()
            s["duration_s"] = s.session_id.map(dur)
    return s


def main():
    from src.eval.gt_rules import is_rare, load_table

    t = load_table()
    rare = set(t.index[is_rare(t)])
    limit = load_decisions()["timing"]["minutes"] * 60
    sess_p = TIMING / "sessions.csv"
    s = summarize(pd.read_csv(TIMING / "claims.csv"), pd.read_csv(sess_p) if sess_p.exists() else None,
                  rare, limit)
    s.to_csv(TIMING / "summary.csv", index=False)
    print(s.to_string(index=False))
    print("\nSố ca hiếm verified (không trùng) trung bình mỗi phiên:")
    print(s.groupby("mode")[["n_verified_unique", "n_wrong"]].mean().round(2))


if __name__ == "__main__":
    main()
