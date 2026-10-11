"""G11 · Tổng hợp 9 run: mean ± std, hiệu tool − random theo seed, luật kết luận, biểu đồ
(FR-EVL-13 → 15; NFR-11, 12) — con số 3.

    python -m src.eval.aggregate

Chỉ chạy khi cả 9 run đã xong (quy tắc 4 của Kế hoạch). Kết luận in ra là kết luận viết vào báo cáo,
kể cả khi tool không thắng.
"""
import pandas as pd

from src.common.paths import REPORT, RESULTS

ARMS = ["random", "meta", "tool"]


def verdict(ap):
    """Luật conclusion_rule đã khóa trong decisions.yaml. ap: DataFrame index=seed, cột = arm."""
    common = ap.dropna(subset=["random", "tool"])
    diff = common["tool"] - common["random"]
    thr = max(ap["tool"].std(ddof=1), ap["random"].std(ddof=1))
    wins, n = int((diff > 0).sum()), len(diff)
    if n and diff.mean() > thr and wins == n:
        v = "Tool tốt hơn random"
    elif n and diff.mean() > 0 and wins >= n - 1:
        v = "Có xu hướng tool tốt hơn"
    else:
        v = "Chưa có bằng chứng tool tốt hơn random"
    return diff, v, thr, wins, n


def summarize(df):
    """df: detector_eval.csv -> (bảng scores, bảng arms dài, Series base theo metric)."""
    df = df[df.run_id != "pilot"].sort_values("evaluated_at").drop_duplicates(["run_id", "metric"], keep="last")
    base = df[df.run_id == "base"].set_index("metric").value
    arms = df[df.run_id != "base"].copy()
    arms["arm"] = arms.run_id.str.rsplit("_s", n=1).str[0]
    arms["seed"] = arms.run_id.str.rsplit("_s", n=1).str[1].astype(int)
    rows = []
    for (arm, metric), g in arms.groupby(["arm", "metric"]):
        g = g.sort_values("seed")
        r = dict(arm=arm, metric=metric, mean=g.value.mean(), std=g.value.std(ddof=1), n_seeds=len(g))
        r.update({f"seed{s}": v for s, v in zip(g.seed, g.value)})
        rows.append(r)
    for metric, v in base.items():
        rows.append(dict(arm="base", metric=metric, mean=v, n_seeds=1))
    return pd.DataFrame(rows), arms, base


def plot(scores, arms, base, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (metric, title) in zip(axes, [("AP50_pedestrian", "AP50 pedestrian — rare slice"),
                                          ("mAP50", "mAP50 — val2k")]):
        s = scores[(scores.metric == metric) & scores.arm.isin(ARMS)].set_index("arm").reindex(ARMS)
        ax.bar(ARMS, s["mean"], yerr=s["std"], capsize=6, color=["#9aa5b1", "#5b8def", "#e8833a"])
        for i, arm in enumerate(ARMS):
            pts = arms[(arms.metric == metric) & (arms.arm == arm)].value
            ax.scatter([i] * len(pts), pts, color="black", s=12, zorder=3)
        ax.set_xticks(range(len(ARMS)), [f"{a}\n{s.loc[a, 'mean']:.3f} ± {s.loc[a, 'std']:.3f}"
                                         if pd.notna(s.loc[a, "mean"]) else a for a in ARMS])
        if metric in base:
            ax.axhline(base[metric], ls="--", color="gray", label=f"base (5.000 ảnh) {base[metric]:.3f}")
            ax.legend(loc="lower right")
        lo = min(s["mean"].min(), base.get(metric, 1)) * 0.9
        ax.set_ylim(lo, None)
        ax.set_title(title)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)


def main():
    df = pd.read_csv(RESULTS / "detector" / "detector_eval.csv")
    scores, arms, base = summarize(df)
    print(f"Số run arm đã chấm: {arms.run_id.nunique()}/9")
    scores.to_csv(RESULTS / "detector_scores.csv", index=False)
    print(scores.round(4).to_string(index=False))

    ap = arms[arms.metric == "AP50_pedestrian"].pivot(index="seed", columns="arm", values="value")
    diff, v, thr, wins, n = verdict(ap)
    pd.DataFrame({"seed": diff.index, "tool_minus_random": diff.values}).to_csv(
        RESULTS / "detector" / "tool_minus_random.csv", index=False)
    msg = f"Kết luận: {v} (mean hiệu {diff.mean():.4f}, ngưỡng {thr:.4f}, thắng {wins}/{n} seed)"
    (RESULTS / "detector" / "verdict.txt").write_text(msg + "\n")
    print(f"\nHiệu tool − random theo seed: {diff.round(4).to_dict()}")
    print(msg)
    plot(scores, arms, base, REPORT / "figures" / "arms_chart.png")
    print("->", REPORT / "figures" / "arms_chart.png")


if __name__ == "__main__":
    main()
