"""G6 · Giải mù, Precision@50 từ nhãn tay, Cohen's kappa và phân xử (FR-EVL-07 → 09).

    python -m src.eval.score_blind

Nhãn cuối của một cặp:
  * người chấm chính (j1) = người chấm nhiều cặp nhất;
  * cặp có ≥ 2 người chấm mà lệch nhau -> lấy theo results/relevance/adjudicated.csv nếu có
    (cột pair_id,relevant); chưa phân xử thì giữ nhãn của j1 và ghi ra disagreements.csv.
"""
import pandas as pd

from src.common.paths import RESULTS

REL = RESULTS / "relevance"


def cohen_kappa(a, b):
    a, b = pd.Series(a).astype(int).values, pd.Series(b).astype(int).values
    po = (a == b).mean()
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return 1.0 if pe == 1 else float((po - pe) / (1 - pe))


def final_labels(labels, adjudicated=None):
    """-> (Series pair_id -> relevant, dict kappa hoặc None, DataFrame cặp lệch chưa phân xử)"""
    labels = labels.drop_duplicates(["pair_id", "judge_id"], keep="last")
    j1 = labels.judge_id.value_counts().index[0]
    main = labels[labels.judge_id == j1].set_index("pair_id").relevant.astype(int)
    wide = labels.pivot(index="pair_id", columns="judge_id", values="relevant")
    multi = wide[wide.notna().sum(axis=1) >= 2]
    disagree = multi[multi.min(axis=1) != multi.max(axis=1)].index

    kappa = None
    if labels.judge_id.nunique() >= 2:
        j2 = labels.judge_id.value_counts().index[1]
        pair = wide[[j1, j2]].dropna()
        if len(pair):
            kappa = dict(judges=f"{j1} vs {j2}", n_pairs=len(pair),
                         agree_pct=round(100 * (pair[j1] == pair[j2]).mean(), 1),
                         kappa=round(cohen_kappa(pair[j1], pair[j2]), 3))

    if adjudicated is not None and len(adjudicated):
        adj = adjudicated.drop_duplicates("pair_id", keep="last").set_index("pair_id").relevant.astype(int)
        main.update(adj)
        disagree = disagree.difference(adj.index)
    return main, kappa, pd.DataFrame({"pair_id": sorted(disagree)})


def main():
    key = pd.read_csv(REL / "blind_key.csv")
    files = sorted(REL.glob("labels_*.csv"))
    if not files:
        raise SystemExit("Chưa có results/relevance/labels_*.csv — chạy app/judge.py trước")
    labels = pd.concat([pd.read_csv(p) for p in files])
    adj_p = REL / "adjudicated.csv"
    rel, kappa, dis = final_labels(labels, pd.read_csv(adj_p) if adj_p.exists() else None)
    missing = set(key.pair_id) - set(rel.index)
    if missing:
        raise SystemExit(f"Người chấm chính chưa chấm đủ: thiếu {len(missing)} cặp")

    key["relevant"] = key.pair_id.map(rel)
    p50 = key.groupby(["query_id", "index_id"]).relevant.sum().div(50).unstack()
    p50.loc["MEAN"] = p50.mean()
    p50.to_csv(REL / "p_at_50_human.csv")
    print(p50.round(3).to_string())
    if kappa:
        pd.DataFrame([kappa]).to_csv(REL / "kappa.csv", index=False)
        print(f"\nCohen's kappa = {kappa['kappa']} trên {kappa['n_pairs']} cặp chấm đôi ({kappa['judges']})")
    if len(dis):
        dis.to_csv(REL / "disagreements.csv", index=False)
        print(f"! {len(dis)} cặp lệch chưa phân xử -> disagreements.csv (đang lấy nhãn người chấm chính)")


if __name__ == "__main__":
    main()
