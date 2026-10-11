# Nguồn dữ liệu

> Điền ở Bước 0.3 và Bước 1. Mọi số ở đây chép từ output của script, không gõ từ trí nhớ.

## BDD100K

| Mục | Giá trị |
|---|---|
| Nguồn | https://bdd-data.berkeley.edu/ (mirror: https://dl.cv.ethz.ch/bdd100k/data/) |
| Giấy phép | BDD100K License — **chỉ dùng cho nghiên cứu**, không phân phối lại |
| Ngày tải | _____ |
| File | `100k_images_train.zip`, `100k_images_val.zip`, `bdd100k_det_20_labels_trainval.zip` |
| Dung lượng | _____ |
| SHA-256 | xem `docs/bdd100k.sha256` |
| Số ảnh đếm được | train _____ / 70.000 · val _____ / 10.000 |
| Bản nhãn | det_20 (lớp người đi bộ tên `pedestrian`; bản cũ `person` được đổi qua `label_aliases`) |

Trích dẫn: Yu et al., *BDD100K: A Diverse Driving Dataset for Heterogeneous Multitask Learning*, CVPR 2020.

## Bước 1.2 — build_meta

```
(dán output của python -m src.data.build_meta)
```

- Ảnh thiếu JSON (OI-22, bị loại): train _____ · val _____

## Bước 1.4 — make_splits

```
(dán output của python -m src.data.make_splits)
```

| Mục | Giá trị |
|---|---|
| Pool: số ảnh night ∧ (rainy ∨ snowy) | _____ (cần ≥ 1.000) |
| Rare slice: mức fallback | _____ |
| Rare slice: số ảnh / số box pedestrian | _____ / _____ |
| val2k trùng rare slice (OI-15) | _____ |
