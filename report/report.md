# N3-07B · Data Engine — Báo cáo

> Viết dần từ W2. Phần **Kết quả** chỉ điền từ CSV trong `results/`, không gõ số từ trí nhớ.
> Mỗi con số ghi kèm file nguồn. Commit tag `freeze`: `_____`.

## 1. Bài toán
- Vì sao công ty xe tự hành cần tìm ca hiếm: phần lớn ảnh là ban ngày, đường thoáng; cảnh khó (người đi bộ ban đêm dưới mưa) hiếm và là chỗ detector hay sai.
- Câu hỏi của thí nghiệm: cùng thêm 500 ảnh có nhãn, ảnh do tool chọn có giúp detector phát hiện người đi bộ ban đêm dưới mưa/tuyết tốt hơn ảnh chọn ngẫu nhiên không?

## 2. Dữ liệu
- BDD100K (Yu et al., CVPR 2020), **giấy phép research-only** (NFR-41). Nguồn, ngày tải, SHA-256: `docs/data_sources.md`.
- 5 split + 3 arm, số ảnh và SHA-256: `splits/registry.csv`. Ảnh thiếu JSON bị loại (OI-22).
- Rare slice: mức fallback _____, _____ ảnh, _____ box pedestrian.
- Thống kê thuộc tính: `results/data_stats.csv`.

## 3. Phương pháp
- CLIP OpenCLIP ViT-B-32 (LAION-2B), preprocess `pad_square`, 7 crop mỗi ảnh (nguyên ảnh + lưới 3×2 chồng lấn 20%). Điểm ảnh = max qua tile và prompt. Chỉ mục FAISS IndexFlatIP (tìm chính xác).
- Embedding trên Colab (CUDA fp32), truy vấn trên Mac (CPU); cosine lệch giữa hai môi trường: _____ (`docs/pilot.md`).
- Uncertainty = #box conf ∈ [0,2; 0,5] / #box conf ≥ 0,2, tính bằng detector gốc.
- Quy tắc chọn arm_tool: 1.500 ảnh CLIP cao nhất → 500 ảnh uncertainty cao nhất. Model chính: `configs/main_model.yaml`.

## 4. Thiết kế thí nghiệm
- 3 arm (random, meta, tool) × 3 seed; mọi run 5.500 ảnh, cùng hyperparameter (`configs/decisions.yaml`), chấm `last.pt`.
- Quyết định khóa trước: tag `lock-v1` ngày _____ (commit `_____`), sớm hơn run base.
- Luật kết luận (khóa trước, NFR-12): xem `conclusion_rule` trong `decisions.yaml`.

## 5. Kết quả

### Con số 1 — Precision@50
- Tự động, 20 câu × 2 chỉ mục: `results/search/p_at_50_auto_table.csv`. MEAN whole _____ · whole_tiles _____.
- Riêng 7 câu vật nhỏ/cảnh tối (A01, A15–A20): whole _____ · whole_tiles _____.
- Câu dưới 50 ảnh dương (không thể đạt 1,0): `results/search/positives_per_query.csv`.
- Chấm mù, 5 câu: `results/relevance/p_at_50_human.csv`. Kappa _____ trên _____ cặp (`kappa.csv`).

### Con số 2 — Ca hiếm tìm được trong 15 phút
- `results/timing/summary.csv`: tool _____ ca verified (claim sai _____) · duyệt tay _____ (claim sai _____).
- Ghi chú nhiễu: 2 người, thứ tự đảo (OI-07). Đây là bằng chứng sơ bộ.

### Con số 3 — AP50 pedestrian trên rare slice
- `results/detector_scores.csv`, `report/figures/arms_chart.png`.

| Arm | seed 0 | seed 1 | seed 2 | mean ± std | mAP50 val2k |
|---|---|---|---|---|---|
| base (5.000) | | — | — | | |
| random | | | | | |
| meta | | | | | |
| tool | | | | | |

- Hiệu tool − random theo seed: `results/detector/tool_minus_random.csv`.
- **Kết luận theo luật đã khóa:** `results/detector/verdict.txt` → _____
- Thành phần 3 arm: `results/select/arm_composition.csv`.

## 6. Hạn chế
- Chỉ 1 lát cắt, 1 dataset (đường phố Mỹ); dùng nhãn có sẵn thay cho gán nhãn thật.
- Base detector chỉ 1 seed (OI-04); arm_random và arm_meta chỉ rút 1 lần (OI-03).
- Timing chỉ có 2 người (OI-07). Chọn model chính dựa trên metadata của pool (OI-18).
- Embedding trên Colab nhưng truy vấn trên Mac (OI-09). Dự án làm một mình.
- Các OI khác: `docs/open_issues.md`.

## 7. Hướng mở rộng
Chọn ảnh qua nhiều vòng (active learning), mở rộng sang video và LiDAR, gán nhãn thật qua CVAT.

## Giấy phép
Ảnh minh họa từ BDD100K (Yu et al., CVPR 2020), chỉ dùng cho nghiên cứu. Ảnh, embedding và weights không được phân phối lại.
