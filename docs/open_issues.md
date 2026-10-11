# Vấn đề mở (OI) của SRD v0.3 — trạng thái

Cột **Đề xuất** là lựa chọn trong checklist làm một mình, đã ghi vào `configs/decisions.yaml` hoặc vào code.
Được đổi **trước** tag `lock-v1`. Cột **Trạng thái** điền khi freeze (AC-09): `đã chốt`, `ghi thành hạn chế`, hoặc `không áp dụng`.

| OI | Vấn đề | Đề xuất (làm một mình) | Nơi ghi | Trạng thái |
|---|---|---|---|---|
| OI-01 | n_box = 0 thì uncertainty = ? | Giữ 0 (ảnh detector không thấy gì xếp cuối) | `uncertainty.empty_value` | |
| OI-02 | Checkpoint chấm; val trong data.yaml | Chấm `last.pt`, không early stopping; khóa val của Ultralytics = split `dev` (1.000 ảnh train ngoài base/pool) | `train.checkpoint_rule`, `sizes.dev` | |
| OI-03 | arm_random/meta rút 1 lần cho 3 seed | Giữ; std chỉ phản ánh nhiễu huấn luyện → ghi hạn chế | Báo cáo | |
| OI-04 | Base detector 1 seed | Giữ; uncertainty và arm_tool phụ thuộc 1 lần train → ghi hạn chế | Báo cáo | |
| OI-05 | Định dạng lưu trữ | Parquet cho meta/lookup (không commit); CSV cho mọi kết quả commit; YAML cho config; txt cho split; npy + faiss cho chỉ mục | Repo | |
| OI-06 | Chấm mù: gộp, số câu, số người | Gộp top-50 của mọi chỉ mục, bỏ trùng, xáo; 5 câu eval_human; j1 chấm hết, j2 ≥ 20% | `export_blind.py` | |
| OI-07 | Timing nhiễu vì người khác nhau | 2 người, mỗi người làm cả 2 chế độ theo thứ tự ngược nhau; mình không tham gia | `docs/plan.md` Bước 8 | |
| OI-08 | Preprocess | `pad_square` (giữ 100% chiều rộng) | `clip.preprocess` | |
| OI-09 | Embedding ở đâu; lệch backend | Embedding trên Colab (CUDA fp32), truy vấn trên Mac (CPU); đo cosine bằng `check_index` và ghi báo cáo | `docs/pilot.md` | |
| OI-10 | Backend/precision của G8 | Colab CUDA fp32, ghi trong `results/uncertainty/params.json` | `run_pool.py` | |
| OI-11 | Phá hòa khi chọn 500 ảnh | uncertainty ↓, rồi điểm CLIP ↓, rồi image_id ↑ | `uncertainty.tie_break`, `tool_select.py` | |
| OI-12 | Luật chọn model chính | P@50 trung bình của 2 tiling cao hơn; hòa hoặc thiếu chỉ mục → ViT-B-32; chỉ dùng P@50 tự động | `clip.main_model_rule` | |
| OI-13 | Tiling của tool_select | `whole_tiles` | `clip.tool_select_tiling` | |
| OI-14 | Luật kết luận | "Tốt hơn" nếu mean(tool − random) > max(std) và thắng 3/3 seed; "có xu hướng" nếu ≥ 2/3; ngược lại "chưa có bằng chứng" | `conclusion_rule`, `aggregate.verdict` | |
| OI-15 | val2k giao rare slice? | Được giao; số ảnh trùng được báo cáo | `val2k_may_overlap_rare` | |
| OI-16 | is_rare; tập ảnh duyệt tay; tool có lọc? | is_rare = night ∧ (rainy ∨ snowy) ∧ ≥ 1 pedestrian; duyệt tay = pool theo thứ tự ngẫu nhiên cố định; tool **không** có bộ lọc metadata, chỉ gõ câu | `gt_rules.is_rare`, `app.py` | |
| OI-17 | Truy vấn ảnh: nguyên ảnh hay 7 tile | Nguyên ảnh, pad_square | `engine.encode_image` | |
| OI-18 | Chọn model bằng metadata pool | Ghi thành hạn chế | Báo cáo | |
| OI-19 | Ai giữ P3 | Làm một mình: giữ toàn bộ | — | không áp dụng |
| OI-20 | Research-only: chia sẻ ảnh/weights, CVAT | Không chia sẻ ra ngoài; Drive chỉ chia sẻ người chấm; bỏ CVAT | README | |
| OI-21 | epochs, batch, thời gian 1 run | Chốt sau pilot: batch 16, E theo công thức | `docs/pilot.md`, `train.epochs` | |
| OI-22 | Ảnh thiếu JSON | Loại khỏi mọi split, số lượng ghi lại | `missing_label_json` | |
| OI-23 | "Queued" ở Mốc 2 | Có dòng `todo` kèm ngày dự kiến trong `runs_registry.csv` | `results/runs_registry.csv` | |
| OI-24 | 20 truy vấn eval commit trước G6? | Có: commit `queries.yaml` + `gt_rules.py` trước lần chạy `eval_search` đầu tiên | Git log | |
| OI-25 | "Hơn một nửa chưa bắt đầu" | ≥ 5/9 run vào thứ Năm W2 | `docs/plan.md` | |
| OI-26 | N mặc định trong app | 50 | `app.py` | |
| OI-27 | Ngưỡng "(đề xuất)" | Giữ nguyên các ngưỡng đề xuất của SRD; ghi số đo thật vào `docs/pilot.md`, `docs/app_perf.md` | — | |
| OI-28 | Backend text encoder trong app | PyTorch CPU trên Mac; tool_select cũng chạy trên Mac nên cùng backend | `engine.py` | |
| OI-29 | Timing và chấm mù: app hay Sheet | Trong app (`app.py` chế độ Timing, `judge.py`) | `app/` | |
| OI-30 | SigLIP không xong trước hạn chọn model | Không lùi tool_select; bảng 2×2 thành 1×2, ghi hạn chế | `clip.main_model_rule` | |
| OI-31 | boxes.parquet: 7 hay 10 lớp | Giữ cả 10 lớp, lớp ngoài class_map có `yolo_class = -1` | `keep_all_classes_in_boxes` | |
