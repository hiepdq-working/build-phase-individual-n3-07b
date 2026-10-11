# Pilot và số đo hiệu năng

## Embedding (Bước 2.3, NFR-23/24)

| Model | GPU | crop/s (`--limit 200`) | Ước tính 280.000 crop | Thực tế |
|---|---|---|---|---|
| ViT-B-32 | T4 | | | |
| ViT-B-16-SigLIP-384 (nếu làm) | T4 | | | |

Cosine Mac (CPU, fp32) vs Colab (`check_index`, NFR-17 / OI-09): min _____ · mean _____

## Pilot huấn luyện (Bước 4.3, NFR-26, OI-21)

| Mục | Giá trị |
|---|---|
| Ngày | |
| GPU | |
| ultralytics | |
| Thời gian 1 epoch trên 1.000 ảnh (T_pilot) | |
| Ước tính 1 epoch trên 5.500 ảnh (T = T_pilot × 5,5) | |
| Số epoch chốt `E = floor(90 / T)`, bội 5, ≥ 10 | |
| Thử ngắt ở epoch 2 → resume chạy tiếp đến epoch 3 (FR-TRN-09) | ☐ đạt |

## Chấm lại một run (Bước 6.2, NFR-16)

| Run | AP50 lần 1 | AP50 lần 2 | Lệch (≤ 0,002) |
|---|---|---|---|
| | | | |
