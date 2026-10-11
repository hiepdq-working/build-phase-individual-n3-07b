# Hiệu năng app trên Mac (Bước 7.2, NFR-19 → 22)

Máy: MacBook M2, RAM _____ GB, macOS _____ · Chỉ mục nạp: `vitb32_whole`, `vitb32_whole_tiles`

| Mục | Ngưỡng (đề xuất) | Đo được |
|---|---|---|
| Truy vấn văn bản top-50 trên 280.000 vector, p95 (20 truy vấn) | ≤ 2 s | |
| Truy vấn ảnh (tính cả mã hóa), p95 (10 ảnh) | ≤ 5 s | |
| Khởi động nguội, nạp 2 chỉ mục (3 lần) | ≤ 60 s | |
| RAM khi đã nạp 2 chỉ mục (Activity Monitor) | ≤ 8 GB | |

## 5 ca lỗi (NFR-33)

| Ca | Kết quả |
|---|---|
| Ô truy vấn rỗng | ☐ báo lỗi, không sập |
| Tải file không phải ảnh | ☐ |
| image_id sai | ☐ |
| Đổi tên tạm meta.json của một chỉ mục | ☐ chỉ mục bị loại khỏi danh sách |
| 6 dòng prompt | ☐ |

## Offline (FR-APP-12)

☐ Tắt Wi-Fi, khởi động lại app: tìm bằng câu và bằng ảnh đều chạy.
