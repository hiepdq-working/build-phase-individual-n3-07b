# Slide — kịch bản 6 phút

| Thời gian | Nội dung | Nguồn |
|---|---|---|
| 0:00–0:45 | Vấn đề: đội xe thu hàng triệu ảnh, ca hiếm nằm lẫn trong đó; lướt tay 15 phút chỉ được bao nhiêu ca | `results/timing/summary.csv` (duyệt tay) |
| 0:45–2:30 | **Demo trực tiếp:** gõ "a pedestrian at night in the rain", lọc metadata, bật sắp theo uncertainty, rồi thả ảnh một ca detector bỏ sót để tìm ca giống | `app/app.py` |
| 2:30–3:15 | Chế độ so sánh whole / tiles, kèm Precision@50 | `results/search/p_at_50_auto_table.csv` |
| 3:15–5:00 | Khép vòng: biểu đồ 3 arm, rồi ảnh trước/sau (detector nhánh tool bắt được người đi bộ, base bỏ sót) | `report/figures/arms_chart.png` |
| 5:00–6:00 | Kết luận theo luật đã khóa, hạn chế, hướng mở rộng (nhiều vòng, video, LiDAR) | `results/detector/verdict.txt` |

- Slide cuối: *Ảnh: BDD100K (Yu et al., CVPR 2020) — research-only.*
- Có video dự phòng (Cmd+Shift+5) đủ 6 tính năng demo, lưu trên Drive, chỉ chia sẻ với người chấm.
- Trước giờ demo: chạy app một lượt với các câu sẽ dùng để model và ảnh nằm sẵn trong cache. Tập 1 lần tắt Wi-Fi.
