# Hướng dẫn chấm mù (Bước 8)

Đọc hết trang này trước khi bắt đầu (khoảng 5 phút). Chạy `streamlit run app/judge.py`, nhập **alias** (ví dụ `j1`), không nhập tên thật hay email.

## Quy tắc chung

1. Mỗi ảnh: đọc **mô tả** và **định nghĩa** hiện trên trang, rồi bấm *Liên quan* hoặc *Không liên quan*.
2. **Không chắc thì chọn Không liên quan.** Đối tượng phải nhìn thấy được ở cỡ ảnh bình thường, không phải đoán.
3. Chấm từng ảnh độc lập. Không bàn với người chấm khác trong lúc chấm.
4. Trang không hiện tên cấu hình hay thứ hạng; thứ tự ảnh đã được xáo. Đừng cố đoán ảnh đến từ đâu.
5. Ảnh hỏng, không mở được → Không liên quan.
6. Nghỉ 5 phút sau mỗi khoảng 100 ảnh. Có thể đóng trang và mở lại; trang nhớ chỗ đang chấm.
7. Chấm xong, nhắn xác nhận: *"Tôi không thấy tên cấu hình hay thứ hạng khi chấm"* (NFR-09).

## Ví dụ biên chung

| Tình huống | Quyết định |
|---|---|
| Đối tượng ở rất xa, chỉ vài pixel, chỉ đoán được | Không liên quan |
| Đối tượng bị che hơn khoảng 70% | Không liên quan |
| Chỉ thấy qua phản chiếu (gương, kính xe) | Không liên quan |
| Kính chắn gió nhòe mưa nhưng vẫn nhận ra đối tượng | Liên quan |

## Từng câu

### H01 · a road construction zone with cones
Thấy công trường trên đường: cọc tiêu, hàng rào chắn hoặc biển công trường chiếm một phần làn đường.
- Một cọc tiêu lẻ ở lề, không có dấu hiệu công trường → Không liên quan.
- Hàng cọc tiêu chuyển làn xe → Liên quan.

### H02 · strong headlight glare at night
Đèn pha xe đối diện gây lóa mạnh, che khuất một phần ảnh.
- Đèn đường sáng nhưng không gây lóa → Không liên quan.
- Vệt lóa lớn quanh đèn xe đối diện làm mất chi tiết → Liên quan.

### H03 · a pedestrian holding an umbrella
Có ít nhất một người đang cầm/che ô nhìn thấy được.
- Ô che nắng của quán, không có người cầm → Không liên quan.
- Người đi xe máy/xe đạp cầm ô → Liên quan. Người cầm ô ở bất kỳ kích thước nào đều tính, miễn thấy rõ ô.

### H04 · a road covered in snow at night
Ảnh ban đêm và mặt đường hoặc lề đường có tuyết phủ nhìn thấy rõ.
- Đường chỉ ướt, phản chiếu ánh đèn → Không liên quan.
- Ảnh ban ngày có tuyết → Không liên quan.

### H05 · a person walking a dog
Có người đi bộ dắt chó (thấy cả người và chó).
- Chó đi một mình → Không liên quan.
- Người và chó cạnh nhau, không thấy dây nhưng rõ là đi cùng → Liên quan.

## Phân xử (người làm dự án)

`python -m src.eval.score_blind` ghi các cặp hai người chấm lệch nhau vào `results/relevance/disagreements.csv`. Xem lại theo hướng dẫn này, rồi ghi `results/relevance/adjudicated.csv` (`pair_id,relevant`) và chạy lại lệnh. Không sửa file `labels_*.csv` của người chấm.

| Cohen's κ | Diễn giải (Landis & Koch) |
|---|---|
| < 0,40 | Yếu → nêu trong báo cáo |
| 0,40 – 0,60 | Trung bình |
| 0,61 – 0,80 | Khá |
| > 0,80 | Rất tốt |
