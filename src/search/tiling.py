"""G1 · Cắt ô (FR-IDX-01) và preprocess pad_square (OI-08).

Ảnh 1280×720 chia 3 cột × 2 hàng chồng lấn 20% cho ô khoảng 492×400 px.
Mỗi crop được đệm thành hình vuông trước khi đưa vào CLIP, nên CLIP không cắt mất hai bên ảnh.
"""
from PIL import Image


def tile_boxes(W, H, cols=3, rows=2, overlap=0.2):
    """Trả 7 hộp (x1, y1, x2, y2): hộp 0 là ảnh nguyên, 1..6 là lưới có chồng lấn."""
    tw = W / (cols - (cols - 1) * overlap)
    th = H / (rows - (rows - 1) * overlap)
    boxes = [(0, 0, W, H)]
    for r in range(rows):
        for c in range(cols):
            x0, y0 = c * tw * (1 - overlap), r * th * (1 - overlap)
            boxes.append((round(x0), round(y0), min(W, round(x0 + tw)), min(H, round(y0 + th))))
    return boxes


def pad_square(im: Image.Image) -> Image.Image:
    s = max(im.size)
    canvas = Image.new("RGB", (s, s), (0, 0, 0))
    canvas.paste(im, ((s - im.width) // 2, (s - im.height) // 2))
    return canvas
