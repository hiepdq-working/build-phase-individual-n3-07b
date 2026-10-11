import numpy as np
from PIL import Image

from src.search.tiling import pad_square, tile_boxes


def test_tiles_inside_cover_and_overlap():
    """FR-IDX-01: hộp nằm trong ảnh, tile 1–6 phủ 100% ảnh, chồng lấn = 20% cạnh tile ± 1 px."""
    W, H = 1280, 720
    b = tile_boxes(W, H)
    assert len(b) == 7 and b[0] == (0, 0, W, H)
    cover = np.zeros((H, W), bool)
    for x1, y1, x2, y2 in b[1:]:
        assert 0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H
        cover[y1:y2, x1:x2] = True
    assert cover.all()
    tw, th = b[1][2] - b[1][0], b[1][3] - b[1][1]
    assert abs((b[1][2] - b[2][0]) - 0.2 * tw) <= 1          # chồng lấn ngang
    assert abs((b[1][3] - b[4][1]) - 0.2 * th) <= 1          # chồng lấn dọc


def test_pad_square_keeps_full_width():
    """OI-08: pad_square giữ 100% chiều rộng ảnh."""
    im = Image.new("RGB", (1280, 720), (255, 0, 0))
    sq = pad_square(im)
    assert sq.size == (1280, 1280)
    px = np.asarray(sq)
    assert (px[640, :, 0] == 255).all()                      # hàng giữa: đủ 1280 px đỏ
    assert (px[0, :, 0] == 0).all()                          # viền đệm đen
