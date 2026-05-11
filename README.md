# ♟️ BTL Chess AI

Chương trình cờ vua tích hợp trí tuệ nhân tạo, xây dựng bằng Python.  
Hỗ trợ **2 chế độ AI**: Minimax (không cần GPU) và Neural Network (CNN).

---

## Yêu cầu

- Python **3.9+**
- macOS / Windows / Linux

---

## Cài đặt

**1. Clone về máy**

```bash
git clone https://github.com/TTheDuyx-145/BTL_Chessbot-ai.git
cd BTL_Chessbot-ai
```

**2. Tạo môi trường ảo**

```bash
python -m venv venv
```

Kích hoạt:

```bash
# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

**3. Cài thư viện**

```bash
pip install -r requirements.txt
```

---

## Chạy chương trình

### Chế độ 1 — Minimax AI *(khuyên dùng, không cần TensorFlow)*

```bash
cd gui
python main_minimax.py
```

### Chế độ 2 — Neural Network AI *(cần TensorFlow)*

```bash
cd gui
python main.py
```

---

## Cách chơi

| Thao tác | Hành động |
|----------|-----------|
| Click vào quân | Chọn quân muốn đi |
| Click vào ô đích | Thực hiện nước đi |
| Icon **⇄** (góc phải) | Đổi bên chơi |
| Icon **↺** (góc phải) | Reset ván mới |

> Mặc định: bạn chơi **Trắng**, AI chơi **Đen**.  
> Tốt đến hàng cuối sẽ **tự động phong Hậu**.

---

## Điều chỉnh độ khó

Mở file `gui/main_minimax.py`, tìm dòng:

```python
ai_black = MinimaxPlayer(colour="black", depth=3, time_limit=5.0)
```

| Depth | Thời gian/nước | ELO ước tính |
|-------|---------------|--------------|
| 3 | ~0.5s | ~1000–1200 |
| 4 | ~2s | ~1200–1400 |
| 5 | ~10s | ~1400–1600 |

---

## Cấu trúc dự án

```
BTL_Chessbot-ai/
├── engine/          # Minimax AI (Alpha-Beta, Quiescence Search...)
├── gui/             # Giao diện Pygame + logic người chơi
│   ├── models/      # Weights Neural Network (700 / 1100 / 1200 ELO)
│   └── images/      # Hình ảnh quân cờ
├── train/           # Code huấn luyện Neural Network
├── data_cleaning/   # Xử lý dữ liệu PGN từ Lichess
├── uci.py           # Giao thức UCI (kết nối Arena, Lichess BOT...)
└── requirements.txt
```

---

## Thư viện sử dụng

| Thư viện | Mục đích |
|----------|----------|
| `python-chess` | Luật cờ vua, FEN, UCI |
| `pygame` | Giao diện đồ họa |
| `tensorflow` | Neural Network (chế độ 2) |
| `numpy` | Xử lý ma trận |

---

## Tác giả

**Trần Duy** — ttheduy1401@gmail.com
