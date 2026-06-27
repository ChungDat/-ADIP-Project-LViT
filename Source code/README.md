# LViT: Language meets Vision Transformer in Medical Image Segmentation

Đây là mã nguồn (được tái hiện và tùy chỉnh) cho mô hình **"LViT: Language meets Vision Transformer in Medical Image Segmentation"**.

**Tham khảo bài báo gốc tại:** 
- [ArXiv](https://arxiv.org/abs/2206.14718)
- [ResearchGate](https://www.researchgate.net/publication/371833348_LViT_Language_meets_Vision_Transformer_in_Medical_Image_Segmentation)
- [IEEEXplore](https://ieeexplore.ieee.org/document/10172039)

---

## 1. Yêu cầu hệ thống và Cài đặt

Dự án sử dụng trình quản lý gói `uv` để cài đặt và đồng bộ môi trường.

**Yêu cầu:**
- Python == 3.12
- Đã cài đặt `uv` trên hệ thống.

**Cài đặt môi trường:**
Chạy lệnh sau tại thư mục chứa file `project.toml` để đồng bộ và cài đặt các thư viện phụ thuộc:
```bash
uv sync
```

---

## 2. Chuẩn bị dữ liệu

Mô hình được huấn luyện riêng biệt trên hai tập dữ liệu: **QaTa-COV19** và **MosMedData+**.

### 2.1. Tải dữ liệu ảnh
- **QaTa-COV19 Dataset:** [Kaggle](https://www.kaggle.com/datasets/aysendegerli/qatacov19-dataset)
- **MosMedData+ Dataset:** [Kaggle](https://www.kaggle.com/datasets/maedemaftouni/covid19-ct-scan-lesion-segmentation-dataset)

### 2.2. Tải chú thích văn bản (Text Annotation)

**QaTa-COV19:**
- Train & Val: [Download Link](https://1drv.ms/x/s!AihndoV8PhTDkm5jsTw5dX_RpuRr?e=uaZq6W) (File phân chia ID train/val tải từ [đây](https://1drv.ms/f/c/c3143e7c85766728/QihndoV8PhQggMO2rwAAAAAADo5kj33mUee33g)).
- Test: [Download Link](https://1drv.ms/x/s!AihndoV8PhTDkj1vvvLt2jDCHqiM?e=954uDF)

**MosMedData+:**
- Train: [Download Link](https://1drv.ms/x/s!AihndoV8PhTDguIIKCRfYB9Z0NL8Dw?e=8rj6rY)
- Val: [Download Link](https://1drv.ms/x/c/c3143e7c85766728/QShndoV8PhQggMMGsQAAAAAAtAgZiRQFYfsAjw)
- Test: [Download Link](https://1drv.ms/x/c/c3143e7c85766728/QShndoV8PhQggMMHsQAAAAAAdHkwXMxGlgU9Tg)

### 2.3. Sắp xếp cấu trúc dữ liệu ban đầu
Giải nén dữ liệu ảnh và tải các file chú thích đặt vào thư mục `datasets/` theo cấu trúc sau:

```text
├── datasets
│   ├── Covid19
│   │   ├── move_data.py
│   │   ├── Test_text_for_Covid19.xlsx
│   │   ├── Train_ID.xlsx
│   │   ├── Train_text_for_Covid19.xlsx
│   │   └── Val_ID.xlsx
│   ├── MosMedDataPlus
│   │   ├── move_data.py
│   │   ├── Test_text_MosMedData+.xlsx
│   │   ├── Train_text_MosMedData+ 1.xlsx
│   │   └── Val_text_MosMedData+ 1.xlsx
│   ├── QaTa-COV-v2      # Thư mục giải nén ảnh QaTa-COV19
│   └── MosMedDataPlus   # Thư mục giải nén ảnh MosMedData+
```

**Xử lý dữ liệu:**
Truy cập vào các thư mục `Covid19` hoặc `MosMedDataPlus` và chạy script `move_data.py` tương ứng. Script này sẽ đọc các file Excel ID và tự động phân chia, di chuyển hình ảnh và mask (nhãn) vào đúng các thư mục con `img` và `labelcol`.

### 2.4. Cấu trúc dữ liệu hoàn chỉnh
Sau khi chạy script chuẩn bị, cấu trúc cây thư mục bên trong `datasets/` sẽ như sau để sẵn sàng đưa vào huấn luyện:

```text
├── datasets
│   ├── Covid19
│   │   ├── Test_Folder
│   │   │   ├── Test_text.xlsx
│   │   │   ├── img
│   │   │   └── labelcol
│   │   ├── Train_Folder
│   │   │   ├── Train_text.xlsx
│   │   │   ├── img
│   │   │   └── labelcol
│   │   └── Val_Folder
│   │       ├── Val_text.xlsx
│   │       ├── img
│   │       └── labelcol
│   └── MosMedDataPlus
│       ├── Test_Folder
│       │   ├── Test_text.xlsx
│       │   ├── img
│       │   └── labelcol
│       ├── Train_Folder
│       │   ├── Train_text.xlsx
│       │   ├── img
│       │   └── labelcol
│       └── Val_Folder
│           ├── Val_text.xlsx
│           ├── img
│           └── labelcol
```

---

## 3. Cấu hình Tham số (Config.py)

Trước khi tiến hành huấn luyện hoặc kiểm thử, cần cấu hình các tham số trong file `Config.py`.

- `task_name`: Tên tập dữ liệu muốn chạy (`'Covid19'` hoặc `'MosMedDataPlus'`).
- `model_name`: Tên mô hình (mặc định: `'LViT'`).
- `epochs`: Số epoch tối đa để huấn luyện.
- `batch_size`: Kích thước batch.
- `learning_rate`: Tốc độ học (Khuyến nghị: `3e-4` cho Covid19, `1e-3` cho MosMedDataPlus).
- `session_name`: Tên phiên huấn luyện (được tạo tự động theo ngày giờ, dùng để tạo thư mục lưu weights và logs).
- `test_session`: Tên thư mục phiên huấn luyện muốn kiểm thử (Ví dụ: `"Session_05.23_14h19"`). **Biến này cần được nhập thủ công khi chạy bước Đánh giá mô hình**.

---

## 4. Huấn luyện Mô hình (Training)

Sau khi thiết lập file `Config.py` và chuẩn bị dữ liệu, chạy lệnh sau để tiến hành huấn luyện:

```bash
python train_model.py
```
**Trong quá trình huấn luyện:**
- Checkpoint của mô hình tốt nhất (best model) sẽ được tự động lưu lại trong thư mục tương ứng: `[task_name]/[model_name]/[session_name]/models/`.
- File log dạng text sẽ được lưu dưới đuôi `.log` và logs đồ thị của Tensorboard sẽ lưu tại mục `[task_name]/[model_name]/[session_name]/`.

---

## 5. Đánh giá Mô hình (Testing / Evaluation)

Để đánh giá mô hình bằng tập dữ liệu Test:
1. Mở file `Config.py`, thay đổi biến `test_session` thành phiên cần đánh giá (chuỗi sinh ra từ `session_name`).
2. Chạy lệnh:

```bash
python test_model.py
```

**Kết quả thu được:**
- Các chỉ số đo lường hiệu suất (metrics) như **Dice** và **IoU** sẽ được tính toán và in ra terminal.
- Các ảnh mask phân đoạn (segmentation mask) dự đoán từ mô hình sẽ được xuất ra dưới định dạng ảnh `.jpg` và lưu vào thư mục `[task_name]_visualize_test/` tại thư mục gốc của project.