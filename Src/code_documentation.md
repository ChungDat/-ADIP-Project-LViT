# Tài liệu Mã nguồn Dự án LViT

## Mục lục
1. [Cấu hình & Các kịch bản chạy chính (Configuration & Execution Scripts)](#1-cấu-hình--các-kịch-bản-chạy-chính-configuration--execution-scripts)
2. [Tải dữ liệu & Tiền xử lý (Data Loading & Preprocessing)](#2-tải-dữ-liệu--tiền-xử-lý-data-loading--preprocessing)
3. [Kiến trúc mô hình (Model Architecture - nets)](#3-kiến-trúc-mô-hình-model-architecture---nets)
4. [Các hàm tiện ích & Lớp hàm mất mát (Utility Functions & Loss Classes)](#4-các-hàm-tiện-ích--lớp-hàm-mất-mát-utility-functions--loss-classes)

---

## 1. Cấu hình & Các kịch bản chạy chính (Configuration & Execution Scripts)

### Config.py
* **Mục đích**: Là tệp cấu hình trung tâm định nghĩa các siêu tham số (hyperparameters), đường dẫn thư mục tập dữ liệu, chế độ huấn luyện và cấu trúc mạng Transformer.
* **Các hàm**:
  - `get_CTranS_config()`: Định nghĩa và trả về một đối tượng cấu hình (sử dụng `ml_collections.ConfigDict`) để thiết lập số lượng đầu chú ý (attention heads), số lượng lớp (layers), kích thước MLP, dropout, kích thước các phân mảnh ảnh (patch sizes), và số lượng kênh cơ bản (base channels) cho mạng Transformer và U-Net.

---

### train_model.py
* **Mục đích**: Kịch bản chạy chính để huấn luyện mô hình phân vùng LViT dưới cả chế độ học có giám sát (supervised) và bán giám sát (semi-supervised).
* **Các hàm**:
  - `logger_config(log_path)`: Cấu hình hệ thống ghi log (logger) để ghi thông báo đồng thời ra một tệp văn bản và bảng điều khiển (console).
  - `save_checkpoint(state, save_path)`: Lưu các tham số mô hình hiện tại (trạng thái bộ tối ưu hóa, số epoch, trọng số mô hình) vào một tệp PyTorch (`.pth.tar`). Các cấu hình tối ưu nhất sẽ được lưu riêng thành `best_model-...`.
  - `worker_init_fn(worker_id)`: Hàm khởi tạo seed cho các tiến trình tải dữ liệu (data loader workers) nhằm đảm bảo tính tái lập (reproducibility) khi áp dụng tăng cường dữ liệu ngẫu nhiên.
  - `main_loop(batch_size, model_type, tensorboard)`: Điều phối quá trình huấn luyện: thiết lập tập dữ liệu, xử lý phân tách tập dữ liệu bán giám sát, xây dựng mạng LViT, tính toán và theo dõi FLOPS/parameters, chạy qua các epoch, lưu checkpoint tốt nhất trên tập validation và quản lý dừng sớm (early stopping).

---

### Train_one_epoch.py
* **Mục đích**: Điều phối vòng lặp huấn luyện và đánh giá trên các lô dữ liệu (batches) của một epoch.
* **Các hàm**:
  - `print_summary(...)`: Định dạng và ghi nhận các thông số đo lường (loss trung bình, IoU, Dice, tốc độ học hiện tại, thời gian chạy) vào logger.
  - `train_one_epoch(...)`: Duyệt qua các batch để tính toán dự đoán và tối ưu hóa tham số mô hình. Trong chế độ bán giám sát, hàm này sẽ nạp thêm dữ liệu không nhãn, tạo ra nhãn giả (pseudo-labels) mục tiêu thông qua bộ nhớ lưu trữ trung bình trượt (EMA Memory Bank), đồng thời tính toán hàm mất mát bán giám sát Dice/BCE cùng với ràng buộc tương đồng Ngôn ngữ - Thị giác (Language-Vision loss), sau đó trả về các độ đo trung bình.

---

### test_model.py
* **Mục đích**: Tải trọng số mô hình đã huấn luyện để thực hiện suy luận (inference) trên tập kiểm thử (test set), tính toán điểm đánh giá tổng quát và xuất kết quả phân vùng dưới dạng hình ảnh trực quan.
* **Các hàm**:
  - `show_image_with_dice(predict_save, labs, save_path)`: Đo lường độ tương đồng giữa nhãn dự đoán và nhãn gốc thông qua hệ số Dice và IoU, thay đổi kích thước ảnh dự đoán về độ phân giải gốc đối với tập dữ liệu MoNuSeg, ghi ảnh mặt nạ phân vùng (mask) ra đĩa bằng thư viện OpenCV và trả về các chỉ số đo lường.
  - `vis_and_save_heatmap(model, input_img, text, img_RGB, labs, vis_save_path, dice_pred, dice_ens)`: Thiết lập mô hình ở chế độ đánh giá, chạy mô hình để lấy dự đoán, áp dụng ngưỡng (>0.5) để chuyển đổi đầu ra thành nhãn nhị phân và gọi hàm `show_image_with_dice` để ghi kết quả cũng như tính toán độ đo.

---

### get_exact_val_metrics.py
* **Mục đích**: Kịch bản bổ trợ để đo lường chính xác các thông số loss, IoU và Dice của mô hình trên tập validation cho từng tác vụ và phiên làm việc cụ thể.
* **Các hàm**:
  - `get_val_metrics(task_name, session_name)`: Thiết lập đường dẫn động, khởi tạo bộ tải dữ liệu validation, xây dựng mô hình LViT, tải trọng số tốt nhất từ checkpoint, thực hiện chạy suy luận qua từng lô dữ liệu, tổng hợp các chỉ số hiệu năng và in ra kết quả chi tiết.

---

### run_evaluation.py
* **Mục đích**: Tự động chạy đánh giá trên nhiều cấu hình khác nhau, phân tích các tệp log và tổng hợp kết quả vào một báo cáo chung.
* **Các hàm**:
  - `modify_config(task_name, test_session)`: Sử dụng biểu thức chính quy (regular expressions) để chỉnh sửa tạm thời tên tác vụ và chế độ kiểm thử bên trong tệp `Config.py`.
  - `parse_log(log_path)`: Phân tích tệp log huấn luyện để trích xuất các giá trị loss, IoU và Dice tốt nhất đạt được ở epoch tối ưu.
  - Luồng chính: Tạo bản sao lưu của `Config.py`, thực hiện đánh giá cho tất cả các tổ hợp tập dữ liệu (Covid19, MosMedDataPlus) và chế độ học (Supervised, Semi_supervised) bằng cách gọi kịch bản `test_model.py`, phân tích log thu được, khôi phục lại cấu hình gốc và ghi bảng tổng hợp kết quả vào tệp `evaluation_summary.md`.

---

## 2. Tải dữ liệu & Tiền xử lý (Data Loading & Preprocessing)

### Load_Dataset.py
* **Mục đích**: Tải ảnh, nhãn gốc và các mô tả văn bản đi kèm để đưa vào mô hình, bao gồm cả việc trích xuất vector đặc trưng văn bản thông qua mô hình ngôn ngữ lớn đã được huấn luyện trước.
* **Các lớp & Hàm**:
  - Lớp `BertEmbedding`: Khởi tạo bộ mã hóa từ và mô hình BERT (`bert-base-uncased`) để chuyển đổi các câu mô tả y khoa dạng văn bản thô thành các ma trận nhúng đặc trưng (embedding vectors).
  - Hàm `random_rot_flip(image, label)`: Áp dụng ngẫu nhiên phép quay 90 độ và lật (ngang/dọc) đồng thời cho cả hình ảnh y tế và nhãn phân vùng tương ứng nhằm đồng bộ dữ liệu.
  - Hàm `random_rotate(image, label)`: Xoay ngẫu nhiên hình ảnh và nhãn y tế một góc trong khoảng từ -20 đến 20 độ.
  - Lớp `RandomGenerator`: Trình bao bọc (wrapper) xử lý chuyển đổi dữ liệu huấn luyện, áp dụng các phép xoay ngẫu nhiên, lật, thu phóng kích thước và chuyển đổi dữ liệu sang dạng tensor của PyTorch.
  - Lớp `ValGenerator`: Trình xử lý dữ liệu validation giúp chuẩn hóa kích thước hình ảnh thông qua phép nội suy thu phóng (zoom) mà không áp dụng các phép tăng cường ngẫu nhiên.
  - Hàm `to_long_tensor(pic)`: Chuyển đổi một hình ảnh hoặc mảng NumPy đầu vào thành định dạng `LongTensor` của PyTorch.
  - Hàm `correct_dims(*images)`: Chuẩn hóa số chiều dữ liệu bằng cách thêm chiều kênh (channel dimension) cho các đầu vào dạng 2D.
  - Lớp `LV2D`: Lớp Dataset của PyTorch tải các mặt nạ mục tiêu và trích xuất câu mô tả văn bản, đồng thời chuyển đổi chúng sang dạng nhúng BERT.
  - Lớp `ImageToImage2D`: Bộ phân tích và quản lý dữ liệu chính cung cấp các bộ dữ liệu dạng từ điển gồm bộ ba: ảnh - nhãn - đặc trưng văn bản cho quá trình huấn luyện và kiểm thử.

---

### move_data.py (Covid19) (trong thư mục datasets/Covid19)
* **Mục đích**: Tổ chức lại các tệp tin từ tập dữ liệu thô `QaTa-COV19-v2` thành cấu trúc chuẩn của dự án (`Train_Folder`, `Val_Folder`, `Test_Folder`) dựa trên danh sách ID trong các tệp Excel, đồng thời lọc các mô tả văn bản tương ứng.

---

### move_data.py (MosMedDataPlus) (trong thư mục datasets/MosMedDataPlus)
* **Mục đích**: Tự động nhóm các lát cắt ảnh CT 2D và nhãn phân vùng từ thư mục thô `archive` vào cấu trúc thư mục huấn luyện của dự án LViT theo các bảng Excel quy định phân chia tập train, val và test.

---

## 3. Kiến trúc mô hình (Model Architecture - nets)

### nets/LViT.py
* **Mục đích**: Triển khai kiến trúc chính của mô hình **LViT**, kết hợp xử lý đặc trưng ngôn ngữ đa phương thức (multimodal language processing) với các hoạt động của bộ mã hóa-giải mã CNN tiêu chuẩn và các khối Vision Transformer (ViT).
* **Các lớp**:
  - Lớp `Flatten`: Tiện ích làm phẳng các tensor nhiều chiều thành dạng ma trận 2D.
  - Lớp `UpblockAttention`: Nhánh giải mã (decoder) thực hiện tăng kích thước đặc trưng (upsampling), tinh chỉnh các kết nối tắt (skip connections) từ nhánh mã hóa thông qua mô-đun chú ý cấp pixel tương tác ngôn ngữ (`PixLevelModule`), kết hợp đầu ra và áp dụng tích chập kép.
  - Lớp `LViT`: Tích hợp các lớp tích chập của U-Net với các khối Vision Transformer chạy song song trên nhiều mức độ phân giải khác nhau. Đặc trưng văn bản được chiếu và hòa trộn một cách thích ứng với các phân mảnh ảnh nhúng (image patch embeddings), trong khi các khối tái cấu trúc (reconstruct blocks) chuyển đổi các chuỗi token từ Transformer trở lại không gian tích chập 2D nhằm định hướng kết quả phân vùng cuối cùng.

---

### nets/UNet.py
* **Mục đích**: Triển khai kiến trúc mạng tích chập U-Net cổ điển phục vụ so sánh hiệu năng.
* **Các lớp & Hàm**:
  - Hàm `get_activation(activation_type)`: Trích xuất một mô-đun kích hoạt từ thư viện `torch.nn` bằng chuỗi tên cấu hình (ví dụ: 'ReLU', 'GELU').
  - Hàm `_make_nConv(in_channels, out_channels, nb_Conv, activation)`: Khởi tạo một chuỗi liên kết gồm `nb_Conv` khối `ConvBatchNorm` liên tiếp.
  - Lớp `ConvBatchNorm`: Khối xây dựng cơ bản thực hiện tích chập 2D, chuẩn hóa Batch Normalization 2D và áp dụng hàm kích hoạt.
  - Lớp `DownBlock`: Khối mã hóa kết hợp phép trích xuất Max Pooling (hệ số 2) và chuỗi tích chập kép để giảm độ phân giải không gian.
  - Lớp `UpBlock`: Khối giải mã áp dụng tích chập chuyển vị (ConvTranspose2d) để tăng kích thước đặc trưng, nối các đặc trưng kết nối tắt từ nhánh mã hóa và áp dụng các lớp tích chập.
  - Lớp `UNet`: Kiến trúc U-Net hoàn chỉnh với 4 cấp độ giảm mẫu ở nhánh mã hóa và 4 cấp độ tăng mẫu tương ứng ở nhánh giải mã.

---

### nets/Vit.py
* **Mục đích**: Định nghĩa các lớp Vision Transformer và các lớp chuyển đổi kênh phục vụ cho thiết kế nhánh xử lý song song CNN-ViT.
* **Các lớp**:
  - Lớp `Reconstruct`: Chuyển đổi các token đặc trưng dạng chuỗi 1D trở lại dạng bản đồ không gian 2D bằng cách tái cấu trúc kích thước tensor, nội suy tăng mẫu (upsampling) và áp dụng tích chập.
  - Lớp `Embeddings`: Chia hình ảnh đầu vào thành các phân mảnh 2D bằng tích chập 2D, cộng thêm các vector vị trí học được và áp dụng dropout.
  - Lớp `MLP`: Khối mạng perceptron đa tầng chứa hai lớp tuyến tính liên tiếp xen kẽ hàm kích hoạt GELU và dropout.
  - Lớp `Attention`: Cơ chế tự chú ý đa đầu (Multi-head Self-Attention) giúp mô hình hóa các mối quan hệ phụ thuộc không gian toàn cục trong ảnh.
  - Lớp `Block`: Một khối mã hóa Transformer tiêu chuẩn kết hợp Layer Normalization, cơ chế tự chú ý đa đầu, DropPath (Stochastic Depth) và MLP.
  - Lớp `ConvTransBN`: Lớp chuyển đổi adapter chứa tích chập 1D, Batch Normalization 1D và ReLU nhằm căn chỉnh số kênh của các chuỗi đặc trưng.
  - Lớp `VisionTransformer`: Điều phối quá trình ánh xạ embeddings, tích hợp thông tin mô tả văn bản vào chuỗi đặc trưng ở cấp phân giải cao nhất, chạy qua các khối mã hóa và xử lý nối chuỗi token trong giai đoạn giải mã.

---

### nets/pixlevel.py
* **Mục đích**: Triển khai cơ chế chú ý cấp pixel (pixel-level attention) giúp nâng cao chất lượng đặc trưng y khoa dưới sự định hướng của văn bản y học.
* **Các lớp**:
  - Lớp `PixLevelModule`: Trích xuất các giá trị thống kê đặc trưng dọc theo chiều kênh (giá trị trung bình và giá trị lớn nhất). Hàm này nối các thống kê đó cùng tổng của chúng, đi qua một mạng MLP cổ chai (bottleneck) để xây dựng bản đồ trọng số chú ý đơn kênh, từ đó nhân tỷ lệ với đầu vào giúp mạng tập trung vào các vùng ảnh quan trọng.

---

## 4. Các hàm tiện ích & Lớp hàm mất mát (Utility Functions & Loss Classes)

### utils.py
* **Mục đích**: Tập hợp các hàm tiện ích tính toán độ đo, định nghĩa các hàm mất mát cho bài toán phân vùng ảnh, các bộ điều chỉnh tốc độ học và trình đọc văn bản.
* **Các lớp & Hàm**:
  - Lớp `WeightedBCE`: Tính toán hàm mất mát Binary Cross Entropy có trọng số nhằm cân bằng sự chênh lệch giữa các pixel nền và pixel vùng tổn thương.
  - Lớp `WeightedDiceLoss`: Đo lường mức độ chồng lấp giữa dự đoán và nhãn gốc với các tham số trọng số đi kèm.
  - Lớp `BinaryDiceLoss`: Hàm mất mát Dice tiêu chuẩn dành cho bài toán phân vùng nhị phân.
  - Lớp `MultiClassDiceLoss`: Tính toán trung bình mất mát Dice trên 5 phân lớp dữ liệu khác nhau.
  - Lớp `DiceLoss`: Mở rộng tính toán Dice cho bài toán phân loại đa lớp sử dụng biểu diễn một nóng (one-hot).
  - Lớp `WeightedDiceCE`: Kết hợp hàm mất mát DiceLoss và Cross-Entropy để tối ưu hóa trong phân loại đa lớp.
  - Lớp `WeightedDiceBCE`: Kết hợp DiceLoss có trọng số và BCELoss có trọng số cho các tác vụ phân vùng nhị phân.
  - Hàm `auc_on_batch(masks, pred)`: Tính toán diện tích dưới đường cong ROC (AUC) trung bình theo từng lô dữ liệu.
  - Hàm `iou_on_batch(masks, pred)`: Tính toán chỉ số giao trên tập hợp (IoU / Jaccard Index) trên từng lô dữ liệu sau khi đã áp ngưỡng nhị phân hóa.
  - Hàm `dice_coef(y_true, y_pred)`: Tính toán hệ số Dice giữa hai mảng NumPy đã được làm phẳng.
  - Hàm `dice_on_batch(masks, pred)`: Tính toán hệ số Dice trung bình theo từng lô dữ liệu.
  - Hàm `save_on_batch(images1, masks, pred, names, vis_path)`: Ghi ảnh dự đoán và nhãn gốc tương ứng ra đĩa phục vụ đánh giá trực quan.
  - Lớp `_LRScheduler`: Lớp cơ sở tự định nghĩa để lập lịch điều chỉnh tốc độ học.
  - Lớp `CosineAnnealingWarmRestarts`: Triển khai cơ chế điều chỉnh tốc độ học theo chu kỳ Cosine (Cosine Annealing) có khởi động lại ấm (warm restarts).
  - Hàm `read_text(filename)`: Đọc câu mô tả từ tệp Excel, tự động đệm thêm từ khóa `EOF XXX` cho các câu ngắn (tối đa 9 từ) và trả về một từ điển ánh xạ từ tên ảnh sang câu mô tả.
  - Hàm `read_text_LV(filename)`: Tương tự như `read_text` nhưng đệm độ dài câu lớn hơn (tối đa 30 từ) phục vụ cho căn chỉnh Language-Vision.
  - Hàm `get_thum(image, size, greyscale)`: Thay đổi kích thước ảnh về kích thước thu nhỏ (thumbnail).
  - Hàm `img_similarity_vectors_via_numpy(image1, image2)`: Tính độ tương đồng cosine giữa vector biểu diễn của hai hình ảnh.
  - Lớp `MemoryBankEPI`: Bộ nhớ đệm lưu trữ và cập nhật nhãn giả cho dữ liệu không nhãn trong huấn luyện bán giám sát bằng cơ chế trung bình trượt lũy thừa (EMA).
  - Lớp `ModelEPI`: Tạo nhãn giả thông qua mô hình giáo viên (teacher model) được cập nhật trọng số động dạng EMA từ mô hình học sinh (student model).
  - Lớp `WeightedDiceBCE_unsup`: Tính toán mất mát Dice và BCE có trọng số giữa dự đoán của mô hình với nhãn giả dạng liên tục trên tập không nhãn.
  - Hàm `LV_loss_unsup(preds_unlabeled, texts_unlabeled, labels_labeled, texts_labeled)`: Tính toán mất mát căn chỉnh đặc trưng Ngôn ngữ - Thị giác trên tập dữ liệu không nhãn bằng cách truy vấn mẫu có nhãn sở hữu câu mô tả tương đồng nhất, sau đó tính khoảng cách cosine giữa dự đoán của mẫu không nhãn với nhãn gốc của mẫu có nhãn đó.
