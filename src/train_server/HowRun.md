# Tạo môi trường ảo
python -m venv .venv

# Kích hoạt môi trường ảo
# Trên Windows (On Windows):
.venv\Scripts\activate
# Trên Linux/Mac (On Linux/Mac):
source .venv/bin/activate

# Cài đặt các package cần thiết (Install required packages)
pip install torch transformers datasets pandas scikit-learn openpyxl
pip install 'accelerate>=0.26.0'
pip install psutil gputil

# Lưu danh sách package đã cài (Save installed packages list)
pip freeze > requirements.txt


# Run: 
```bash
CUDA_VISIBLE_DEVICES=0 python train.py
```



====



# Train ngay cả khi tắt Terminal: 
```bash
# Cài đặt tmux
sudo apt-get install tmux

# Tạo session mới
tmux new -s training

# Run train
CUDA_VISIBLE_DEVICES=0 python train.py

# Bước 4: Kiểm tra session vẫn chạy
tmux ls

# Bước 5: Khi muốn xem lại training
tmux attach -t training
```




Đây là hướng dẫn chi tiết sử dụng tmux:

1. Cài đặt tmux:
```bash
sudo apt-get install tmux
```

2. Các lệnh cơ bản:
```bash
# Tạo session mới
tmux new -s training

# Trong session, chạy training:
CUDA_VISIBLE_DEVICES=0 python train.py

# Thoát session (detach) nhưng vẫn giữ process chạy:
# Nhấn Ctrl + B, sau đó nhấn D

# Xem danh sách sessions đang có
tmux ls

# Quay lại session cũ
tmux attach -t training

# Xóa session
tmux kill-session -t training
```

3. Các phím tắt quan trọng (trong tmux):
```
Ctrl + B, sau đó:
    d           -> detach (thoát session)
    [           -> scroll mode (dùng PageUp/Down để cuộn)
    c           -> tạo window mới
    n           -> next window
    p           -> previous window
    %           -> chia màn hình dọc
    "           -> chia màn hình ngang
    mũi tên     -> di chuyển giữa các panel
    x           -> đóng panel hiện tại
    ?           -> xem tất cả phím tắt
    q           -> thoát scroll mode
```

4. Ví dụ workflow hoàn chỉnh:
```bash
# Bước 1: Tạo session mới
tmux new -s training

# Bước 2: Trong session, kích hoạt môi trường và chạy training
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=0 python train.py

# Bước 3: Detach session
# Nhấn Ctrl + B, sau đó D

# Bước 4: Kiểm tra session vẫn chạy
tmux ls

# Bước 5: Khi muốn xem lại training
tmux attach -t training

# Bước 6: Trong session, xem log
# Nhấn Ctrl + B, sau đó [ để vào scroll mode
# Dùng PageUp/Down để cuộn
# Nhấn q để thoát scroll mode
```

Tips:
1. Bạn có thể chia màn hình để vừa chạy training vừa monitor:
```bash
# Trong tmux:
# Chia đôi màn hình: Ctrl + B, %
# Di chuyển qua panel khác: Ctrl + B, mũi tên
# Panel 1: chạy training
# Panel 2: htop hoặc nvidia-smi -l 1
```

2. Đặt tên có ý nghĩa cho session để dễ quản lý:
```bash
tmux new -s training_bert_v1
tmux new -s training_roberta_v2
```
