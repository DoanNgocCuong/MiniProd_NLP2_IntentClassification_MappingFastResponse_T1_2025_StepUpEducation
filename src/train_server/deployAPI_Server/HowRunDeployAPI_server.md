Lỗi bạn gặp phải là:

```
ERROR:    [Errno 98] Address already in use
```


**Chạy ứng dụng trên cổng khác**:
   - Nếu bạn không muốn dừng ứng dụng khác, bạn có thể chạy ứng dụng FastAPI trên một cổng khác. Ví dụ, để chạy trên cổng 8001, bạn có thể sử dụng lệnh sau:
     ```bash
     uvicorn api_export:app --reload --port 8001
     ```


** Bình thường khi mình chưa xài SSH-WSL, mình thường deploy lên server bằng cách run: 'docker compose up -d' (sudo vào cổng nào đó, xong chạy)**
Giờ vào SSH-WSL cũng tương tự. 




Nếu bạn đang ở trong máy chủ và muốn chạy ứng dụng FastAPI của mình, bạn có thể làm theo các bước sau:

### Bước 1: Chạy ứng dụng FastAPI

Sử dụng lệnh sau để chạy ứng dụng FastAPI:

```bash
uvicorn api_export:app --host 0.0.0.0 --port 25041 --reload
```

- `--host 0.0.0.0`: Điều này cho phép ứng dụng lắng nghe trên tất cả các địa chỉ IP, vì vậy bạn có thể truy cập nó từ bên ngoài máy chủ.
- `--port 25041`: Đây là cổng mà ứng dụng sẽ chạy.
- `--reload`: Tùy chọn này cho phép tự động tải lại ứng dụng khi có thay đổi trong mã nguồn (chỉ nên sử dụng trong môi trường phát triển).

### Bước 2: Kiểm tra ứng dụng

Sau khi chạy lệnh trên, bạn có thể mở trình duyệt và truy cập vào địa chỉ:

```
http://<địa_chỉ_IP_của_máy_chủ>:25041/docs
```

Thay `<địa_chỉ_IP_của_máy_chủ>` bằng địa chỉ IP thực tế của máy chủ của bạn. Bạn sẽ thấy giao diện Swagger UI, nơi bạn có thể kiểm tra các endpoint của API.

### Bước 3: Chạy ứng dụng trong nền (tùy chọn)

Nếu bạn muốn ứng dụng chạy trong nền và không bị dừng khi bạn thoát khỏi SSH, bạn có thể sử dụng `nohup` hoặc `screen`:

- **Sử dụng `nohup`**:
   ```bash
   nohup uvicorn api_export:app --host 0.0.0.0 --port 25041 &
   ```

- **Sử dụng `screen`**:
   ```bash
   screen
   uvicorn api_export:app --host 0.0.0.0 --port 25041
   ```

   Sau đó, bạn có thể nhấn `Ctrl + A`, sau đó `D` để thoát khỏi phiên `screen` mà không dừng ứng dụng.



---

API test 