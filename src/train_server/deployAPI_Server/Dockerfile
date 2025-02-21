# Sử dụng image chính thức cho FastAPI
FROM tiangolo/uvicorn-gunicorn-fastapi:python3.10

# Đặt thư mục làm việc
WORKDIR /app

# Sao chép file requirements.txt và cài đặt các phụ thuộc - nếu ko thay đổi gì file requirements.txt thì sẽ được cache
COPY requirements.txt ./
# RUN pip install --no-cache-dir -r requirements.txt
RUN pip install -r requirements.txt

# Sao chép mã nguồn vào container
COPY ./api_export.py ./

# Sao chép mô hình từ checkpoint-1288 vào container
COPY ./results/v5_trainsets_checkpoint-140_XLMRoBERTa_10eps ./results/v5_trainsets_checkpoint-140_XLMRoBERTa_10eps
COPY ./results/v7_trainsets_ckp-300_XLMRoBERTa_20eps ./results/v7_trainsets_ckp-300_XLMRoBERTa_20eps

# Copy .env vào nhé 
COPY .env ./

# Chạy ứng dụng
CMD ["uvicorn", "api_export:app", "--host", "0.0.0.0", "--port", "25041"]
