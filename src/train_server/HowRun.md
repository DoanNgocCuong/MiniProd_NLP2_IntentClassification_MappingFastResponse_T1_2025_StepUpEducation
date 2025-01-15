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

# Lưu danh sách package đã cài (Save installed packages list)
pip freeze > requirements.txt