import torch
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import time

# Cấu hình
model_path = "results/checkpoint-1288"  # Thay xxx bằng số checkpoint cụ thể
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
model = AutoModelForSequenceClassification.from_pretrained(model_path)

print("Model loaded:", model)
print("Tokenizer loaded:", tokenizer)

# Tạo FastAPI app
app = FastAPI()

# Định nghĩa mô hình dữ liệu đầu vào
class InputData(BaseModel):
    robot: str
    user_answer: str

# Định nghĩa mô hình dữ liệu đầu ra
class OutputData(BaseModel):
    user_intent: str
    confidence_score: float
    response_time_ms: float

# Hàm để chuẩn bị dữ liệu
def prepare_input(robot: str, user_answer: str):
    input_text = f"question: {robot.strip().lower()}. answer: {user_answer.strip().lower()}"
    return input_text

# API endpoint
@app.post("/predict", response_model=OutputData)
def predict(data: InputData):
    input_text = prepare_input(data.robot, data.user_answer)

    # Tokenize input
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=128)

    # Xác định thiết bị
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)  # Chuyển model sang device phù hợp

    # Đo response time
    start_time = time.time()
    with torch.no_grad():
        outputs = model(**{k: v.to(device) for k, v in inputs.items()})
    end_time = time.time()

    # Xử lý kết quả
    predictions = outputs.logits.argmax(dim=-1).item()
    prediction_scores = torch.nn.functional.softmax(outputs.logits, dim=-1)
    confidence_score = prediction_scores.max().item()
    response_time_ms = (end_time - start_time) * 1000  # Convert to milliseconds

    # Chuyển đổi nhãn dự đoán thành tên nhãn
    label2id = {label: idx for idx, label in enumerate(sorted(tokenizer.get_vocab().keys()))}  # Cập nhật với nhãn thực tế
    user_intent = [k for k, v in label2id.items() if v == predictions][0]

    return OutputData(user_intent=user_intent, confidence_score=confidence_score, response_time_ms=response_time_ms)

# Chạy server
# Để chạy server, sử dụng lệnh sau trong terminal:
# uvicorn api_export:app --reload 