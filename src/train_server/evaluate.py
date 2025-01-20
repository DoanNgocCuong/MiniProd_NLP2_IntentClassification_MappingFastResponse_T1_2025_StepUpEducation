import torch
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import time

# Cấu hình
model_path = "deployAPI_server/results/checkpoint-1720_v6data"  # Thay xxx bằng số checkpoint cụ thể, ví dụ checkpoint-500
file_path = "test2_processed_benchmark_1000data123new.xlsx"

text_columns = ["robot", "user_answer"]
label_column = "user_intent"
max_seq_length = 128
print(f"Model path: {model_path}")
print(f"File path: {file_path}")
# Load tokenizer và model đã train
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
model = AutoModelForSequenceClassification.from_pretrained(model_path)

def prepare_dataset(file_path, text_columns, label_column, tokenizer, max_seq_length):
    """
    Chuẩn bị dữ liệu từ file Excel.
    """
    # Đọc dữ liệu
    df = pd.read_excel(file_path)
    
    # Ghép văn bản
    def combine_text(row):
        question = row[text_columns[0]].strip().lower()
        answer = row[text_columns[1]].strip().lower() if pd.notna(row[text_columns[1]]) else ""
        return f"question: {question}. answer: {answer}"

    df["input_text"] = df.apply(combine_text, axis=1)

    # # Chuyển đổi nhãn thành số
    # unique_labels = sorted(df[label_column].unique())
    # label2id = {label: idx for idx, label in enumerate(unique_labels)}
    # Sử dụng nhãn đã định nghĩa sẵn
    # label2id = {
    #     'intent_fallback': 0,
    #     'intent_learn_more': 1,
    #     'intent_negative': 2,
    #     'intent_neutral': 3,
    #     'intent_positive': 4,
    #     'silence': 5
    # }
    # or user_intent_fallback, user_intent_learn_more, user_intent_negative, user_intent_neutral, user_intent_positive, user_intent_silence
    label2id = {
        'user_intent_fallback': 0,
        'user_intent_learn_more': 1,
        'user_intent_negative': 2,
        'user_intent_neutral': 3,
        'user_intent_positive': 4,
        'user_intent_silence': 5
    }    
    
    df["label"] = df[label_column].map(label2id)    

    # Function để tokenize DataFrame
    def tokenize_df(df):
        tokenized_data = tokenizer(
            list(df["input_text"]),
            truncation=True,
            padding=True,
            max_length=max_seq_length
        )
        tokenized_data["labels"] = list(df["label"])
        return Dataset.from_dict(tokenized_data)

    # Tokenize dữ liệu
    dataset = tokenize_df(df)
    return dataset, label2id

def compute_metrics(eval_pred):
    """Hàm tính toán các metric."""
    predictions, labels = eval_pred
    preds = predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted")
    return {"accuracy": acc, "f1": f1}

# Chuẩn bị dataset và trainer
test_dataset, label2id = prepare_dataset(file_path, text_columns, label_column, tokenizer, max_seq_length)

trainer = Trainer(
    model=model,
    compute_metrics=compute_metrics
)

# Đánh giá
results = trainer.predict(test_dataset)
print("\nMetrics:", results.metrics)

# Xử lý và lưu kết quả
predictions = results.predictions.argmax(axis=1)

# Đo response time cho từng mẫu
response_times = []
model.eval()

# Xác định thiết bị
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)  # Chuyển model sang device phù hợp

with torch.no_grad():
    for i in range(len(test_dataset)):
        sample = test_dataset[i]
        inputs = {k: torch.tensor(v).unsqueeze(0).to(device) for k, v in sample.items() if k != 'labels'}
        
        start_time = time.time()
        _ = model(**inputs)
        end_time = time.time()
        
        response_times.append((end_time - start_time) * 1000)  # Convert to milliseconds

# Đọc lại file gốc để giữ nguyên tất cả thông tin
original_df = pd.read_excel(file_path)

# Thêm các cột dự đoán vào DataFrame gốc
original_df["predicted_label"] = predictions
original_df["predicted_label_name"] = original_df["predicted_label"].map({v: k for k, v in label2id.items()})

# Thêm cột confidence scores
prediction_scores = torch.nn.functional.softmax(torch.tensor(results.predictions), dim=-1)
original_df["confidence_score"] = prediction_scores.max(dim=1).values

# Thêm cột is_True để so sánh kết quả dự đoán với nhãn thực tế
original_df["is_True"] = original_df["predicted_label_name"] == original_df[label_column]

# Thêm cột response time
original_df["response_time_ms"] = response_times

# Lưu kết quả ra file
output_file = f"evaluationResults.xlsx"
original_df.to_excel(output_file, index=False)
print(f"\nTest results saved to {output_file}")

# In thêm thống kê về độ chính xác và thời gian phản hồi trung bình
correct_predictions = original_df["is_True"].sum()
total_predictions = len(original_df)
accuracy = correct_predictions / total_predictions
avg_response_time = original_df["response_time_ms"].mean()
print(f"\nAccuracy: {accuracy:.2%} ({correct_predictions}/{total_predictions} correct predictions)")
print(f"Average response time: {avg_response_time:.2f} ms") 