import torch
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, f1_score

# Kiểm tra CUDA
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA device name:", torch.cuda.get_device_name(0))
    print("CUDA device count:", torch.cuda.device_count())

# 1. Chuẩn bị dữ liệu
def prepare_dataset(file_path, text_columns, label_column, tokenizer, max_seq_length):
    """
    Chuẩn bị dữ liệu từ file Excel.
    - Ghép câu `robot` và `user_answer` thành input
    - Tokenize dữ liệu.
    """
    # Đọc dữ liệu
    df = pd.read_excel(file_path)


    # Ghép văn bản
    def combine_text(row):
        question = row[text_columns[0]].strip().lower()
        answer = row[text_columns[1]].strip().lower() if pd.notna(row[text_columns[1]]) else ""
        return f"question: {question}. answer: {answer}"

    df["input_text"] = df.apply(combine_text, axis=1)

    # Chuyển đổi nhãn thành số
    unique_labels = sorted(df[label_column].unique())
    label2id = {label: idx for idx, label in enumerate(unique_labels)}
    df["label"] = df[label_column].map(label2id)

    # Tokenize
    tokenized_data = tokenizer(
        list(df["input_text"]),
        truncation=True,
        padding=True,
        max_length=max_seq_length
    )
    tokenized_data["labels"] = list(df["label"])

    # Chuyển thành HuggingFace Dataset
    dataset = Dataset.from_dict(tokenized_data)
    return dataset, label2id

# Cấu hình
file_path = "processed_data_example_v4_15000Data.xlsx"  # Đường dẫn file dữ liệu
text_columns = ["robot", "user_answer"]
label_column = "user_intent"
max_seq_length = 128
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

# Chuẩn bị dataset
train_dataset, label2id = prepare_dataset(file_path, text_columns, label_column, tokenizer, max_seq_length)
print(f"Label mapping: {label2id}")

# 2. Huấn luyện mô hình
num_labels = len(label2id)
model = AutoModelForSequenceClassification.from_pretrained("xlm-roberta-base", num_labels=num_labels)

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    num_train_epochs=1,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    report_to="none"
)

def compute_metrics(eval_pred):
    """Hàm tính toán các metric."""
    predictions, labels = eval_pred
    preds = predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted")
    return {"accuracy": acc, "f1": f1}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=train_dataset,  # Thay bằng valid_dataset nếu có
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()

# 3. Đánh giá trên tập test
def evaluate_model(test_file_path):
    """Đánh giá mô hình trên tập test."""
    test_dataset, _ = prepare_dataset(test_file_path, text_columns, label_column, tokenizer, max_seq_length)
    results = trainer.predict(test_dataset)

    print("Metrics:", results.metrics)

    # Xử lý nhãn dự đoán
    predictions = results.predictions.argmax(axis=1)
    test_dataset = test_dataset.to_pandas()
    test_dataset["predicted_label"] = predictions
    test_dataset["predicted_label_name"] = test_dataset["predicted_label"].map({v: k for k, v in label2id.items()})

    # Lưu kết quả ra file
    test_dataset.to_excel("test_results.xlsx", index=False)
    print("Test results saved to test_results.xlsx")

# Đường dẫn tập test
test_file_path = "test1_processed_TEST_500to1000Phrase.xlsx"
evaluate_model(test_file_path)
