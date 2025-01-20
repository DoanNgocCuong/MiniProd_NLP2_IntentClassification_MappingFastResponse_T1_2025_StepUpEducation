import torch
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, f1_score
import psutil
import os
import GPUtil
from threading import Thread
import time
from sklearn.model_selection import train_test_split

# Kiểm tra CUDA
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA device name:", torch.cuda.get_device_name(0))
    print("CUDA device count:", torch.cuda.device_count())

# 1. Chuẩn bị dữ liệu
def prepare_dataset(file_path, text_columns, label_column, tokenizer, max_seq_length):
    print(f"Reading data from: {file_path}")
    df = pd.read_excel(file_path)
    
    def combine_text(row):
        question = row[text_columns[0]].strip().lower()
        answer = row[text_columns[1]].strip().lower() if pd.notna(row[text_columns[1]]) else ""
        return f"question: {question}. answer: {answer}"

    df["input_text"] = df.apply(combine_text, axis=1)
    unique_labels = sorted(df[label_column].unique())
    label2id = {label: idx for idx, label in enumerate(unique_labels)}
    df["label"] = df[label_column].map(label2id)

    print(f"Unique labels: {unique_labels}")
    print(f"Label to ID mapping: {label2id}")

    train_df, valid_df = train_test_split(df, test_size=0.25, random_state=42, stratify=df["label"])
    print(f"Training samples: {len(train_df)}, Validation samples: {len(valid_df)}")

    def tokenize_df(df):
        tokenized_data = tokenizer(
            list(df["input_text"]),
            truncation=True,
            padding=True,
            max_length=max_seq_length
        )
        tokenized_data["labels"] = list(df["label"])
        return Dataset.from_dict(tokenized_data)

    train_dataset = tokenize_df(train_df)
    valid_dataset = tokenize_df(valid_df)

    return train_dataset, valid_dataset, label2id

file_path = "processed_data_example_v6_15000Data_dang123new_cleaned_data.xlsx"
text_columns = ["robot", "user_answer"]
label_column = "user_intent"
max_seq_length = 128
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

train_dataset, valid_dataset, label2id = prepare_dataset(file_path, text_columns, label_column, tokenizer, max_seq_length)
print(f"Label mapping: {label2id}")

# 2. Huấn luyện mô hình 
model = AutoModelForSequenceClassification.from_pretrained("xlm-roberta-base", num_labels=len(label2id))

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted")
    print(f"Evaluation - Accuracy: {acc:.4f}, F1 Score: {f1:.4f}")
    return {"accuracy": acc, "f1": f1}

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    num_train_epochs=100,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    compute_metrics=compute_metrics
)

def monitor_resources():
    while True:
        ram = psutil.virtual_memory()
        print(f"\nRAM Usage: {ram.percent}%")
        print(f"Used RAM: {ram.used/1024/1024/1024:.2f}GB")
        print(f"Available RAM: {ram.available/1024/1024/1024:.2f}GB")
        if torch.cuda.is_available():
            gpus = GPUtil.getGPUs()
            for gpu in gpus:
                print(f"GPU {gpu.id} Memory Usage: {gpu.memoryUsed}MB/{gpu.memoryTotal}MB ({gpu.memoryUtil*100:.1f}%)")
        time.sleep(30)

monitor_thread = Thread(target=monitor_resources, daemon=True)
monitor_thread.start()

print("Starting training...")

trainer.train()

# Lưu best model và checkpoint cuối cùng
best_model_path = trainer.state.best_model_checkpoint
final_model_path = "./final_model"
print(f"Best model checkpoint: {best_model_path}")
print(f"Saving best model to: {final_model_path}")
trainer.save_model(final_model_path)

# 3. Đánh giá mô hình
def evaluate_model(test_file_path):
    print(f"Evaluating model on test data: {test_file_path}")
    test_dataset, _, _ = prepare_dataset(test_file_path, text_columns, label_column, tokenizer, max_seq_length)
    results = trainer.predict(test_dataset)

    print("Metrics:", results.metrics)

    predictions = results.predictions.argmax(axis=1)
    test_dataset = test_dataset.to_pandas()
    test_dataset["predicted_label"] = predictions
    test_dataset["predicted_label_name"] = test_dataset["predicted_label"].map({v: k for k, v in label2id.items()})

    original_df = pd.read_excel(test_file_path)
    original_df["predicted_label"] = test_dataset["predicted_label"]
    original_df["predicted_label_name"] = test_dataset["predicted_label_name"]

    output_file = "eval_results_test2_1000processedDang123new.xlsx"
    original_df.to_excel(output_file, index=False)
    print(f"Test results saved to: {output_file}")

test_file_path = "test2_1000processedDang123new.xlsx"
evaluate_model(test_file_path)
