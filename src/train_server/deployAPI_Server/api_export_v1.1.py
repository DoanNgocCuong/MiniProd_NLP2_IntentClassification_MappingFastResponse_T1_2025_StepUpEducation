import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import time
import random  # Thêm import random

# Configuration
model_path = "./trained_models/v7_trainsets_ckp-300_XLMRoBERTa_20eps"  # Replace xxx with specific checkpoint number
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
model = AutoModelForSequenceClassification.from_pretrained(model_path)

print("Model loaded:", model)
print("Tokenizer loaded:", tokenizer)

# Create FastAPI app
app = FastAPI()

# Define input data model
class InputData(BaseModel):
    robot: str
    user_answer: str

# Define output data model
class OutputData(BaseModel):
    user_intent: str
    confidence_score: float
    response_time_ms: float
    fast_response: str

# Function to prepare data
def prepare_input(robot: str, user_answer: str):
    input_text = f"question: {robot.strip().lower()}. answer: {user_answer.strip().lower()}"
    return input_text

# Label mapping
label2id = {
    'intent_fallback': 0,
    'intent_learn_more': 1,
    'intent_negative': 2,
    'intent_neutral': 3,
    'intent_positive': 4,
    'silence': 5
}

def intent_to_fast_response(user_intent: str) -> str:
    """
    Trả về phản hồi nhanh tương ứng với user_intent.
    
    :param user_intent: Ý định của người dùng
    :return: Phản hồi nhanh tương ứng
    """
    
    intent_responses = {
        'intent_positive': ["Sounds great","That’s interesting", "interesting answer", "That sounds good", ],
        'intent_negative': ["Hmm, got it", "Hmm, I understand", "I get what you mean"],
        'intent_neutral': ["Hmm, let me see", "Để Pika nghĩ xem nào*#*#", "Nice try*#*#", "Hmm, let me think*#*#", "Let’s see*#*#"],
        'intent_learn_more': ["Để xem nào*#*#", "Cùng tìm hiểu thêm nhé*#*#", "Thú vị ghê*#*#"],
        'intent_fallback': ["I get what you mean", "Hmm, got it"],
        'silence': ["Hmm, let me think*#*#", "Hmm, let me see"]
    }
    #  'intent_fallback': ["Tớ nghĩ chủ đề này không phù hợp lắm*#*#", "Cùng tập trung vào bài học với tớ nhé*#*#"],
    #        'silence': ["Cậu còn ở đó không? Tớ vẫn đợi cậu nói nè*#*#", "Cậu có nghe thấy không?*#*#", "Tớ vẫn đang chờ cậu nè!*#*#"]
    responses = intent_responses.get(user_intent, ["Cậu chắc chứ?"])
    return random.choice(responses)  # Chọn ngẫu nhiên một phản hồi từ danh sách


# API endpoint
@app.post("/predict", response_model=OutputData)
def predict(data: InputData):
    input_text = prepare_input(data.robot, data.user_answer)

    # Tokenize input
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=128)

    # Determine device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)  # Move model to appropriate device

    # Measure response time
    start_time = time.time()
    with torch.no_grad():
        outputs = model(**{k: v.to(device) for k, v in inputs.items()})
    end_time = time.time()

    # Process results
    predictions = outputs.logits.argmax(dim=-1).item()
    prediction_scores = torch.nn.functional.softmax(outputs.logits, dim=-1)
    confidence_score = prediction_scores.max().item()
    response_time_ms = (end_time - start_time) * 1000  # Convert to milliseconds

    # Convert predicted label to label name
    user_intent = [k for k, v in label2id.items() if v == predictions][0]

    # Lấy phản hồi nhanh từ mapping
    fast_response = intent_to_fast_response(user_intent)

    return OutputData(user_intent=user_intent, confidence_score=confidence_score, response_time_ms=response_time_ms, fast_response=fast_response)


# Run server
# To run the server, use the following command in the terminal:
# uvicorn api_export:app --reload 