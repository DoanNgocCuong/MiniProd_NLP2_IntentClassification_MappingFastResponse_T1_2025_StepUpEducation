import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import time

# Configuration
model_path = "./results/v5_trainsets_checkpoint-140_XLMRoBERTa_10eps"  # Replace xxx with specific checkpoint number
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
    # Mapping giữa user_intent và fast_response
    intent_responses = {
        'intent_positive': "À há",
        'intent_negative': "Ra là vậy",
        'intent_neutral': "Tớ hiểu rồi",
        'intent_learn_more': "Để xem nào",
        'intent_fallback': "Cậu chắc chứ?",
        'cant_hear': "Cậu còn ở đó không? Tớ vẫn đợi cậu nói nè"
    }
    
    return intent_responses.get(user_intent, "Phản hồi không xác định")


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