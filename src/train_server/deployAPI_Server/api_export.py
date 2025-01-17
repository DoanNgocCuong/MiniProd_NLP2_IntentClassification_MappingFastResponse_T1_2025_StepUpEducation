import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import time

# Configuration
model_path = "./results/checkpoint-1288"  # Replace xxx with specific checkpoint number
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

    return OutputData(user_intent=user_intent, confidence_score=confidence_score, response_time_ms=response_time_ms)

# Run server
# To run the server, use the following command in the terminal:
# uvicorn api_export:app --reload 