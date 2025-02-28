import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import time
import random  # Thêm import random

# Configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_path = "./trained_models/v7_trainsets_ckp-300_XLMRoBERTa_20eps"
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.to(device)  # Move model to device once
model.eval()  # Set model to evaluation mode

print("Model loaded:", model)
print("Tokenizer loaded:", tokenizer)

# Create FastAPI app
app = FastAPI()

# Define input data model
class InputData(BaseModel):
    """
    Model cho input data của API.
    
    Attributes:
        robot (str): Câu hỏi của robot (ví dụ: "Cậu biết con mèo là gì không?")
        user_answer (str): Câu trả lời của user (ví dụ: "cat")
        robot_type (str, optional): Loại robot - "Agent" hoặc "Workflow". Mặc định là None (sẽ được xử lý như "Workflow")
    """
    robot: str
    user_answer: str
    robot_type: str | None = None

# Define output data model
class OutputData(BaseModel):
    """
    Model cho output data của API.
    
    Attributes:
        robot_type (str): Loại robot đã xử lý ("Agent" hoặc "Workflow")
        user_intent (str): Intent được phân loại (ví dụ: intent_positive, intent_negative,...)
        confidence_score (float): Độ tin cậy của việc phân loại intent (0-1)
        response_time_ms (float): Thời gian xử lý (milliseconds)
        fast_response (str): Câu trả lời nhanh dựa trên intent hoặc robot_type
    """
    robot_type: str | None = None
    user_intent: str
    confidence_score: float
    response_time_ms: float
    fast_response: str

# Function to prepare data
def prepare_input(robot: str, user_answer: str):
    """
    Chuẩn bị input text để đưa vào model phân loại.
    
    Args:
        robot (str): Câu hỏi của robot
        user_answer (str): Câu trả lời của user
    
    Returns:
        str: Text đã được format theo cấu trúc "question: {robot}. answer: {user_answer}"
              và đã được lowercase, strip whitespace
    
    Example:
        >>> prepare_input("Cậu biết con mèo là gì không?", "cat")
        "question: cậu biết con mèo là gì không?. answer: cat"
    """
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
    Trả về fast response ngẫu nhiên dựa trên intent được phân loại.
    Chỉ được sử dụng cho robot_type="Workflow".
    
    Args:
        user_intent (str): Intent được phân loại từ model 
                          (intent_positive/negative/neutral/learn_more/fallback/silence)
    
    Returns:
        str: Một câu trả lời ngẫu nhiên từ danh sách responses tương ứng với intent
    
    Example:
        >>> intent_to_fast_response("intent_positive")
        "Sounds great"  # hoặc bất kỳ câu nào khác trong danh sách intent_positive
    """
    
    intent_responses = {
        'intent_positive': [
            "Sounds great", 
            "That's interesting", 
            "Interesting answer", 
            "That sounds good", 
            "Câu trả lời này chắc chắn có gì đó rất hay ho, để Pika xem thử nha!", 
            "Bạn có biết không? Đây là một câu trả lời rất đáng chú ý nha!", 
            "Hmm, câu này nghe có vẻ hợp lý lắm á!", 
            "Ồ! Câu này có vẻ đúng nha!", 
            "Nghe có vẻ hợp lí phết nha!", 
            "Ồ ồ, có vẻ bạn biết đáp án rồi đó nha!", 
            "Pika cảm nhận được sự thông minh từ câu trả lời này!", 
            "Chờ Pika chút, để Pika kiểm tra độ chính xác nào!", 
            "Ồ, Pika thấy có gì đó đúng đúng rồi đây!", 
            "Pika sắp bật mí kết quả rồi đây!", 
            "Để xem có gì thú vị không nào!", 
            "Pika đang kiểm tra nè, chờ chút nha!", 
            "Pika đang lắc não để kiểm tra nè!", 
            "Pika sẽ soi kỹ câu này một chút nha!", 
            "Nghe là thấy có lý rồi nè, để Pika xem thử!", 
            "Bạn có vẻ rất chắc chắn với câu trả lời này hen!", 
            "Câu này nghe lọt tai lắm luôn á!", 
            "Pika đang rất háo hức kiểm tra câu trả lời của bạn đây!", 
            "Ui, câu này nghe chắc chắn lắm luôn!", 
            "Ooo, that's an idea!",
            "That sounds like a great answer!",
            "Hmm… this looks good! Let me confirm!",
            "Whoa, let me admire this answer for a second!",
            "I think I hear a 'correct' coming! Wait a sec!", 
            "Hold on! This looks like a great answer!", 
            "Wait, let me process this smart answer!", 
            "Oh wow, that sounded super smart!", 
        ],
        
        'intent_negative': [
            "Hmm, got it", 
            "Hmm, I understand", 
            "I get what you mean", 
            "Pika cần suy nghĩ thêm một chút về câu này!",
            "Pika cần vài giây để phân tích câu trả lời này!",
            "Pika đang kiểm tra nè, chờ chút nha!",
            "Để xem câu trả lời này có gì thú vị không nha!",
            "Để Pika sắp xếp lại thông tin chút xíu nha!",
            "Pika ghi nhận câu trả lời của bạn nha!",
            "Để Pika xem thử nào!",
            "Pika đang suy nghĩ đây, chờ xíu nha!",
            "Okay con dê, để Pika kiểm tra đã!",
            "Pika nhận rồi, để coi có gì hay ho không nha!",
            "Chờ xíu, Pika đang xử lý nè!",
            "Pika lưu câu này vào não rồi đó!",
            "Pika đang nghiền ngẫm câu này đây!",
            "Để Pika cân nhắc chút xíu nha!",
            "Chờ tí nha, Pika sắp ra kết quả rồi!",
            "E hèm, để mình kiểm tra đã!",
            "Pika cần một chút thời gian nè!",
            "Để xem câu này có gì đặc biệt không nào!",
            "Được rồi, để mình xét kỹ lại nha!",
            "Pika sẽ không vội đâu, cứ từ từ nào!",
            "Chờ xíu nha, mình sắp xong rồi nè!",
            "Let me take a look!",
            "Oh, that's your answer?",
            "Oh wow, let me double-check!",
            "Hmm, let me think for a moment!",
            "That's a creative guess! Let's see...",
            "Something seems a little off... Let's check!",
            "Oh wow! That's an unexpected answer! Let's see…",
            "Pika's thinking… thinking… still thinking!",
            "Not quite sure about that one",
            "Great effort! Let's see....",
            "I see what you mean!",
            "I see what you mean!",
            "Let me think for a sec!",
            "Hang on, let me check!",
            "Give me a moment to process that!",
            "Alright, let's see here...",
            "Hmm… let me take a look!",
            "Hold up, I'm on it!",
            "One sec, I'm figuring this out!",
            "Let me take a closer look at that!",
            "Alright, let me go over this!",
            "Thinking… thinking… still thinking…",
            "Let's see what happens next!",
            "Hmm, let's take our time with this!",
            "I'll just double-check for a moment!",
            "Give me a sec to work through this!",
            "Gotta make sure I get this right!",
            "This might take just a little moment!",
            "Let me pull this up real quick!",
            "Almost there, hold tight!",
            "I'm piecing this together now!",
            "Just a tiny moment longer…",
        ],
        
        'intent_neutral': [
            "Hmm, let me see", 
            "Để Pika nghĩ xem nào*#*#", 
            "Nice try*#*#", 
            "Hmm, let me think*#*#", 
            "Let's see*#*#", 
            "Pika ghi nhận câu trả lời của bạn nha!",
            "Để Pika xem thử nào!",
            "Pika đang suy nghĩ đây, chờ xíu nha!",
            "Okay con dê, để Pika kiểm tra đã!",
            "Pika nhận rồi, để coi có gì hay ho không nha!",
            "Chờ xíu, Pika đang xử lý nè!",
            "Pika lưu câu này vào não rồi đó!",
            "Pika đang nghiền ngẫm câu này đây!",
            "Để Pika cân nhắc chút xíu nha!",
            "Chờ tí nha, Pika sắp ra kết quả rồi!",
            "E hèm, để mình kiểm tra đã!",
            "Pika cần một chút thời gian nè!",
            "Để xem câu này có gì đặc biệt không nào!",
            "Được rồi, để mình xét kỹ lại nha!",
            "Pika sẽ không vội đâu, cứ từ từ nào!",
            "Chờ xíu nha, mình sắp xong rồi nè!",
            "Bạn đoán xem, Pika sẽ nói gì tiếp đây?",
            "I see what you mean!",
            "Let me think for a sec!",
            "Hang on, let me check!",
            "Give me a moment to process that!",
            "Alright, let's see here...",
            "Hmm… let me take a look!",
            "Hold up, I'm on it!",
            "One sec, I'm figuring this out!",
            "Let me take a closer look at that!",
            "Alright, let me go over this!",
            "Thinking… thinking… still thinking…",
            "Let's see what happens next!",
            "Hmm, let's take our time with this!",
            "I'll just double-check for a moment!",
            "Give me a sec to work through this!",
            "Gotta make sure I get this right!",
            "This might take just a little moment!",
            "Let me pull this up real quick!",
            "Almost there, hold tight!",
            "I'm piecing this together now!",
            "Just a tiny moment longer…",
        ],
        
        'intent_learn_more': [
            "Để xem nào*#*#", 
            "Cùng tìm hiểu thêm nhé*#*#", 
            "Thú vị ghê*#*#", 
            "Pika ghi nhận câu trả lời của bạn nha!",
            "Để Pika xem thử nào!",
            "Pika đang suy nghĩ đây, chờ xíu nha!",
            "Okay con dê, để Pika kiểm tra đã!",
            "Pika nhận rồi, để coi có gì hay ho không nha!",
            "Chờ xíu, Pika đang xử lý nè!",
            "Pika lưu câu này vào não rồi đó!",
            "Pika đang nghiền ngẫm câu này đây!",
            "Để Pika cân nhắc chút xíu nha!",
            "Chờ tí nha, Pika sắp ra kết quả rồi!",
            "E hèm, để mình kiểm tra đã!",
            "Pika cần một chút thời gian nè!",
            "Để xem câu này có gì đặc biệt không nào!",
            "Được rồi, để mình xét kỹ lại nha!",
            "Pika sẽ không vội đâu, cứ từ từ nào!",
            "Chờ xíu nha, mình sắp xong rồi nè!",
            "Bạn đoán xem, Pika sẽ nói gì tiếp đây?",
            "I see what you mean!",
            "Let me think for a sec!",
            "Hang on, let me check!",
            "Give me a moment to process that!",
            "Alright, let's see here...",
            "Hmm… let me take a look!",
            "Hold up, I'm on it!",
            "One sec, I'm figuring this out!",
            "Let me take a closer look at that!",
            "Alright, let me go over this!",
            "Thinking… thinking… still thinking…",
            "Let's see what happens next!",
            "Hmm, let's take our time with this!",
            "I'll just double-check for a moment!",
            "Give me a sec to work through this!",
            "Gotta make sure I get this right!",
            "This might take just a little moment!",
            "Let me pull this up real quick!",
            "Almost there, hold tight!",
            "I'm piecing this together now!",
            "Just a tiny moment longer…",
        ],
        
        'intent_fallback': [
            "I get what you mean", 
            "Hmm, got it", 
            "Pika ghi nhận câu trả lời của bạn nha!",
            "Để Pika xem thử nào!",
            "Pika đang suy nghĩ đây, chờ xíu nha!",
            "Okay con dê, để Pika kiểm tra đã!",
            "Pika nhận rồi, để coi có gì hay ho không nha!",
            "Chờ xíu, Pika đang xử lý nè!",
            "Pika lưu câu này vào não rồi đó!",
            "Pika đang nghiền ngẫm câu này đây!",
            "Để Pika cân nhắc chút xíu nha!",
            "Chờ tí nha, Pika sắp ra kết quả rồi!",
            "E hèm, để mình kiểm tra đã!",
            "Pika cần một chút thời gian nè!",
            "Để xem câu này có gì đặc biệt không nào!",
            "Được rồi, để mình xét kỹ lại nha!",
            "Pika sẽ không vội đâu, cứ từ từ nào!",
            "Chờ xíu nha, mình sắp xong rồi nè!",
            "Bạn đoán xem, Pika sẽ nói gì tiếp đây?",
            "I see what you mean!",
            "Let me think for a sec!",
            "Hang on, let me check!",
            "Give me a moment to process that!",
            "Alright, let's see here...",
            "Hmm… let me take a look!",
            "Hold up, I'm on it!",
            "One sec, I'm figuring this out!",
            "Let me take a closer look at that!",
            "Alright, let me go over this!",
            "Thinking… thinking… still thinking…",
            "Let's see what happens next!",
            "Hmm, let's take our time with this!",
            "I'll just double-check for a moment!",
            "Give me a sec to work through this!",
            "Gotta make sure I get this right!",
            "This might take just a little moment!",
            "Let me pull this up real quick!",
            "Almost there, hold tight!",
            "I'm piecing this together now!",
            "Just a tiny moment longer…",              
        ],
        
        'silence': [
            "Hmm, let me think*#*#", 
            "Hmm, let me see", 
            "Ơ kìa… sao mà yên ắng quá vậy nè?",
            "Sao im re vậy ta? Hay bạn đang bí mật suy nghĩ?",
            "Trời ơi, tự nhiên im lặng làm Pika hồi hộp ghê!",
            "Pika không muốn phải nói chuyện một mình đâu",
            "Hình như có ai đó đang suy nghĩ thật kỹ nè!",
            "Ủa, không biết Pika có bỏ lỡ gì không ta?",
            'Hình như có ai đó đang chơi trò "im lặng là vàng" nè!',
            "E hèm! Pika gõ cửa nè, có ai ở nhà không?",
            "Pika có nên giả vờ đoán ý bạn luôn không ta?",
            "Hình như xung quanh Pika chỉ còn tiếng gió thổi qua thôi nè!",
            "Nếu Pika mà biết đọc suy nghĩ thì tốt biết mấy ha!",
            "Đừng lo, Pika sẽ không để sự yên lặng này kéo dài quá lâu đâu!",
            "Pika nghe thấy… một sự im lặng hoàn hảo!",            
        ]
    }
    #  'intent_fallback': ["Tớ nghĩ chủ đề này không phù hợp lắm*#*#", "Cùng tập trung vào bài học với tớ nhé*#*#"],
    #        'silence': ["Cậu còn ở đó không? Tớ vẫn đợi cậu nói nè*#*#", "Cậu có nghe thấy không?*#*#", "Tớ vẫn đang chờ cậu nè!*#*#"]
    responses = intent_responses.get(user_intent, ["Cậu chắc chứ?"])
    return random.choice(responses)  # Chọn ngẫu nhiên một phản hồi từ danh sách

# Add new function for Agent-specific responses
def get_agent_fast_response() -> str:
    """
    Trả về fast response ngẫu nhiên dành riêng cho robot_type="Agent".
    Các responses này đơn giản và ngắn gọn hơn, không phụ thuộc vào intent.
    
    Returns:
        str: Một câu trả lời ngẫu nhiên từ danh sách agent_responses
    
    Example:
        >>> get_agent_fast_response()
        "Để Pika kiểm tra nào"  # hoặc bất kỳ câu nào khác trong danh sách
    """
    agent_responses = [
        "Để Pika kiểm tra nào",
        "Hmm, để xem...",
        "Pika đang xử lý...",
        "Chờ Pika một chút nhé",
        "Để Pika phân tích thông tin",
        "Pika đang tính toán...",
        "Đợi Pika xíu nha",
        "Pika đang nghĩ...",
    ]
    return random.choice(agent_responses)

# API endpoint
@app.post("/predict", response_model=OutputData)
def predict(data: InputData):
    """
    Endpoint chính của API để xử lý câu trả lời của user.
    
    Process:
    1. Chuẩn bị input text
    2. Dùng model để phân loại intent
    3. Tính confidence score
    4. Chọn fast response dựa trên robot_type:
       - Nếu là "Agent": dùng get_agent_fast_response()
       - Nếu là "Workflow" hoặc None: dùng intent_to_fast_response()
    
    Args:
        data (InputData): Object chứa robot (câu hỏi), user_answer (câu trả lời) 
                         và robot_type (tùy chọn)
    
    Returns:
        OutputData: Object chứa kết quả phân tích và fast response
    
    Examples:
        >>> # Với Workflow
        >>> predict({"robot": "Cậu biết con mèo là gì không?", "user_answer": "cat"})
        {
            "robot_type": "Workflow",
            "user_intent": "intent_positive",
            "confidence_score": 0.98,
            "response_time_ms": 24.12,
            "fast_response": "Sounds great"
        }
        
        >>> # Với Agent
        >>> predict({
        ...     "robot": "Cậu biết con mèo là gì không?",
        ...     "user_answer": "cat",
        ...     "robot_type": "Agent"
        ... })
        {
            "robot_type": "Agent",
            "user_intent": "intent_positive",
            "confidence_score": 0.98,
            "response_time_ms": 22.5,
            "fast_response": "Để Pika kiểm tra nào"
        }
    """
    input_text = prepare_input(data.robot, data.user_answer)
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    
    # Move inputs to device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    start_time = time.time()
    with torch.no_grad():
        outputs = model(**inputs)  # Model đã ở trên device rồi
    end_time = time.time()

    # Process results
    predictions = outputs.logits.argmax(dim=-1).item()
    prediction_scores = torch.nn.functional.softmax(outputs.logits, dim=-1)
    confidence_score = prediction_scores.max().item()
    response_time_ms = (end_time - start_time) * 1000

    # Convert predicted label to label name
    user_intent = [k for k, v in label2id.items() if v == predictions][0]

    # Handle robot_type specific logic
    robot_type = data.robot_type if data.robot_type else "Workflow"  # Default to Workflow if not specified
    
    # Get appropriate fast response based on robot_type
    if robot_type == "Agent":
        fast_response = get_agent_fast_response()
    else:  # Workflow or default case
        fast_response = intent_to_fast_response(user_intent)

    return OutputData(
        robot_type=robot_type,
        user_intent=user_intent,
        confidence_score=confidence_score,
        response_time_ms=response_time_ms,
        fast_response=fast_response
    )


# Run server
# To run the server, use the following command in the terminal:
# uvicorn api_export:app --reload 