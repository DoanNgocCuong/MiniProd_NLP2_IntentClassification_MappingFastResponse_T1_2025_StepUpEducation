Version cũ là mình đang truyền: 

{"robot":"Cậu biết con mèo là gì không?","user_answer":"cat"}
```bash
curl --location 'http://103.253.20.13:25041/predict' \
--header 'Content-Type: application/json' \
--data '{"robot":"Cậu biết con mèo là gì không?","user_answer":"cat", "robot_type": "Workflow/Agent (lấy từ backend)"}'
```

---
version mới thì sẽ truyền. 
1. {"robot":"Cậu biết con mèo là gì không?","user_answer":"cat"}
logic như cũ 
2. Hoặc: 
{"robot":"Cậu biết con mèo là gì không?","user_answer":"cat", "robot_type": "Workflow"}
logic như cũ
Chẳng hạn: 
```
{   "robot_type": "Workflow"
    "user_intent": "intent_neutral",
    "confidence_score": 0.9978618025779724,
    "response_time_ms": 24.12700653076172,
    "fast_response": "Để Pika nghĩ xem nào*#*#"
}
```
3. Hoặc: 
{"robot":"Cậu biết con mèo là gì không?","user_answer":"cat", "robot_type": "Agent"}
trả ra: 
```
{   "robot_type": "Agent"
    "user_intent": "no"
    "confidence_score": 1.00,
    "response_time_ms": ...,
    "fast_response": "RANDOM TỪ LIST FAST RESPONSE FOR AGENT"
}
```
