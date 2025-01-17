anh @Quân Trần Ngọc ơi. 
---
Em gửi ver 1 `Fast Response API` ạ 

```
curl --location 'http://103.253.20.13:25041/predict' \
--header 'Content-Type: application/json' \
--data '{"robot": "Cậu có thể nói về gia đình cậu không? Like, who do you live with?", "user_answer": "Tớ sống với mẹ và ba, nhưng không có gì đặc biệt."}'
```

response
```
{
    "user_intent": "intent_neutral",
    "confidence_score": 0.9947407841682434,
    "response_time_ms": 30.63511848449707
}
```

---
