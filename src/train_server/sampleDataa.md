Sample Data: 
```
Sample Data: 
robot user_answer user_intent
Cậu có muốn học thêm về các cụm từ khác không? Maybe something like 'Run fast'? Không, tớ không muốn học thêm đâu. intent_negative
Cậu có hiểu cách sử dụng 'Jump now' không? Do you feel confident using it? Tớ không biết, có thể tớ cần thêm thời gian. intent_neutral
Cậu có thể nói lại câu 'Jump now' không? Can you repeat it? Tớ không chắc lắm, nhưng tớ sẽ thử. intent_neutral
Cậu có muốn tìm hiểu thêm về các động từ hành động không? Like 'jump'? Có, tớ muốn biết thêm về các động từ khác. intent_learn_more
Cậu có muốn học cách kết hợp 'Jump now' với các câu khác không? For example, 'Jump now and have fun!' Ừ, tớ muốn học thêm về cách kết hợp. intent_learn_more
Cậu có thể nói 'Jump now' bằng tiếng Việt không? Can you translate it? Tớ không biết, có thể là 'Nhảy ngay'. intent_fallback
Cậu có nghĩ rằng 'Jump now' là một câu khó không? Is it difficult for you? Tớ không biết, có thể là dễ. intent_fallback
Cậu có muốn thử nói 'Jump now' ngay bây giờ không? Can you say it out loud? silence
Cậu có muốn cùng tớ thực hành câu này không? Let's practice 'Jump now' together! silence
```


====
6 nhãn, 
em mới test trên 2 model: 
- BERT : 46%
- XLM RoBERTa: 80%
-----

Model a Hoàng mới recommend em chưa test, 

==================
Code train model: https://github.com/DoanNgocCuong/MiniProd_NLP2_InstantResponse_IntentClassification_T1_2025_StepUpEducation/tree/main/src/train_server



===================
(trong link git có 15000 dòng data, file test 500 dòng). (Data ver1 này thì chưa chuẩn lắm). Tụi em đang update lại data. 
---
nếu được thì a @Hoang Xuan To  support tụi em xem có model nào NGON HƠN cả ModernBERT nữa không ạ. 
(em có gửi code train XLM-RoBERTa), a Hoàng xem thử ạ