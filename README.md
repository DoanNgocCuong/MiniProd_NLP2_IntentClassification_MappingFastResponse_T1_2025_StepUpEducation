# MiniProd_NLP2_InstantResponse_T1_2025_StepUpEducation

1. Data Generation: 
- Gen Data sử dụng Prompting để gen. 

Hướng Prompt 1: Gen dạng JSON: AI - User - Intent
Hướng Prompt 2 (Hướng về sau): 
    - Gen dạng JSON: AI - User - Intent
    - Thêm 1 step: Prompting để (AI - User) => Detect ra Intent. 
[Có 1 tool Extract Json2Excel để bổ trợ]

2. data Training: 
- Deploy 1.1: Training nhầm dataset 1100 dòng. (Nhưng acc lại cho thấy khá ngon với Acc 80%) - NGUYÊN NHÂN LÀ DO: DATA TRAINING 15000 DÒNG NHIỀU KHI KO ỔN, KO QUÁ ĐÚNG INTENT. 

+, Case Robot bảo User nói câu tích cực, tiêu cực, trung tính => User nhắc lại theo => Detect ra Intent: gì ? => Chốt chung là neutral, ....

- Deploy 1.2: Update deploy1.1 thêm log 

