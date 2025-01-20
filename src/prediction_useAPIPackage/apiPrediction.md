
```
curl --location 'http://103.253.20.13:25041/predict' \
--header 'Content-Type: application/json' \
--data '{"robot": "Cậu biết từ '\''Engineer’s ruler'\'' trong tiếng Việt là gì không? Nó là '\''thước kẻ kỹ sư'\''. Nào, cậu thử nhắc lại với tớ: Engineer’s ruler!", "user_answer": "Engineer’s ruler!"}'
```


```
Nhận vào: 
1 file excel có 2 cột: 
robot	user_answer
Hello! Xin chào! Tớ là Pika. Hôm nay, chúng ta có thể nói về bất cứ chủ đề nào mà cậu thích như đồ ăn, động vật, trò chơi, bạn bè. What do you want to talk about?	Tớ thích động vật. Cậu thích gì?

--
Copy file gốc ra 1 file riêng, đi qua từng dòng lấy robot, user_answer, thực hiện call API và lấy JSON trả về parse ra cho vào các cột của file output đó. ```
