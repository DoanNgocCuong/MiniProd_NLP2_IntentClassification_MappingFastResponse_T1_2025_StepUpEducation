```bash
[ubuntu@mgc-dev-3090-01:~/hungdv/ … /FastResponse_IntentClassification]└5 [.venv] $ du -h results/checkpoint-2460/*
4.0K    results/checkpoint-2460/config.json
1.1G    results/checkpoint-2460/model.safetensors
2.1G    results/checkpoint-2460/optimizer.pt
16K     results/checkpoint-2460/rng_state.pth
4.0K    results/checkpoint-2460/scheduler.pt
4.0K    results/checkpoint-2460/special_tokens_map.json
4.0K    results/checkpoint-2460/tokenizer_config.json
17M     results/checkpoint-2460/tokenizer.json
48K     results/checkpoint-2460/trainer_state.json
8.0K    results/checkpoint-2460/training_args.bin
[ubuntu@mgc-dev-3090-01:~/hungdv/ … /FastResponse_IntentClassification]└5 [.venv] $ 
```



Để inference, bạn chỉ cần những file sau từ checkpoint:

1. `model.safetensors` (1.1G) - file chứa weights của model
2. `config.json` (4.0K) - cấu hình model
3. `tokenizer.json` (17M) - file tokenizer
4. `tokenizer_config.json` (4.0K) - cấu hình tokenizer
5. `special_tokens_map.json` (4.0K) - mapping các special tokens

Các file không cần thiết cho inference:
- `optimizer.pt` (2.1G) - trạng thái optimizer (chỉ cần cho training)
- `rng_state.pth` (16K) - trạng thái random number generator
- `scheduler.pt` (4.0K) - trạng thái learning rate scheduler
- `trainer_state.json` (48K) - trạng thái của trainer
- `training_args.bin` (8.0K) - các tham số training

Tổng dung lượng cần thiết cho inference: khoảng 1.12GB (giảm đáng kể so với 3.2GB của toàn bộ checkpoint)
