import pandas as pd
import requests
import logging

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Đọc file Excel gốc
input_file = 'needPrediction.xlsx'  # Thay thế bằng đường dẫn file gốc của bạn
df = pd.read_excel(input_file)
logging.info("Đã đọc file gốc: %s", input_file)

# Sao chép file gốc ra file mới
output_file = 'output_file.xlsx'  # Đường dẫn file đầu ra
df.to_excel(output_file, index=False)
logging.info("Đã sao chép file gốc sang file mới: %s", output_file)

# Tạo danh sách để lưu kết quả
results = []

# Duyệt qua từng dòng trong DataFrame
for index, row in df.iterrows():
    robot = row['robot']
    user_answer = row['user_answer']
    
    # Thực hiện call API
    api_url = 'http://103.253.20.13:25041/predict'  # Đường dẫn API mới
    payload = {
        'robot': robot,
        'user_answer': user_answer
    }
    
    try:
        logging.info("Đang xử lý dòng %d: robot='%s', user_answer='%s'", index, robot, user_answer)
        response = requests.post(api_url, json=payload, headers={'Content-Type': 'application/json'})
        response.raise_for_status()  # Kiểm tra lỗi HTTP
        json_data = response.json()  # Phân tích JSON trả về
        
        # Thay đổi các trường theo yêu cầu
        results.append({
            'robot': robot,
            'user_answer': user_answer,
            'user_intent': json_data.get('user_intent', ''),
            'confidence_score': json_data.get('confidence_score', ''),
            'response_time_ms': json_data.get('response_time_ms', ''),
            'fast_response': json_data.get('fast_response', '')
        })
        logging.info("Dòng %d đã được xử lý thành công.", index)
    except Exception as e:
        logging.error("Lỗi khi xử lý dòng %d: %s", index, e)

# Chuyển kết quả thành DataFrame
results_df = pd.DataFrame(results)

# Lưu kết quả vào file output
results_df.to_excel(output_file, index=False)
logging.info("Đã lưu kết quả vào file: %s", output_file)