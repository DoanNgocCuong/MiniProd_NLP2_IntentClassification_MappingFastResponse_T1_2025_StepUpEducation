*VOZ-HSD Dataset* là một bộ dữ liệu dành cho nghiên cứu *phát hiện phát ngôn thù ghét (Hate Speech Detection)* bằng tiếng Việt. 

### *Thông tin chính:*
1. *Nguồn gốc:*
   - Thu thập từ các bình luận trên diễn đàn *VOZ Forum*.
   - Gán nhãn bằng mô hình AI (*ViSoBERT-HSD*).

2. *Chi tiết dữ liệu:*
   - Số lượng bình luận: *10.7 triệu*.
   - Dung lượng: *1.89 GB* (dạng gốc), *1.16 GB* (dạng Parquet).
   - Nhãn:
     - *HATE (1):* Phát ngôn thù ghét.
     - *CLEAN (0):* Không thù ghét.

3. *Mục đích:*
   - Hỗ trợ nghiên cứu *Xử lý ngôn ngữ tự nhiên (NLP)*.
   - Nâng cao khả năng phát hiện phát ngôn thù ghét tiếng Việt.

4. *Ứng dụng liên quan:*
   - *ViHateT5*: Mô hình Transformer tối ưu hóa cho phát hiện phát ngôn thù ghét tiếng Việt, được giới thiệu tại hội nghị *ACL 2024*.

5. *Lưu ý:*
   - Dataset này chỉ dùng cho *nghiên cứu*, không khuyến khích sử dụng nhãn để tinh chỉnh mô hình khác.

6. *Liên hệ:*
   - Tác giả: Nguyễn Thanh.
   - Email: *luannt@uit.edu.vn*.

*Tóm lại*, VOZ-HSD là một nguồn dữ liệu lớn và chất lượng, dành cho nghiên cứu về phân loại văn bản và phát ngôn thù ghét trong tiếng Việt.
---
https://huggingface.co/datasets/tarudesu/VOZ-HSD/viewer/default/train?f[labels][min]=1&f[labels][imax]=1