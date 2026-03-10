Trong thư mục này của chúng em bao gồm:

- Thư mục Build RAG: chứa các file về RAG
- Evaluate Model: chứa các file liên quan đến việc đánh giá model sinh ảnh
- Fine-tune Time: chứa các file liên quan đến finetune model sinh ảnh
- Report
- Slide trình bày về project này
- Apiwatwords_Framework.ipynb: file tổng hợp cách thức chạy từ đầu đến cuối,và thực hiện pipeline:

    1) Truy vấn kiến thức lịch sử từ Milvus (Milvus Lite DB file)
    2) RAG + Prompt Engineering** (Qwen2.5-3B-Instruct) để tạo diffusion prompt (EN, <100 tokens)
    3) Sinh N ảnh bằng Qwen-Image
    4) Chọn ảnh tốt nhất theo DPG-style (LLM constraint checking) và kết hợp CLIP (Best-of-N)
Các tham số cần nhập là: user_input (prompt của người dùng), N_IMAGES (số lượng ảnh model sẽ sinh ra)

**CHÚ Ý: Khi chạy file Apiwatwords_Framework, làm đúng theo TODO: hướng dẫn mà chúng em có đề cập.**
