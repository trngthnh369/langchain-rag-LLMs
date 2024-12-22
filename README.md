# LangChain RAG System

Một hệ thống Retrieval-Augmented Generation (RAG) đơn giản sử dụng LangChain, FAISS và HuggingFace để tìm kiếm và trả lời câu hỏi dựa trên tài liệu.

## Tính năng

- Đọc và xử lý tài liệu văn bản từ files markdown
- Tạo vector embeddings sử dụng HuggingFace
- Lưu trữ và tìm kiếm vector sử dụng FAISS
- Trả lời câu hỏi dựa trên context từ tài liệu

## Yêu cầu hệ thống

- Python 3.8+
- pip (Python package manager)

## Cài đặt

1. Clone repository:
```bash
git clone [repository-url]
cd langchain-rag
```

2. Tạo và kích hoạt môi trường ảo:
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. Cài đặt các thư viện cần thiết:
```bash
pip install langchain langchain-community langchain-huggingface
pip install faiss-cpu
pip install sentence-transformers
pip install python-dotenv
pip install huggingface_hub
```

4. Cấu hình môi trường:
- Tạo file `.env` trong thư mục gốc
- Thêm HuggingFace API token của bạn:
- langchain-rag/ ├── create_database.py # Script tạo vector database ├── query_data.py # Script truy vấn và trả lời câu hỏi ├── .env # File cấu hình (chứa API tokens) ├── data/ │ └── books/ # Thư mục chứa tài liệu markdown │ └── alice_in_wonderland.md └── faiss/ # Thư mục chứa FAISS index
  
## Cách sử dụng

1. Chuẩn bị dữ liệu:
- Đặt các file markdown vào thư mục `data/books/`
- Đảm bảo files có định dạng .md

2. Tạo vector database:
```bash
python create_database.py
Truy vấn dữ liệu:
python query_data.py "Câu hỏi của bạn?"
Ví dụ:
python query_data.py "Who is Alice?"
Cách lấy HuggingFace API Token
Đăng ký tại HuggingFace
Vào Settings -> Access Tokens
Click "New Token"
Đặt tên và chọn role "read"
Copy token và thêm vào file .env
Xử lý lỗi thường gặp
Lỗi không tìm thấy file:

Kiểm tra đường dẫn file trong thư mục data/books/
Đảm bảo file có đuôi .md
Lỗi API Token:

Kiểm tra token trong file .env
Đảm bảo token còn hiệu lực
Tạo token mới nếu cần
Lỗi FAISS index:

Chạy lại create_database.py
Kiểm tra thư mục faiss đã được tạo
Đóng góp
Mọi đóng góp đều được chào đón! Hãy tạo issue hoặc pull request nếu bạn muốn cải thiện dự án.

License
MIT License

Liên hệ
gmail: truongthinhnguyen30303@gmail.com
