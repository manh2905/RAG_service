# RAG Education Service (v3)

Microservice xử lý RAG (Retrieval-Augmented Generation) cho hệ thống Trợ lý Giáo dục.
Được thiết kế để giao tiếp với Node.js Backend qua các REST API theo mô hình bất đồng bộ (Async + Callback).

## Công nghệ sử dụng
- **Framework**: FastAPI (Python 3.11)
- **Vector Database**: Qdrant
- **LLM & Embedding**: Google Gemini (gemini-2.0-flash, text-embedding-004)
- **Document Parsing**: LlamaParse (PDF, DOCX) + Fallback (pypdf, python-docx)
- **NLP**: Underthesea (Tách từ Tiếng Việt)

## Cài đặt và Chạy cục bộ (Local)

1. **Cài đặt môi trường:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # hoặc venv\Scripts\activate trên Windows
   pip install -r requirements.txt
   ```

2. **Cấu hình môi trường:**
   Tạo file `.env` từ `.env.example` và điền các API keys:
   ```env
   GOOGLE_API_KEY=your_key_here
   LLAMA_CLOUD_API_KEY=your_llama_key_here
   INTERNAL_SECRET=your_secret_here
   ```

3. **Khởi chạy Qdrant:**
   ```bash
   docker run -p 6333:6333 qdrant/qdrant
   ```

4. **Khởi chạy service:**
   ```bash
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

## Chạy với Docker Compose
```bash
docker-compose up -d --build
```
Dịch vụ sẽ chạy ở `http://localhost:8000` và Qdrant ở `http://localhost:6333`.

## Tài liệu API
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`
- **API Contract (JSON)**: Xem file `docs/api_contract.md`

## Testing
Chạy test bằng pytest:
```bash
pytest tests/ -v
```
