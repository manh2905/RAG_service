# API Contract — RAG Service & Node.js

Tài liệu này định nghĩa cấu trúc JSON (request/response/callback) giao tiếp giữa Node.js và Python RAG Service theo kiến trúc bất đồng bộ.

---

## 1. Nạp Tài Liệu (Ingest Flow)

**Node.js gọi Python:** `POST /api/ingest`
```json
// Request từ Node.js
{
  "doc_id": "doc_123",
  "job_id": "job_abc",
  "subject_id": "sub_456",
  "file_path": "/app/uploads/doc_123.pdf",
  "callback_url": "http://localhost:3000/internal/callback",
  "teacher_metadata": {
    "author": "Nguyen Van A",
    "note": "Tài liệu môn Toán"
  }
}
```

**Python trả ngay:** `202 Accepted`
```json
{
  "status": "accepted",
  "job_id": "job_abc",
  "message": "Tài liệu doc_123 đang được xử lý"
}
```

**Python Callback (PROGRESS):**
```json
{
  "job_id": "job_abc",
  "attempt_count": 1,
  "event_type": "PROGRESS",
  "stage": "parsing" // "parsing", "chunking", "embedding", "indexing"
}
```

**Python Callback (SUCCEEDED):**
```json
{
  "job_id": "job_abc",
  "attempt_count": 1,
  "event_type": "SUCCEEDED",
  "chunks_count": 42,
  "chunk_manifest": [
    {
      "chunk_id": "uuid-1234...",
      "chunk_index": 0,
      "page_number": 1,
      "chapter": "Chương 1",
      "section": "1.1 Giới thiệu",
      "text_preview": "Đây là nội dung bắt đầu của tài liệu..."
    }
  ]
}
```

**Python Callback (FAILED):**
```json
{
  "job_id": "job_abc",
  "attempt_count": 1,
  "event_type": "FAILED",
  "error": {
    "code": "EMPTY_DOCUMENT",
    "message": "Không đọc được nội dung từ file"
  }
}
```

---

## 2. Chat / Query RAG

**Node.js gọi Python:** `POST /api/query`
```json
// Request từ Node.js
{
  "question": "Giới hạn hàm số là gì?",
  "conversation_id": "conv_789",
  "history": [
    {
      "role": "user",
      "content": "Chào bạn"
    },
    {
      "role": "assistant",
      "content": "Chào bạn, tôi có thể giúp gì?"
    }
  ]
}
```

**Python trả kết quả:** `200 OK`
```json
// Có câu trả lời (RAG_REQUIRED hoặc CHIT_CHAT)
{
  "answer": "Giới hạn hàm số là... theo nguồn [1].",
  "citations": [
    {
      "doc_id": "doc_123",
      "page_number": 5,
      "snippet": "Nội dung gốc về giới hạn hàm số...",
      "chapter": "Chương 2",
      "section": "2.1 Định nghĩa"
    }
  ],
  "confidence": "high",
  "no_answer": false,
  "usage": {
    "prompt_tokens": 120,
    "completion_tokens": 50,
    "total_tokens": 170,
    "model": "models/gemini-2.0-flash"
  }
}

// Không có câu trả lời (no_answer = true)
{
  "answer": "Không đủ dữ liệu/Không tìm thấy thông tin liên quan.",
  "citations": [],
  "confidence": "low",
  "no_answer": true,
  "usage": null
}
```

---

## 3. Ẩn / Hiện Tài Liệu (Visibility Flow)

**Node.js gọi Python:** `PATCH /api/docs/{doc_id}/visibility`
```json
// Request từ Node.js
{
  "job_id": "job_xyz",
  "action": "hide", // hoặc "unhide"
  "callback_url": "http://localhost:3000/internal/callback"
}
```

**Python trả ngay:** `202 Accepted`
```json
{
  "status": "accepted",
  "job_id": "job_xyz"
}
```

**Python Callback (SUCCEEDED):**
```json
{
  "job_id": "job_xyz",
  "attempt_count": 1,
  "event_type": "SUCCEEDED",
  "updated_count": 42 // Số lượng vector được ẩn/hiện
}
```

---

## 4. Xóa Tài Liệu (Delete Flow)

**Node.js gọi Python:** `DELETE /api/ingest/{doc_id}`
```json
// Request body từ Node.js
{
  "job_id": "job_del",
  "callback_url": "http://localhost:3000/internal/callback"
}
```

**Python trả ngay:** `202 Accepted`
```json
{
  "status": "accepted",
  "job_id": "job_del"
}
```

**Python Callback (SUCCEEDED):**
```json
{
  "job_id": "job_del",
  "attempt_count": 1,
  "event_type": "SUCCEEDED",
  "deleted_count": 42 // Số lượng vector bị xóa
}
```

---

## 5. Error Format (Dùng chung cho các response lỗi 4xx, 5xx)

```json
{
  "error_code": "FILE_NOT_FOUND",
  "message": "Không tìm thấy file: /app/uploads/doc_123.pdf",
  "timestamp": "2024-03-10T12:00:00Z"
}
```
