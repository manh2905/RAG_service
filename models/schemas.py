"""
models/schemas.py
-----------------
Định nghĩa các Pydantic models dùng để validate dữ liệu
giao tiếp giữa Python RAG service và Node.js backend.
"""

from typing import List
from pydantic import BaseModel, Field


# ============================================================
# INPUT SCHEMAS — Dữ liệu nhận từ Node.js
# ============================================================

class MessageHistory(BaseModel):
    """Một lượt hội thoại trong lịch sử chat."""
    role: str = Field(..., description="Vai trò: 'user' hoặc 'assistant'")
    content: str = Field(..., description="Nội dung tin nhắn")


class QueryRequest(BaseModel):
    """
    Request gửi tới endpoint /api/query.
    Chứa câu hỏi, subject để lọc vector, và lịch sử hội thoại.
    """
    question: str = Field(..., description="Câu hỏi của người dùng")
    subject_id: str = Field(..., description="ID môn học để lọc vector trong Qdrant")
    conversation_id: str = Field(..., description="ID cuộc hội thoại hiện tại")
    history: List[MessageHistory] = Field(
        default=[],
        description="Lịch sử hội thoại trước đó (tuỳ chọn)"
    )


class IngestRequest(BaseModel):
    """
    Request gửi tới endpoint /api/ingest.
    Chứa thông tin tài liệu cần nạp vào hệ thống.
    """
    doc_id: str = Field(..., description="ID duy nhất của tài liệu")
    subject_id: str = Field(..., description="ID môn học mà tài liệu thuộc về")
    file_path: str = Field(..., description="Đường dẫn tuyệt đối tới file PDF trên server")


# ============================================================
# OUTPUT SCHEMAS — Dữ liệu trả về cho Node.js
# ============================================================

class Citation(BaseModel):
    """Trích dẫn nguồn từ tài liệu gốc."""
    doc_id: str = Field(..., description="ID của tài liệu được trích dẫn")
    page_number: int = Field(..., description="Số trang chứa thông tin")
    snippet: str = Field(..., description="Đoạn trích ngắn từ tài liệu gốc")


class QueryResponse(BaseModel):
    """
    Response trả về từ endpoint /api/query.
    Bao gồm câu trả lời, danh sách trích dẫn, và cờ đánh giá.
    """
    answer: str = Field(..., description="Câu trả lời được sinh bởi LLM")
    citations: List[Citation] = Field(
        default=[],
        description="Danh sách trích dẫn nguồn tương ứng với [1], [2],..."
    )
    confidence: str = Field(
        default="high",
        description="Mức độ tin cậy: 'high', 'medium', 'low'"
    )
    no_answer: bool = Field(
        default=False,
        description="True nếu không tìm thấy thông tin liên quan trong tài liệu"
    )


class IngestResponse(BaseModel):
    """Response trả về từ endpoint /api/ingest."""
    status: str = Field(default="success", description="Trạng thái xử lý")
    message: str = Field(default="", description="Thông tin chi tiết (nếu có)")
    chunks_count: int = Field(default=0, description="Số lượng chunks đã lưu")
