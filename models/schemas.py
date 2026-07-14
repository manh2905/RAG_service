"""
models/schemas.py
-----------------
Định nghĩa các Pydantic models dùng để validate dữ liệu
giao tiếp giữa Python RAG service và Node.js backend.

Phiên bản v2:
- Thêm QueryIntent cho Query Router (phân loại CHIT_CHAT / RAG_REQUIRED).
- QueryRequest chuyển sang Single-turn (bỏ history) + Global Search (bỏ subject_id).
- IngestRequest hỗ trợ teacher_metadata.
- Citation bổ sung chapter và section từ heading hierarchy.
"""

from typing import List, Literal, Optional
from pydantic import BaseModel, Field


# ============================================================
# ROUTER SCHEMA — Phân loại ý định câu hỏi
# ============================================================

class QueryIntent(BaseModel):
    """
    Kết quả phân loại ý định câu hỏi bởi Query Router.
    LLM sẽ trả về Structured Output theo schema này.

    - CHIT_CHAT:     Câu hỏi giao tiếp thông thường (chào hỏi, cảm ơn, ...).
    - RAG_REQUIRED:  Câu hỏi cần tra cứu tài liệu để trả lời.
    """
    intent: Literal["CHIT_CHAT", "RAG_REQUIRED"] = Field(
        ...,
        description="Loại ý định: 'CHIT_CHAT' hoặc 'RAG_REQUIRED'"
    )


# ============================================================
# INPUT SCHEMAS — Dữ liệu nhận từ Node.js
# ============================================================

class QueryRequest(BaseModel):
    """
    Request gửi tới endpoint /api/query.
    Chế độ Single-turn: chỉ chứa câu hỏi và conversation_id.
    Tìm kiếm Global: không cần subject_id, search toàn bộ Vector DB.
    """
    question: str = Field(..., description="Câu hỏi của người dùng")
    conversation_id: str = Field(..., description="ID cuộc hội thoại hiện tại")


class IngestRequest(BaseModel):
    """
    Request gửi tới endpoint /api/ingest.
    Chứa thông tin tài liệu cần nạp vào hệ thống.
    """
    doc_id: str = Field(..., description="ID duy nhất của tài liệu")
    subject_id: str = Field(..., description="ID môn học mà tài liệu thuộc về")
    file_path: str = Field(..., description="Đường dẫn tuyệt đối tới file PDF trên server")
    teacher_metadata: Optional[dict] = Field(
        default={},
        description="Metadata bổ sung từ giáo viên (tên tác giả, ghi chú, ...)"
    )


# ============================================================
# OUTPUT SCHEMAS — Dữ liệu trả về cho Node.js
# ============================================================

class Citation(BaseModel):
    """
    Trích dẫn nguồn từ tài liệu gốc.
    Bao gồm thông tin heading hierarchy (chapter, section) để
    người dùng dễ dàng tra cứu lại vị trí trong tài liệu.
    """
    doc_id: str = Field(..., description="ID của tài liệu được trích dẫn")
    page_number: int = Field(..., description="Số trang chứa thông tin")
    snippet: str = Field(..., description="Đoạn trích ngắn từ tài liệu gốc")
    chapter: Optional[str] = Field(
        default=None,
        description="Tên chương (H1) chứa đoạn trích dẫn"
    )
    section: Optional[str] = Field(
        default=None,
        description="Tên mục/phần (H2/H3) chứa đoạn trích dẫn"
    )


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
