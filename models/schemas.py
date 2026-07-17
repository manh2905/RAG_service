"""
models/schemas.py
-----------------
Định nghĩa các Pydantic models dùng để validate dữ liệu
giao tiếp giữa Python RAG service và Node.js backend.

Phiên bản v3 — Theo sơ đồ luồng:
- Async pattern: IngestRequest có job_id + callback_url, trả 202.
- Callback schemas: PROGRESS / SUCCEEDED / FAILED / CANCELLED.
- Hide/Unhide/Delete schemas.
- Query nâng cấp: history (multi-turn) + usage tracking.
- Error response thống nhất.
"""

from datetime import datetime, timezone
from typing import List, Literal, Optional

from pydantic import BaseModel, Field


# ============================================================
# ERROR RESPONSE — Format lỗi thống nhất
# ============================================================

class ErrorResponse(BaseModel):
    """
    Format lỗi thống nhất cho tất cả API endpoints.
    Node.js sẽ luôn nhận lỗi theo format này.
    """
    error_code: str = Field(..., description="Mã lỗi (VD: FILE_NOT_FOUND, INVALID_FORMAT)")
    message: str = Field(..., description="Mô tả chi tiết lỗi")
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="Thời điểm xảy ra lỗi (ISO format)"
    )


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
# INGEST SCHEMAS — Nạp tài liệu (Async pattern)
# ============================================================

class IngestRequest(BaseModel):
    """
    Request gửi tới endpoint POST /api/ingest.
    Node.js tạo job_id trước, truyền kèm callback_url để Python
    gọi ngược khi xử lý xong.
    """
    doc_id: str = Field(..., description="ID duy nhất của tài liệu")
    job_id: str = Field(..., description="ID job do Node.js tạo trước")
    attempt_count: int = Field(..., description="Processing attempt hiện tại, số nguyên từ 1")
    subject_id: str = Field(..., description="ID môn học mà tài liệu thuộc về")
    file_path: str = Field(..., description="Đường dẫn tuyệt đối tới file trên server")
    callback_url: str = Field(..., description="URL để Python callback kết quả về Node.js")
    teacher_metadata: Optional[dict] = Field(
        default={},
        description="Metadata bổ sung từ giáo viên (tên tác giả, ghi chú, ...)"
    )


class IngestAcceptedResponse(BaseModel):
    """
    Response 202 Accepted — Python nhận request và bắt đầu xử lý nền.
    Node.js nhận response này ngay lập tức, không cần chờ xử lý xong.
    """
    status: str = Field(default="accepted", description="Luôn là 'accepted'")
    job_id: str = Field(..., description="Job ID để tracking")
    message: str = Field(default="Tài liệu đang được xử lý", description="Thông báo")


# ============================================================
# CALLBACK SCHEMAS — Python gọi ngược Node.js
# ============================================================

class ChunkManifestItem(BaseModel):
    """Thông tin tóm tắt của một chunk trong manifest."""
    chunk_index: int = Field(..., description="Thứ tự chunk (0-indexed)")
    chunk_id: str = Field(..., description="UUID thực dùng làm Qdrant point ID")
    chunk_text: str = Field(..., description="Toàn bộ text đã được embedding/index")
    content_hash: str = Field(..., description="SHA-256 lowercase hex của exact UTF-8 chunk_text")
    token_count: Optional[int] = Field(default=None, description="Số lượng token (ước tính)")
    page_number: Optional[int] = Field(default=None, description="Số trang nguồn (1-based)")
    chapter: Optional[str] = Field(default=None, description="Tên chương")
    section: Optional[str] = Field(default=None, description="Tên mục/phần")
    text_preview: Optional[str] = Field(default=None, description="50 ký tự đầu tiên của chunk")


class CallbackPayload(BaseModel):
    """
    Payload Python gửi tới callback_url của Node.js.
    Dùng chung cho cả ingest, hide/unhide, delete.

    eventType:
    - PROGRESS:  Đang xử lý (kèm stage)
    - SUCCEEDED: Hoàn tất thành công
    - FAILED:    Thất bại (kèm error)
    - CANCELLED: Đã bị hủy
    """
    job_id: str = Field(..., description="Job ID matching với request ban đầu")
    attempt_count: int = Field(..., description="Processing-job attempt, giữ nguyên từ request")
    event_type: Literal["PROGRESS", "SUCCEEDED", "FAILED", "CANCELLED"] = Field(
        ..., description="Loại sự kiện"
    )

    # === Dùng cho PROGRESS ===
    stage: Optional[str] = Field(
        default=None,
        description="Giai đoạn hiện tại: 'parsing', 'chunking', 'embedding', 'indexing'"
    )

    # === Dùng cho SUCCEEDED (ingest) ===
    chunks_count: Optional[int] = Field(default=None, description="Tổng số chunks đã tạo")
    chunk_manifest: Optional[List[ChunkManifestItem]] = Field(
        default=None, description="Danh sách metadata của từng chunk"
    )

    # === Dùng cho SUCCEEDED (delete) ===
    deleted_count: Optional[int] = Field(default=None, description="Số vectors đã xóa")

    # === Dùng cho SUCCEEDED (hide/unhide) ===
    updated_count: Optional[int] = Field(default=None, description="Số vectors đã cập nhật")

    # === Dùng cho FAILED ===
    error: Optional[dict] = Field(
        default=None,
        description="Chi tiết lỗi: {code: str, message: str}"
    )


# ============================================================
# DOCUMENT MANAGEMENT SCHEMAS — Hide/Unhide/Delete
# ============================================================

class VisibilityRequest(BaseModel):
    """
    Request gửi tới PATCH /api/docs/{doc_id}/visibility.
    Ẩn hoặc hiện tài liệu trong RAG (bật/tắt truy xuất).
    """
    job_id: str = Field(..., description="ID job do Node.js tạo")
    attempt_count: int = Field(..., description="Processing attempt hiện tại")
    action: Literal["hide", "unhide"] = Field(
        ..., description="'hide' = ẩn khỏi RAG, 'unhide' = hiện lại"
    )
    callback_url: str = Field(..., description="URL callback kết quả")


class DeleteRequest(BaseModel):
    """
    Request body cho DELETE /api/ingest/{doc_id}.
    Xóa toàn bộ vectors của tài liệu khỏi Qdrant.
    """
    job_id: str = Field(..., description="ID job do Node.js tạo")
    attempt_count: int = Field(..., description="Processing attempt hiện tại")
    callback_url: str = Field(..., description="URL callback kết quả")


class AcceptedResponse(BaseModel):
    """
    Response 202 chung cho hide/unhide/delete.
    """
    status: str = Field(default="accepted", description="Luôn là 'accepted'")
    job_id: str = Field(..., description="Job ID để tracking")


# ============================================================
# QUERY SCHEMAS — Chat/Query RAG
# ============================================================

class ChatMessage(BaseModel):
    """Một tin nhắn trong lịch sử hội thoại."""
    role: Literal["user", "assistant"] = Field(..., description="Vai trò: user hoặc assistant")
    content: str = Field(..., description="Nội dung tin nhắn")


class QueryRequest(BaseModel):
    """
    Request gửi tới endpoint POST /api/query.
    Node.js gửi câu hỏi kèm lịch sử gần nhất để Python dùng làm context.
    Search trên toàn bộ tài liệu READY + VISIBLE (is_hidden != true).
    """
    question: str = Field(..., description="Câu hỏi của người dùng")
    conversation_id: str = Field(..., description="ID cuộc hội thoại hiện tại do NodeJS tạo")
    history: Optional[List[ChatMessage]] = Field(
        default=[],
        description="Lịch sử hội thoại gần nhất (Node.js gửi kèm)"
    )
    request_id: Optional[str] = Field(default=None, description="Correlation/idempotency extension")
    user_id: Optional[str] = Field(default=None, description="Correlation context")


class Citation(BaseModel):
    """
    Trích dẫn nguồn từ tài liệu gốc.
    Bao gồm thông tin heading hierarchy (chapter, section) để
    người dùng dễ dàng tra cứu lại vị trí trong tài liệu.
    """
    vector_node_id: str = Field(..., description="Qdrant point ID của retrieved chunk")
    doc_id: str = Field(..., description="ID của tài liệu được trích dẫn")
    snippet: str = Field(..., description="Đoạn trích ngắn từ tài liệu gốc")
    page_number: Optional[int] = Field(default=None, description="Số trang chứa thông tin")
    chapter: Optional[str] = Field(
        default=None,
        description="Tên chương (H1) chứa đoạn trích dẫn"
    )
    section: Optional[str] = Field(
        default=None,
        description="Tên mục/phần (H2/H3) chứa đoạn trích dẫn"
    )


class UsageInfo(BaseModel):
    """
    Thông tin sử dụng LLM — để Node.js lưu vào MySQL cho dashboard.
    """
    prompt_tokens: int = Field(default=0, description="Số token trong prompt")
    completion_tokens: int = Field(default=0, description="Số token LLM sinh ra")
    total_tokens: int = Field(default=0, description="Tổng token")
    model: str = Field(default="", description="Tên model đã sử dụng")


class QueryResponse(BaseModel):
    """
    Response trả về từ endpoint POST /api/query.
    Bao gồm câu trả lời, danh sách trích dẫn, đánh giá, và usage.
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
    usage: Optional[UsageInfo] = Field(
        default=None,
        description="Thông tin sử dụng LLM (token counts, model name)"
    )
