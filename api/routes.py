"""
api/routes.py
-------------
Định nghĩa các API endpoints cho RAG microservice.
Giao tiếp nội bộ với Node.js backend qua HTTP.

Phiên bản v3 — Theo sơ đồ luồng:
  POST   /api/ingest              — Nạp tài liệu (async, 202)
  POST   /api/query               — Chat/Query RAG (sync, 200)
  PATCH  /api/docs/{doc_id}/visibility — Hide/Unhide (async, 202)
  DELETE /api/ingest/{doc_id}      — Xóa vectors (async, 202)
  GET    /api/health               — Health check

Error handling thống nhất dùng ErrorResponse.
"""

import logging
import time
from datetime import datetime, timezone

from fastapi import APIRouter, BackgroundTasks, HTTPException, status
from fastapi.responses import JSONResponse

from models.schemas import (
    IngestRequest,
    IngestAcceptedResponse,
    VisibilityRequest,
    DeleteRequest,
    AcceptedResponse,
    QueryRequest,
    QueryResponse,
    ErrorResponse,
)
from services.ingestion import ingest_document_background
from services.doc_manager import (
    hide_document_background,
    unhide_document_background,
    delete_document_background,
)
from services.rag_engine import process_query

logger = logging.getLogger(__name__)

# ── Khởi tạo router với prefix chung ────────────────────────────
router = APIRouter(prefix="/api", tags=["RAG"])


# ══════════════════════════════════════════════════════════════════
# HELPER: Error Response thống nhất
# ══════════════════════════════════════════════════════════════════

def _error_response(
    status_code: int,
    error_code: str,
    message: str,
) -> JSONResponse:
    """Tạo error response theo format thống nhất."""
    return JSONResponse(
        status_code=status_code,
        content=ErrorResponse(
            error_code=error_code,
            message=message,
            timestamp=datetime.now(timezone.utc).isoformat(),
        ).model_dump(),
    )


# ══════════════════════════════════════════════════════════════════
# ENDPOINT 1: Nạp tài liệu (Async — 202 Accepted)
# ══════════════════════════════════════════════════════════════════

@router.post(
    "/ingest",
    response_model=IngestAcceptedResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Nạp tài liệu — Async, trả 202 ngay, callback khi xong",
    description=(
        "Nhận request nạp tài liệu, trả 202 Accepted ngay lập tức. "
        "Xử lý nền: Parse → Chunk → Embed → Lưu Qdrant. "
        "Gọi callback_url khi hoàn tất (SUCCEEDED/FAILED)."
    ),
)
async def ingest_endpoint(
    request: IngestRequest,
    background_tasks: BackgroundTasks,
) -> IngestAcceptedResponse:
    """
    Xử lý nạp tài liệu vào Qdrant (async pattern).

    Luồng: Trả 202 → Background task → Callback khi xong.
    """
    logger.info(
        "[INGEST] Nhận request: doc_id=%s, job_id=%s, file=%s",
        request.doc_id,
        request.job_id,
        request.file_path,
    )

    # Thêm task xử lý vào background
    background_tasks.add_task(ingest_document_background, request)

    # Trả 202 ngay lập tức
    return IngestAcceptedResponse(
        status="accepted",
        job_id=request.job_id,
        message=f"Tài liệu {request.doc_id} đang được xử lý",
    )


# ══════════════════════════════════════════════════════════════════
# ENDPOINT 2: Chat/Query RAG (Sync — 200 OK)
# ══════════════════════════════════════════════════════════════════

@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Truy vấn RAG — Hỏi đáp với Query Router",
    description=(
        "Nhận câu hỏi + lịch sử hội thoại. Query Router phân loại intent:\n"
        "- CHIT_CHAT → LLM trả lời giao tiếp.\n"
        "- RAG_REQUIRED → Search READY+VISIBLE docs trong Qdrant → LLM → Citations.\n"
        "Trả kèm usage (token counts) để Node.js lưu cho dashboard."
    ),
)
async def query_endpoint(request: QueryRequest) -> QueryResponse:
    """Xử lý truy vấn RAG (sync, trả kết quả ngay)."""
    start_time = time.time()

    try:
        logger.info(
            "[QUERY] Nhận request: conv=%s, q='%s', history=%d msgs",
            request.conversation_id,
            request.question[:80],
            len(request.history or []),
        )

        response = await process_query(request)

        elapsed = time.time() - start_time
        logger.info(
            "[QUERY] Hoàn tất trong %.2fs | no_answer=%s | citations=%d",
            elapsed,
            response.no_answer,
            len(response.citations),
        )

        return response

    except FileNotFoundError as e:
        return _error_response(
            status.HTTP_404_NOT_FOUND,
            "FILE_NOT_FOUND",
            str(e),
        )

    except ValueError as e:
        return _error_response(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "INVALID_INPUT",
            str(e),
        )

    except Exception as e:
        logger.exception("[QUERY] Lỗi không xác định")
        return _error_response(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "INTERNAL_ERROR",
            f"Lỗi server khi xử lý truy vấn: {str(e)}",
        )


# ══════════════════════════════════════════════════════════════════
# ENDPOINT 3: Hide/Unhide tài liệu (Async — 202 Accepted)
# ══════════════════════════════════════════════════════════════════

@router.patch(
    "/docs/{doc_id}/visibility",
    response_model=AcceptedResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Ẩn/Hiện tài liệu — Bật/tắt truy xuất trong RAG",
    description=(
        "Hide: set is_hidden=true → tài liệu không xuất hiện khi search.\n"
        "Unhide: set is_hidden=false → tài liệu xuất hiện lại khi search.\n"
        "Async: trả 202, callback khi xong."
    ),
)
async def visibility_endpoint(
    doc_id: str,
    request: VisibilityRequest,
    background_tasks: BackgroundTasks,
) -> AcceptedResponse:
    """Xử lý hide/unhide tài liệu (async pattern)."""
    logger.info(
        "[VISIBILITY] Nhận request: doc_id=%s, action=%s, job_id=%s",
        doc_id,
        request.action,
        request.job_id,
    )

    if request.action == "hide":
        background_tasks.add_task(
            hide_document_background,
            doc_id=doc_id,
            job_id=request.job_id,
            callback_url=request.callback_url,
        )
    else:  # unhide
        background_tasks.add_task(
            unhide_document_background,
            doc_id=doc_id,
            job_id=request.job_id,
            callback_url=request.callback_url,
        )

    return AcceptedResponse(
        status="accepted",
        job_id=request.job_id,
    )


# ══════════════════════════════════════════════════════════════════
# ENDPOINT 4: Xóa tài liệu (Async — 202 Accepted)
# ══════════════════════════════════════════════════════════════════

@router.delete(
    "/ingest/{doc_id}",
    response_model=AcceptedResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Xóa vectors tài liệu — Dọn dẹp Qdrant",
    description=(
        "Xóa toàn bộ vectors có doc_id khỏi Qdrant.\n"
        "File gốc và lịch sử MySQL vẫn được giữ (Node.js quản lý).\n"
        "Async: trả 202, callback khi xong."
    ),
)
async def delete_endpoint(
    doc_id: str,
    request: DeleteRequest,
    background_tasks: BackgroundTasks,
) -> AcceptedResponse:
    """Xử lý xóa vectors tài liệu (async pattern)."""
    logger.info(
        "[DELETE] Nhận request: doc_id=%s, job_id=%s",
        doc_id,
        request.job_id,
    )

    background_tasks.add_task(
        delete_document_background,
        doc_id=doc_id,
        job_id=request.job_id,
        callback_url=request.callback_url,
    )

    return AcceptedResponse(
        status="accepted",
        job_id=request.job_id,
    )


# ══════════════════════════════════════════════════════════════════
# ENDPOINT 5: Health Check
# ══════════════════════════════════════════════════════════════════

@router.get(
    "/health",
    summary="Kiểm tra trạng thái service",
    description="Endpoint để monitoring/load balancer kiểm tra service còn hoạt động.",
)
async def health_check() -> dict:
    """Trả về trạng thái hoạt động của service."""
    return {
        "status": "healthy",
        "service": "rag-education-service",
        "version": "3.0.0",
    }
