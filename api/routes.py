"""
api/routes.py
-------------
Định nghĩa các API endpoints chính cho RAG microservice.
Giao tiếp nội bộ với Node.js backend qua HTTP.

Phiên bản v2:
- Query endpoint: tích hợp Query Router (CHIT_CHAT / RAG_REQUIRED).
- Bỏ subject_id khỏi query log (Single-turn, Global Search).

Endpoints:
  POST /api/query   — Truy vấn RAG (có router phân loại intent)
  POST /api/ingest  — Nạp tài liệu: đọc PDF, chia chunks, lưu vectors
  GET  /api/health  — Health check cho monitoring
"""

import logging
import time

from fastapi import APIRouter, HTTPException, status

from models.schemas import (
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
)
from services.ingestion import ingest_document
from services.rag_engine import process_query

logger = logging.getLogger(__name__)

# ── Khởi tạo router với prefix chung ────────────────────────────
router = APIRouter(prefix="/api", tags=["RAG"])


# ══════════════════════════════════════════════════════════════════
# ENDPOINT 1: Truy vấn RAG (với Query Router)
# ══════════════════════════════════════════════════════════════════

@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Truy vấn RAG — Hỏi đáp với Query Router",
    description=(
        "Nhận câu hỏi từ người dùng. Query Router sẽ phân loại intent:\n"
        "- CHIT_CHAT → LLM trả lời giao tiếp bình thường.\n"
        "- RAG_REQUIRED → Tìm kiếm Global trong Qdrant + Gemini LLM sinh "
        "câu trả lời kèm trích dẫn nguồn."
    ),
)
async def query_endpoint(request: QueryRequest) -> QueryResponse:
    """
    Xử lý truy vấn với Query Router.

    Luồng xử lý:
    1. Validate request (Pydantic tự động).
    2. Query Router phân loại intent.
    3. Rẽ nhánh: CHIT_CHAT → giao tiếp | RAG_REQUIRED → RAG pipeline.
    4. Trả về QueryResponse.
    """
    start_time = time.time()

    try:
        logger.info(
            "[QUERY] Nhận request: conv=%s, q='%s'",
            request.conversation_id,
            request.question[:80],
        )

        # Gọi RAG engine (có tích hợp Router bên trong)
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
        logger.error("[QUERY] File không tồn tại: %s", str(e))
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Không tìm thấy tài liệu: {str(e)}",
        )

    except ValueError as e:
        logger.error("[QUERY] Dữ liệu không hợp lệ: %s", str(e))
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Dữ liệu không hợp lệ: {str(e)}",
        )

    except Exception as e:
        logger.exception("[QUERY] Lỗi không xác định khi xử lý query")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi server khi xử lý truy vấn: {str(e)}",
        )


# ══════════════════════════════════════════════════════════════════
# ENDPOINT 2: Nạp tài liệu (Ingestion)
# ══════════════════════════════════════════════════════════════════

@router.post(
    "/ingest",
    response_model=IngestResponse,
    summary="Nạp tài liệu — Đọc PDF, chia chunks phân cấp, lưu vectors",
    description=(
        "Nhận thông tin tài liệu PDF, đọc nội dung, trích xuất heading hierarchy "
        "(chapter/section), chia chunks bằng SentenceSplitter (LlamaIndex), "
        "tạo embeddings bằng Gemini, và lưu vào Qdrant với metadata đầy đủ."
    ),
)
async def ingest_endpoint(request: IngestRequest) -> IngestResponse:
    """
    Xử lý nạp tài liệu vào Qdrant.

    Luồng xử lý:
    1. Validate request (Pydantic tự động).
    2. Đọc PDF → trích xuất headings → chia chunks → embedding → lưu Qdrant.
    3. Trả về IngestResponse với trạng thái và số chunks.
    """
    start_time = time.time()

    try:
        logger.info(
            "[INGEST] Nhận request: doc_id=%s, subject=%s, file=%s",
            request.doc_id,
            request.subject_id,
            request.file_path,
        )

        # Gọi service xử lý ingestion
        response = await ingest_document(request)

        elapsed = time.time() - start_time
        logger.info(
            "[INGEST] Hoàn tất trong %.2fs | status=%s | chunks=%d",
            elapsed,
            response.status,
            response.chunks_count,
        )

        return response

    except FileNotFoundError as e:
        logger.error("[INGEST] File không tồn tại: %s", str(e))
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Không tìm thấy file: {str(e)}",
        )

    except ValueError as e:
        logger.error("[INGEST] File không hợp lệ: %s", str(e))
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"File không hợp lệ: {str(e)}",
        )

    except Exception as e:
        logger.exception("[INGEST] Lỗi không xác định khi xử lý ingestion")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi server khi nạp tài liệu: {str(e)}",
        )


# ══════════════════════════════════════════════════════════════════
# ENDPOINT 3: Health Check
# ══════════════════════════════════════════════════════════════════

@router.get(
    "/health",
    summary="Kiểm tra trạng thái service",
    description="Endpoint để monitoring/load balancer kiểm tra service còn hoạt động.",
)
async def health_check() -> dict:
    """
    Trả về trạng thái hoạt động của service.
    Có thể mở rộng để kiểm tra kết nối Qdrant, Gemini API, v.v.
    """
    return {
        "status": "healthy",
        "service": "rag-education-service",
        "version": "2.0.0",
    }
