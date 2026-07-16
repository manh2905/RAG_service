"""
services/ingestion.py
---------------------
Xử lý luồng Ingestion theo pattern async + callback.

Theo sơ đồ 3+4+5:
1. Nhận request → trả 202 ngay.
2. Background: Parse → Chunk → Embed → Store Qdrant.
3. Callback PROGRESS ở mỗi bước.
4. Callback SUCCEEDED khi xong (kèm chunk manifest).
5. Callback FAILED nếu có lỗi.

Phiên bản v3:
- Hỗ trợ PDF, DOCX, TXT qua parser.py (LlamaParse + fallback).
- Thêm is_hidden=false vào payload Qdrant.
- Callback mechanism thay vì trả kết quả sync.
"""

import logging
import uuid

# pyrefly: ignore [missing-import]
from qdrant_client import models

# pyrefly: ignore [missing-import]
from llama_index.core.node_parser import SentenceSplitter
# pyrefly: ignore [missing-import]
from llama_index.core.schema import Document as LlamaDocument

from core.config import get_settings
from core.database import get_qdrant_client
from core.llm_setup import get_embedding_model
from models.schemas import (
    IngestRequest,
    ChunkManifestItem,
)
from services.parser import parse_document
from services.callback import (
    send_progress,
    send_succeeded_ingest,
    send_failed,
)

logger = logging.getLogger(__name__)


async def ingest_document_background(request: IngestRequest) -> None:
    """
    Xử lý nạp tài liệu vào Qdrant (chạy trong BackgroundTasks).

    Bước 1: Parse file (PDF/DOCX/TXT).
    Bước 2: Chia chunks bằng SentenceSplitter.
    Bước 3: Tạo embeddings.
    Bước 4: Lưu vào Qdrant.
    Mỗi bước gửi callback PROGRESS.
    Cuối cùng gửi SUCCEEDED hoặc FAILED.
    """
    settings = get_settings()
    callback_url = request.callback_url
    job_id = request.job_id

    try:
        # ── Bước 1: Parse tài liệu ────────────────────────────────
        await send_progress(callback_url, job_id, "parsing")

        logger.info("[INGEST] Parsing file: %s", request.file_path)
        pages = await parse_document(request.file_path)

        if not pages:
            await send_failed(
                callback_url, job_id,
                "EMPTY_DOCUMENT", "Không đọc được nội dung từ file"
            )
            return

        logger.info("[INGEST] Đã parse %d pages", len(pages))

        # ── Bước 2: Chia chunks ───────────────────────────────────
        await send_progress(callback_url, job_id, "chunking")

        documents = _build_llama_documents(
            pages=pages,
            doc_id=request.doc_id,
            subject_id=request.subject_id,
            teacher_metadata=request.teacher_metadata or {},
        )

        splitter = SentenceSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
        )
        nodes = splitter.get_nodes_from_documents(documents)

        if not nodes:
            await send_failed(
                callback_url, job_id,
                "NO_CHUNKS", "Tài liệu không có đủ nội dung để chia chunks"
            )
            return

        logger.info("[INGEST] Tạo được %d chunks", len(nodes))

        # ── Bước 3: Embedding ─────────────────────────────────────
        await send_progress(callback_url, job_id, "embedding")

        embed_model = get_embedding_model()
        texts = [node.get_content() for node in nodes]

        try:
            embeddings = await embed_model.aget_text_embedding_batch(texts)
        except Exception as e:
            await send_failed(
                callback_url, job_id,
                "EMBEDDING_ERROR", f"Lỗi khi tạo embedding: {str(e)}"
            )
            return

        logger.info("[INGEST] Đã tạo embeddings cho %d chunks", len(embeddings))

        # ── Bước 4: Lưu vào Qdrant ───────────────────────────────
        await send_progress(callback_url, job_id, "indexing")

        client = await get_qdrant_client()
        points = []
        chunk_manifest = []

        for i, (node, embedding) in enumerate(zip(nodes, embeddings)):
            metadata = node.metadata
            chunk_id = str(uuid.uuid4())

            point = models.PointStruct(
                id=chunk_id,
                vector=embedding,
                payload={
                    "text": node.get_content(),
                    "doc_id": metadata.get("doc_id", request.doc_id),
                    "subject_id": metadata.get("subject_id", request.subject_id),
                    "page_number": metadata.get("page_number", 0),
                    "chapter": metadata.get("chapter", ""),
                    "section": metadata.get("section", ""),
                    "chunk_index": i,
                    "is_hidden": False,  # Mặc định VISIBLE
                    # Teacher metadata
                    **{f"teacher_{k}": v for k, v in (request.teacher_metadata or {}).items()},
                },
            )
            points.append(point)

            # Chunk manifest cho callback
            chunk_manifest.append(
                ChunkManifestItem(
                    chunk_id=chunk_id,
                    chunk_index=i,
                    page_number=metadata.get("page_number", 0),
                    chapter=metadata.get("chapter", ""),
                    section=metadata.get("section", ""),
                    text_preview=node.get_content()[:50],
                )
            )

        # Upload theo batch
        BATCH_SIZE = 100
        for batch_start in range(0, len(points), BATCH_SIZE):
            batch = points[batch_start: batch_start + BATCH_SIZE]
            client.upsert(
                collection_name=settings.QDRANT_COLLECTION_NAME,
                points=batch,
            )

        logger.info("[INGEST] Lưu thành công %d chunks vào Qdrant", len(points))

        # ── Callback SUCCEEDED ────────────────────────────────────
        await send_succeeded_ingest(
            callback_url=callback_url,
            job_id=job_id,
            chunks_count=len(points),
            chunk_manifest=[m.model_dump() for m in chunk_manifest],
        )

        logger.info("[INGEST] Hoàn tất: doc_id=%s, %d chunks ✓", request.doc_id, len(points))

    except FileNotFoundError as e:
        logger.error("[INGEST] File không tồn tại: %s", str(e))
        await send_failed(callback_url, job_id, "FILE_NOT_FOUND", str(e))

    except ValueError as e:
        logger.error("[INGEST] File không hợp lệ: %s", str(e))
        await send_failed(callback_url, job_id, "INVALID_FORMAT", str(e))

    except Exception as e:
        logger.exception("[INGEST] Lỗi không xác định")
        await send_failed(callback_url, job_id, "INTERNAL_ERROR", str(e))


# ══════════════════════════════════════════════════════════════════
# HÀM PHỤ TRỢ
# ══════════════════════════════════════════════════════════════════

def _build_llama_documents(
    pages: list[dict],
    doc_id: str,
    subject_id: str,
    teacher_metadata: dict,
) -> list[LlamaDocument]:
    """
    Chuyển đổi pages từ parser thành LlamaIndex Documents kèm metadata.
    """
    documents = []

    for page in pages:
        metadata = {
            "doc_id": doc_id,
            "subject_id": subject_id,
            "page_number": page.get("page_number", 0),
            "chapter": page.get("chapter", ""),
            "section": page.get("section", ""),
        }

        doc = LlamaDocument(
            text=page["text"],
            metadata=metadata,
        )
        documents.append(doc)

    return documents
