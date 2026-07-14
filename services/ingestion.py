"""
services/ingestion.py
---------------------
Xử lý luồng Ingestion: đọc PDF → chia chunks phân cấp → embedding → lưu vào Qdrant.

Phiên bản v2:
- Sử dụng SentenceSplitter của LlamaIndex thay cho TextSplitter tự viết.
- Trích xuất Heading hierarchy (H1→chapter, H2/H3→section) từ nội dung PDF.
- Mỗi chunk được gắn metadata: doc_id, subject_id, page_number, chapter, section.
- Hỗ trợ teacher_metadata bổ sung.
"""

import logging
import re
import uuid
from pathlib import Path

from qdrant_client import models

from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document as LlamaDocument

from core.config import get_settings
from core.database import get_qdrant_client
from core.llm_setup import get_embedding_model
from models.schemas import IngestRequest, IngestResponse

logger = logging.getLogger(__name__)


async def ingest_document(request: IngestRequest) -> IngestResponse:
    """
    Luồng chính xử lý nạp tài liệu vào Qdrant.

    Bước 1: Đọc nội dung file PDF (từng trang).
    Bước 2: Trích xuất heading hierarchy (chapter/section) từ text.
    Bước 3: Chia nhỏ bằng SentenceSplitter (LlamaIndex) với metadata.
    Bước 4: Tạo vector embedding cho từng chunk.
    Bước 5: Lưu tất cả vectors + metadata vào Qdrant.

    Args:
        request: IngestRequest chứa doc_id, subject_id, file_path, teacher_metadata.

    Returns:
        IngestResponse với trạng thái và số lượng chunks đã lưu.
    """
    settings = get_settings()

    logger.info(
        "Bắt đầu ingest tài liệu: doc_id=%s, subject_id=%s, file=%s",
        request.doc_id,
        request.subject_id,
        request.file_path,
    )

    # ── Bước 1: Đọc nội dung PDF ──────────────────────────────────
    pages = _read_pdf(request.file_path)
    if not pages:
        logger.warning("Không đọc được nội dung từ file: %s", request.file_path)
        return IngestResponse(
            status="warning",
            message="Không đọc được nội dung từ file PDF",
            chunks_count=0,
        )

    logger.info("Đã đọc %d trang từ PDF", len(pages))

    # ── Bước 2: Trích xuất heading hierarchy + tạo LlamaIndex Documents ──
    documents = _build_documents_with_headings(
        pages=pages,
        doc_id=request.doc_id,
        subject_id=request.subject_id,
        teacher_metadata=request.teacher_metadata or {},
    )

    logger.info("Đã tạo %d documents với heading metadata", len(documents))

    # ── Bước 3: Chia chunks bằng SentenceSplitter (LlamaIndex) ────
    splitter = SentenceSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
    )

    nodes = splitter.get_nodes_from_documents(documents)

    if not nodes:
        logger.warning("Không tạo được chunk nào từ tài liệu")
        return IngestResponse(
            status="warning",
            message="Tài liệu không có đủ nội dung để chia chunks",
            chunks_count=0,
        )

    logger.info("SentenceSplitter tạo ra %d nodes/chunks", len(nodes))

    # ── Bước 4: Tạo embeddings cho tất cả chunks ──────────────────
    embed_model = get_embedding_model()
    texts = [node.get_content() for node in nodes]

    try:
        embeddings = await embed_model.aget_text_embedding_batch(texts)
    except Exception as e:
        logger.error("Lỗi khi tạo embedding: %s", str(e))
        raise

    logger.info("Đã tạo embeddings cho %d chunks ✓", len(embeddings))

    # ── Bước 5: Lưu vào Qdrant ────────────────────────────────────
    client = await get_qdrant_client()

    points = []
    for i, (node, embedding) in enumerate(zip(nodes, embeddings)):
        metadata = node.metadata

        point = models.PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding,
            payload={
                "text": node.get_content(),
                "doc_id": metadata.get("doc_id", request.doc_id),
                "subject_id": metadata.get("subject_id", request.subject_id),
                "page_number": metadata.get("page_number", 0),
                "chapter": metadata.get("chapter", ""),
                "section": metadata.get("section", ""),
                "chunk_index": i,
                # Teacher metadata gộp vào payload
                **{f"teacher_{k}": v for k, v in (request.teacher_metadata or {}).items()},
            },
        )
        points.append(point)

    # Upload theo batch để tối ưu hiệu suất
    BATCH_SIZE = 100
    for batch_start in range(0, len(points), BATCH_SIZE):
        batch = points[batch_start : batch_start + BATCH_SIZE]
        client.upsert(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            points=batch,
        )
        logger.info(
            "Đã lưu batch %d/%d (%d points)",
            batch_start // BATCH_SIZE + 1,
            (len(points) + BATCH_SIZE - 1) // BATCH_SIZE,
            len(batch),
        )

    logger.info(
        "Hoàn tất ingest: doc_id=%s, tổng %d chunks ✓",
        request.doc_id,
        len(points),
    )

    return IngestResponse(
        status="success",
        message=f"Đã nạp thành công {len(points)} chunks từ tài liệu",
        chunks_count=len(points),
    )


# ══════════════════════════════════════════════════════════════════
# HÀM PHỤ TRỢ (PRIVATE HELPERS)
# ══════════════════════════════════════════════════════════════════

def _read_pdf(file_path: str) -> list[dict]:
    """
    Đọc file PDF và trả về danh sách các trang.
    Mỗi trang là dict với key: 'page_number' (1-indexed) và 'text'.

    Args:
        file_path: Đường dẫn tuyệt đối tới file PDF.

    Returns:
        List[dict]: Danh sách các trang đã đọc.

    Raises:
        FileNotFoundError: Nếu file không tồn tại.
        ValueError: Nếu file không phải PDF.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {file_path}")

    if not path.suffix.lower() == ".pdf":
        raise ValueError(f"File không phải PDF: {file_path}")

    try:
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        pages = []

        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            text = text.strip()

            if text:  # Chỉ thêm trang có nội dung
                pages.append({
                    "page_number": page_num,
                    "text": text,
                })

        return pages

    except Exception as e:
        logger.error("Lỗi khi đọc PDF '%s': %s", file_path, str(e))
        raise


def _extract_headings(text: str) -> dict:
    """
    Trích xuất heading hierarchy từ text của một trang PDF.
    Nhận diện các pattern heading phổ biến trong tài liệu giáo dục:

    H1 (Chapter):
      - "Chương 1: ...", "CHƯƠNG I: ...", "Chapter 1: ..."
      - Dòng VIẾT HOA TOÀN BỘ (>= 5 ký tự, <= 100 ký tự)

    H2/H3 (Section):
      - "1.1 ...", "1.1. ...", "I.1. ..."
      - "Phần 1: ...", "Bài 1: ..."

    Args:
        text: Nội dung text của một trang.

    Returns:
        dict: {"chapter": str | None, "section": str | None}
    """
    chapter = None
    section = None

    lines = text.split("\n")

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        # ── Nhận diện H1 (Chapter) ───────────────────────────────
        # Pattern: "Chương X:", "CHƯƠNG X:", "Chapter X:"
        chapter_match = re.match(
            r"^(?:Chương|CHƯƠNG|Chapter)\s+[\dIVXivx]+[:\.\s]?\s*(.+)",
            stripped,
            re.IGNORECASE,
        )
        if chapter_match:
            chapter = stripped
            continue

        # Pattern: Dòng VIẾT HOA hoàn toàn (tiêu đề chương)
        if (
            stripped.isupper()
            and 5 <= len(stripped) <= 100
            and not stripped.startswith(("HTTP", "URL", "ISBN"))
        ):
            chapter = stripped
            continue

        # ── Nhận diện H2/H3 (Section) ───────────────────────────
        # Pattern: "1.1 ...", "1.1. ...", "2.3.1 ..."
        section_match = re.match(
            r"^(\d+(?:\.\d+)+)\.?\s+(.+)",
            stripped,
        )
        if section_match:
            section = stripped
            continue

        # Pattern: "Phần X:", "Bài X:", "Mục X:"
        section_match2 = re.match(
            r"^(?:Phần|Bài|Mục|Section)\s+[\dIVXivx]+[:\.\s]?\s*(.+)",
            stripped,
            re.IGNORECASE,
        )
        if section_match2:
            section = stripped
            continue

    return {"chapter": chapter, "section": section}


def _build_documents_with_headings(
    pages: list[dict],
    doc_id: str,
    subject_id: str,
    teacher_metadata: dict,
) -> list[LlamaDocument]:
    """
    Chuyển đổi các trang PDF thành LlamaIndex Documents, kèm theo
    heading metadata (chapter, section) được trích xuất từ nội dung.

    Heading có tính kế thừa: chapter/section ở trang trước sẽ được
    kế thừa cho các trang sau nếu trang sau không có heading mới.

    Args:
        pages: Danh sách trang đã đọc từ PDF.
        doc_id: ID tài liệu.
        subject_id: ID môn học.
        teacher_metadata: Metadata bổ sung từ giáo viên.

    Returns:
        List[LlamaDocument]: Documents sẵn sàng đưa vào SentenceSplitter.
    """
    documents = []

    # Heading kế thừa: trang sau dùng heading trang trước nếu không có heading mới
    current_chapter = None
    current_section = None

    for page in pages:
        text = page["text"]
        page_number = page["page_number"]

        # Trích xuất heading từ trang hiện tại
        headings = _extract_headings(text)

        # Cập nhật heading hiện tại (kế thừa nếu không tìm thấy mới)
        if headings["chapter"]:
            current_chapter = headings["chapter"]
            current_section = None  # Reset section khi vào chapter mới
        if headings["section"]:
            current_section = headings["section"]

        # Tạo LlamaIndex Document với metadata đầy đủ
        metadata = {
            "doc_id": doc_id,
            "subject_id": subject_id,
            "page_number": page_number,
            "chapter": current_chapter or "",
            "section": current_section or "",
        }

        doc = LlamaDocument(
            text=text,
            metadata=metadata,
        )
        documents.append(doc)

    logger.info(
        "Xây dựng %d documents: last_chapter='%s', last_section='%s'",
        len(documents),
        current_chapter or "N/A",
        current_section or "N/A",
    )

    return documents
