"""
services/ingestion.py
---------------------
Xử lý luồng Ingestion: đọc PDF → chia chunks → embedding → lưu vào Qdrant.
Mỗi chunk được gắn metadata (doc_id, subject_id, page_number) để hỗ trợ
filter và trích dẫn nguồn khi truy vấn.
"""

import logging
import uuid
from pathlib import Path

from qdrant_client import models

from core.config import get_settings
from core.database import get_qdrant_client
from core.llm_setup import get_embedding_model
from models.schemas import IngestRequest, IngestResponse

logger = logging.getLogger(__name__)


async def ingest_document(request: IngestRequest) -> IngestResponse:
    """
    Luồng chính xử lý nạp tài liệu vào Qdrant.

    Bước 1: Đọc nội dung file PDF (từng trang).
    Bước 2: Chia nhỏ nội dung thành các chunks với metadata.
    Bước 3: Tạo vector embedding cho từng chunk.
    Bước 4: Lưu tất cả vectors + metadata vào Qdrant collection chung.

    Args:
        request: IngestRequest chứa doc_id, subject_id, file_path.

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

    # ── Bước 2: Chia thành chunks kèm metadata ────────────────────
    chunks = _split_into_chunks(
        pages=pages,
        doc_id=request.doc_id,
        subject_id=request.subject_id,
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
    )

    if not chunks:
        logger.warning("Không tạo được chunk nào từ tài liệu")
        return IngestResponse(
            status="warning",
            message="Tài liệu không có đủ nội dung để chia chunks",
            chunks_count=0,
        )

    logger.info("Đã chia thành %d chunks", len(chunks))

    # ── Bước 3: Tạo embeddings cho tất cả chunks ──────────────────
    embed_model = get_embedding_model()
    texts = [chunk["text"] for chunk in chunks]

    try:
        embeddings = await embed_model.aget_text_embedding_batch(texts)
    except Exception as e:
        logger.error("Lỗi khi tạo embedding: %s", str(e))
        raise

    logger.info("Đã tạo embeddings cho %d chunks ✓", len(embeddings))

    # ── Bước 4: Lưu vào Qdrant ────────────────────────────────────
    client = await get_qdrant_client()

    points = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        point = models.PointStruct(
            id=str(uuid.uuid4()),  # ID duy nhất cho mỗi vector
            vector=embedding,
            payload={
                "text": chunk["text"],
                "doc_id": chunk["doc_id"],
                "subject_id": chunk["subject_id"],
                "page_number": chunk["page_number"],
                "chunk_index": i,
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


def _split_into_chunks(
    pages: list[dict],
    doc_id: str,
    subject_id: str,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
) -> list[dict]:
    """
    Chia nội dung các trang thành các chunks nhỏ hơn.
    Sử dụng thuật toán RecursiveCharacterTextSplitter đơn giản:
    chia theo ký tự với overlap giữa các chunks liền kề.

    Mỗi chunk luôn kèm metadata:
    - doc_id: ID tài liệu gốc
    - subject_id: ID môn học (dùng để filter trong Qdrant)
    - page_number: Số trang nguồn

    Args:
        pages: Danh sách các trang đã đọc từ PDF.
        doc_id: ID của tài liệu.
        subject_id: ID môn học.
        chunk_size: Kích thước tối đa mỗi chunk (ký tự).
        chunk_overlap: Số ký tự overlap giữa 2 chunks liên tiếp.

    Returns:
        List[dict]: Danh sách chunks, mỗi chunk là dict chứa text + metadata.
    """
    chunks = []

    # Dấu phân tách ưu tiên: đoạn văn → câu → dấu phẩy → khoảng trắng
    separators = ["\n\n", "\n", ". ", ", ", " "]

    for page in pages:
        text = page["text"]
        page_number = page["page_number"]

        # Chia text của trang thành các chunks
        page_chunks = _recursive_split(text, chunk_size, chunk_overlap, separators)

        for chunk_text in page_chunks:
            chunk_text = chunk_text.strip()
            if len(chunk_text) < 20:
                # Bỏ qua các chunk quá ngắn (nhiễu)
                continue

            chunks.append({
                "text": chunk_text,
                "doc_id": doc_id,
                "subject_id": subject_id,
                "page_number": page_number,
            })

    return chunks


def _recursive_split(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    separators: list[str],
) -> list[str]:
    """
    Thuật toán chia text đệ quy theo danh sách dấu phân tách ưu tiên.
    Ưu tiên giữ nguyên cấu trúc đoạn văn / câu khi chia.

    Args:
        text: Văn bản cần chia.
        chunk_size: Kích thước tối đa.
        chunk_overlap: Số ký tự chồng lấp.
        separators: Danh sách dấu phân tách theo thứ tự ưu tiên.

    Returns:
        List[str]: Danh sách các đoạn text đã chia.
    """
    # Nếu text đã đủ ngắn, trả về nguyên
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    # Tìm separator phù hợp nhất (ưu tiên từ trái sang phải)
    chosen_separator = separators[-1] if separators else ""
    for sep in separators:
        if sep in text:
            chosen_separator = sep
            break

    # Chia text theo separator đã chọn
    parts = text.split(chosen_separator)
    chunks = []
    current_chunk = ""

    for part in parts:
        # Thêm separator lại (trừ trường hợp separator là khoảng trắng)
        candidate = part if not current_chunk else current_chunk + chosen_separator + part

        if len(candidate) <= chunk_size:
            current_chunk = candidate
        else:
            # Lưu chunk hiện tại nếu có nội dung
            if current_chunk.strip():
                chunks.append(current_chunk)

            # Nếu phần hiện tại vẫn lớn hơn chunk_size,
            # đệ quy chia tiếp với separator nhỏ hơn
            if len(part) > chunk_size:
                remaining_seps = separators[separators.index(chosen_separator) + 1:] if chosen_separator in separators else []
                if remaining_seps:
                    sub_chunks = _recursive_split(part, chunk_size, chunk_overlap, remaining_seps)
                    chunks.extend(sub_chunks)
                else:
                    # Không còn separator nào → cắt cứng
                    for i in range(0, len(part), chunk_size - chunk_overlap):
                        chunks.append(part[i : i + chunk_size])
                current_chunk = ""
            else:
                current_chunk = part

    # Thêm chunk cuối cùng
    if current_chunk.strip():
        chunks.append(current_chunk)

    # Áp dụng overlap: thêm phần đuôi chunk trước vào đầu chunk sau
    if chunk_overlap > 0 and len(chunks) > 1:
        overlapped_chunks = [chunks[0]]
        for i in range(1, len(chunks)):
            prev_tail = chunks[i - 1][-chunk_overlap:]
            overlapped_chunks.append(prev_tail + chunks[i])
        chunks = overlapped_chunks

    return chunks
