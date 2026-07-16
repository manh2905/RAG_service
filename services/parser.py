"""
services/parser.py
------------------
Parser đa định dạng: PDF, DOCX, TXT.

Theo sơ đồ 3 (Ingest Flow):
- Đọc cấu trúc tài liệu (LlamaParse / fallback)
- Bảo toàn từ ghép Tiếng Việt (Underthesea)
- Trích xuất heading hierarchy (chapter, section)

Strategy:
- Primary: LlamaParse (nếu có LLAMA_CLOUD_API_KEY)
- Fallback: pypdf (PDF) + python-docx (DOCX) + plain read (TXT)
"""

import logging
import re
from pathlib import Path
from typing import Optional

from core.config import get_settings

try:
    # pyrefly: ignore [missing-import]
    from underthesea import word_tokenize
    HAS_UNDERTHESEA = True
except ImportError:
    HAS_UNDERTHESEA = False

logger = logging.getLogger(__name__)

# Định dạng file được hỗ trợ
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".doc", ".txt"}


# ══════════════════════════════════════════════════════════════════
# HÀM CHÍNH: PARSE TÀI LIỆU
# ══════════════════════════════════════════════════════════════════

async def parse_document(file_path: str) -> list[dict]:
    """
    Parse tài liệu thành danh sách pages với text và metadata.

    Args:
        file_path: Đường dẫn tuyệt đối tới file.

    Returns:
        List[dict]: Mỗi item là {page_number, text, chapter, section}

    Raises:
        FileNotFoundError: File không tồn tại.
        ValueError: Định dạng file không hỗ trợ.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {file_path}")

    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Định dạng '{ext}' không được hỗ trợ. "
            f"Chỉ hỗ trợ: {', '.join(SUPPORTED_EXTENSIONS)}"
        )

    settings = get_settings()

    # Thử LlamaParse trước nếu có API key
    if settings.LLAMA_CLOUD_API_KEY:
        try:
            pages = await _parse_with_llamaparse(file_path, settings.LLAMA_CLOUD_API_KEY)
            if pages:
                logger.info("LlamaParse thành công: %d pages từ %s", len(pages), path.name)
                return _enrich_with_headings(pages)
        except Exception as e:
            logger.warning("LlamaParse thất bại, chuyển sang fallback: %s", str(e))

    # Fallback: parser cục bộ
    if ext == ".pdf":
        pages = _parse_pdf_fallback(file_path)
    elif ext in (".docx", ".doc"):
        pages = _parse_docx_fallback(file_path)
    elif ext == ".txt":
        pages = _parse_txt(file_path)
    else:
        raise ValueError(f"Không có parser cho định dạng: {ext}")

    logger.info("Fallback parser: %d pages từ %s", len(pages), path.name)
    return _enrich_with_headings(pages)


# ══════════════════════════════════════════════════════════════════
# LLAMAPARSE (PRIMARY)
# ══════════════════════════════════════════════════════════════════

async def _parse_with_llamaparse(file_path: str, api_key: str) -> list[dict]:
    """
    Parse tài liệu bằng LlamaParse (LlamaIndex Cloud).
    Hỗ trợ PDF, DOCX, TXT — trả về structured markdown.
    """
    try:
        # pyrefly: ignore [missing-import]
        from llama_parse import LlamaParse

        parser = LlamaParse(
            api_key=api_key,
            result_type="markdown",
            language="vi",
        )

        documents = await parser.aload_data(file_path)

        pages = []
        for i, doc in enumerate(documents, start=1):
            text = doc.text.strip()
            if text:
                pages.append({
                    "page_number": i,
                    "text": text,
                })

        return pages

    except ImportError:
        logger.warning("llama-parse chưa được cài. Dùng: pip install llama-parse")
        return []


# ══════════════════════════════════════════════════════════════════
# FALLBACK PARSERS
# ══════════════════════════════════════════════════════════════════

def _parse_pdf_fallback(file_path: str) -> list[dict]:
    """Đọc PDF bằng pypdf (fallback khi không có LlamaParse)."""
    try:
        # pyrefly: ignore [missing-import]
        from pypdf import PdfReader

        reader = PdfReader(file_path)
        pages = []

        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            text = text.strip()
            if text:
                pages.append({
                    "page_number": page_num,
                    "text": text,
                })

        return pages

    except Exception as e:
        logger.error("Lỗi khi đọc PDF '%s': %s", file_path, str(e))
        raise


def _parse_docx_fallback(file_path: str) -> list[dict]:
    """Đọc DOCX bằng python-docx (fallback)."""
    try:
        # pyrefly: ignore [missing-import]
        from docx import Document

        doc = Document(file_path)
        full_text = []
        for para in doc.paragraphs:
            if para.text.strip():
                full_text.append(para.text)

        # DOCX không có page concept rõ ràng → gộp thành 1 "page"
        # Chia giả lập: mỗi ~3000 ký tự = 1 page
        combined = "\n".join(full_text)
        pages = []
        page_size = 3000

        for i in range(0, len(combined), page_size):
            chunk = combined[i:i + page_size].strip()
            if chunk:
                pages.append({
                    "page_number": i // page_size + 1,
                    "text": chunk,
                })

        return pages

    except Exception as e:
        logger.error("Lỗi khi đọc DOCX '%s': %s", file_path, str(e))
        raise


def _parse_txt(file_path: str) -> list[dict]:
    """Đọc file TXT thuần."""
    try:
        path = Path(file_path)
        content = path.read_text(encoding="utf-8")

        if not content.strip():
            return []

        # Chia thành pages giả lập (~3000 ký tự)
        pages = []
        page_size = 3000

        for i in range(0, len(content), page_size):
            chunk = content[i:i + page_size].strip()
            if chunk:
                pages.append({
                    "page_number": i // page_size + 1,
                    "text": chunk,
                })

        return pages

    except Exception as e:
        logger.error("Lỗi khi đọc TXT '%s': %s", file_path, str(e))
        raise


# ══════════════════════════════════════════════════════════════════
# HEADING EXTRACTION + VIETNAMESE NORMALIZATION
# ══════════════════════════════════════════════════════════════════

def _enrich_with_headings(pages: list[dict]) -> list[dict]:
    """
    Bổ sung heading hierarchy (chapter, section) cho mỗi page.
    Heading có tính kế thừa: trang trước truyền cho trang sau.
    Đồng thời normalize text tiếng Việt nếu có Underthesea.
    """
    current_chapter = None
    current_section = None

    for page in pages:
        text = page["text"]

        # Normalize tiếng Việt
        text = _normalize_vietnamese(text)
        page["text"] = text

        # Trích xuất headings
        headings = _extract_headings(text)

        if headings["chapter"]:
            current_chapter = headings["chapter"]
            current_section = None  # Reset section khi vào chapter mới

        if headings["section"]:
            current_section = headings["section"]

        page["chapter"] = current_chapter or ""
        page["section"] = current_section or ""

    return pages


def _extract_headings(text: str) -> dict:
    """
    Trích xuất heading hierarchy từ text.
    Nhận diện pattern heading phổ biến trong tài liệu giáo dục.
    """
    chapter = None
    section = None

    lines = text.split("\n")

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        # ── Nhận diện H1 (Chapter) ───────────────────────────────
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
        section_match = re.match(
            r"^(\d+(?:\.\d+)+)\.?\s+(.+)",
            stripped,
        )
        if section_match:
            section = stripped
            continue

        section_match2 = re.match(
            r"^(?:Phần|Bài|Mục|Section)\s+[\dIVXivx]+[:\.\s]?\s*(.+)",
            stripped,
            re.IGNORECASE,
        )
        if section_match2:
            section = stripped
            continue

    return {"chapter": chapter, "section": section}


def _normalize_vietnamese(text: str) -> str:
    """
    Normalize text tiếng Việt bằng Underthesea (word segmentation).
    Nếu Underthesea không available, trả về text gốc (graceful fallback).
    """
    if HAS_UNDERTHESEA:
        try:
            return word_tokenize(text, format="text")
        except Exception as e:
            logger.warning("Underthesea lỗi, dùng text gốc: %s", str(e))
            return text
    return text
