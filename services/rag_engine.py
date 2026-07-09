"""
services/rag_engine.py
----------------------
Luồng RAG chính: nhận câu hỏi → tìm kiếm ngữ cảnh trong Qdrant
→ xây dựng prompt → gọi Gemini LLM → trích xuất trích dẫn → trả kết quả.

Bao gồm các guardrail:
- Similarity threshold: nếu không tìm thấy chunk liên quan → trả no_answer.
- Strict prompting: ép LLM chỉ dùng context, bắt buộc trích dẫn nguồn.
"""

import logging
import re
from typing import Any

from qdrant_client import models

from core.config import get_settings
from core.database import get_qdrant_client
from core.llm_setup import get_embedding_model, get_llm
from models.schemas import (
    Citation,
    MessageHistory,
    QueryRequest,
    QueryResponse,
)

logger = logging.getLogger(__name__)


async def process_query(request: QueryRequest) -> QueryResponse:
    """
    Xử lý truy vấn RAG đầy đủ theo các bước:

    1. Embedding câu hỏi thành vector.
    2. Tìm Top-K chunks tương tự trong Qdrant (filter theo subject_id).
    3. Kiểm tra similarity threshold → nếu quá thấp trả no_answer.
    4. Xây dựng prompt kết hợp history + chunks + question.
    5. Gọi Gemini LLM sinh câu trả lời.
    6. Trích xuất citations từ metadata chunks.
    7. Trả về QueryResponse hoàn chỉnh.

    Args:
        request: QueryRequest chứa câu hỏi, subject_id, lịch sử hội thoại.

    Returns:
        QueryResponse với câu trả lời, trích dẫn, và đánh giá confidence.
    """
    settings = get_settings()

    logger.info(
        "Bắt đầu xử lý query: subject_id=%s, conversation_id=%s, question='%s'",
        request.subject_id,
        request.conversation_id,
        request.question[:100],  # Log 100 ký tự đầu để tránh quá dài
    )

    # ── Bước 1: Embedding câu hỏi ─────────────────────────────────
    embed_model = get_embedding_model()

    try:
        question_vector = await embed_model.aget_text_embedding(request.question)
    except Exception as e:
        logger.error("Lỗi khi tạo embedding cho câu hỏi: %s", str(e))
        raise

    # ── Bước 2: Tìm kiếm Top-K chunks trong Qdrant ────────────────
    # BẮT BUỘC filter theo subject_id để chỉ tìm trong phạm vi môn học
    client = await get_qdrant_client()

    try:
        search_results = client.query_points(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            query=question_vector,
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="subject_id",
                        match=models.MatchValue(value=request.subject_id),
                    )
                ]
            ),
            limit=settings.TOP_K,
            with_payload=True,
        )
    except Exception as e:
        logger.error("Lỗi khi truy vấn Qdrant: %s", str(e))
        raise

    results = search_results.points
    logger.info("Tìm thấy %d kết quả từ Qdrant", len(results))

    # ── Bước 3: Guardrail — kiểm tra similarity threshold ─────────
    # Lọc bỏ các kết quả có điểm tương đồng quá thấp
    filtered_results = [
        r for r in results
        if r.score >= settings.SIMILARITY_THRESHOLD
    ]

    if not filtered_results:
        logger.warning(
            "Không có chunk nào vượt ngưỡng similarity %.2f. "
            "Điểm cao nhất: %.4f",
            settings.SIMILARITY_THRESHOLD,
            results[0].score if results else 0.0,
        )
        return QueryResponse(
            answer="Không đủ dữ liệu/Không tìm thấy thông tin liên quan "
                   "trong tài liệu của môn học này.",
            citations=[],
            confidence="low",
            no_answer=True,
        )

    logger.info(
        "Có %d/%d chunks vượt ngưỡng similarity (top score: %.4f)",
        len(filtered_results),
        len(results),
        filtered_results[0].score,
    )

    # ── Bước 4: Xây dựng prompt cho LLM ───────────────────────────
    context_text = _build_context(filtered_results)
    prompt = _build_prompt(
        question=request.question,
        context=context_text,
        history=request.history,
    )

    # ── Bước 5: Gọi Gemini LLM sinh câu trả lời ──────────────────
    llm = get_llm()

    try:
        llm_response = await llm.acomplete(prompt)
        answer_text = llm_response.text.strip()
    except Exception as e:
        logger.error("Lỗi khi gọi Gemini LLM: %s", str(e))
        raise

    logger.info("LLM đã sinh câu trả lời (%d ký tự)", len(answer_text))

    # ── Bước 6: Trích xuất citations từ câu trả lời ───────────────
    citations = _extract_citations(answer_text, filtered_results)

    # ── Bước 7: Đánh giá confidence ───────────────────────────────
    confidence = _evaluate_confidence(filtered_results)

    return QueryResponse(
        answer=answer_text,
        citations=citations,
        confidence=confidence,
        no_answer=False,
    )


def _build_context(results: list[Any]) -> str:
    """
    Xây dựng đoạn context từ danh sách kết quả Qdrant.
    Đánh số mỗi chunk [1], [2], ... để LLM có thể trích dẫn.

    Args:
        results: Danh sách ScoredPoint từ Qdrant.

    Returns:
        str: Đoạn context đã format, sẵn sàng đưa vào prompt.
    """
    context_parts = []

    for idx, result in enumerate(results, start=1):
        payload = result.payload
        text = payload.get("text", "")
        doc_id = payload.get("doc_id", "N/A")
        page = payload.get("page_number", 0)

        context_parts.append(
            f"[{idx}] (Tài liệu: {doc_id}, Trang: {page})\n{text}"
        )

    return "\n\n---\n\n".join(context_parts)


def _build_prompt(
    question: str,
    context: str,
    history: list[MessageHistory],
) -> str:
    """
    Xây dựng prompt hoàn chỉnh cho LLM, bao gồm:
    - System instruction (strict rules)
    - Lịch sử hội thoại (nếu có)
    - Context từ Qdrant
    - Câu hỏi hiện tại

    RÀNG BUỘC QUAN TRỌNG:
    - LLM chỉ được dùng context đã cung cấp.
    - Bắt buộc trích dẫn nguồn [1], [2], ... sau mỗi ý.
    - Không được tự bịa thông tin.

    Args:
        question: Câu hỏi của người dùng.
        context: Đoạn context đã format từ Qdrant chunks.
        history: Lịch sử hội thoại trước đó.

    Returns:
        str: Prompt đầy đủ gửi tới LLM.
    """
    # ── System instruction — ép buộc LLM tuân thủ ──────────────
    system_instruction = (
        "Bạn là trợ lý giáo dục thông minh. Hãy tuân thủ NGHIÊM NGẶT các quy tắc sau:\n\n"
        "1. CHỈ sử dụng thông tin từ phần CONTEXT bên dưới để trả lời câu hỏi.\n"
        "2. BẮT BUỘC trích dẫn nguồn ở định dạng [1], [2], ... đằng sau mỗi ý "
        "tương ứng với số thứ tự trong CONTEXT.\n"
        "3. KHÔNG ĐƯỢC tự bịa thông tin hoặc sử dụng kiến thức bên ngoài.\n"
        "4. Nếu CONTEXT không đủ thông tin để trả lời, hãy nói rõ ràng rằng "
        "không tìm thấy thông tin.\n"
        "5. Trả lời bằng tiếng Việt, rõ ràng, có cấu trúc.\n"
    )

    # ── Lịch sử hội thoại (nếu có) ─────────────────────────────
    history_text = ""
    if history:
        history_lines = []
        for msg in history[-6:]:  # Giới hạn 6 tin nhắn gần nhất
            role_label = "Người dùng" if msg.role == "user" else "Trợ lý"
            history_lines.append(f"{role_label}: {msg.content}")
        history_text = (
            "\n--- LỊCH SỬ HỘI THOẠI ---\n"
            + "\n".join(history_lines)
            + "\n--- HẾT LỊCH SỬ ---\n"
        )

    # ── Ghép prompt hoàn chỉnh ──────────────────────────────────
    full_prompt = (
        f"{system_instruction}\n"
        f"{history_text}\n"
        f"--- CONTEXT ---\n"
        f"{context}\n"
        f"--- HẾT CONTEXT ---\n\n"
        f"Câu hỏi: {question}\n\n"
        f"Hãy trả lời câu hỏi trên dựa HOÀN TOÀN vào CONTEXT. "
        f"Nhớ trích dẫn nguồn [1], [2], ... sau mỗi ý."
    )

    return full_prompt


def _extract_citations(
    answer: str,
    results: list[Any],
) -> list[Citation]:
    """
    Trích xuất danh sách Citation từ câu trả lời của LLM.
    Tìm các pattern [1], [2], ... trong answer và khớp với metadata
    của chunks tương ứng.

    Args:
        answer: Câu trả lời từ LLM (chứa [1], [2], ...).
        results: Danh sách ScoredPoint từ Qdrant (đã filter).

    Returns:
        List[Citation]: Danh sách trích dẫn nguồn.
    """
    # Tìm tất cả số trong dấu ngoặc vuông: [1], [2], [3], ...
    cited_indices = set()
    matches = re.findall(r"\[(\d+)\]", answer)

    for match in matches:
        idx = int(match)
        if 1 <= idx <= len(results):
            cited_indices.add(idx)

    # Xây dựng danh sách Citation từ metadata các chunks được trích dẫn
    citations = []
    for idx in sorted(cited_indices):
        result = results[idx - 1]  # Convert 1-indexed → 0-indexed
        payload = result.payload

        citations.append(
            Citation(
                doc_id=payload.get("doc_id", "unknown"),
                page_number=payload.get("page_number", 0),
                snippet=payload.get("text", "")[:200],  # Lấy 200 ký tự đầu
            )
        )

    logger.info("Trích xuất được %d trích dẫn từ câu trả lời", len(citations))
    return citations


def _evaluate_confidence(results: list[Any]) -> str:
    """
    Đánh giá mức độ tin cậy dựa trên điểm similarity của các chunks.

    Quy tắc:
    - high:   Điểm trung bình >= 0.7
    - medium: Điểm trung bình >= 0.5
    - low:    Điểm trung bình < 0.5

    Args:
        results: Danh sách ScoredPoint đã vượt ngưỡng.

    Returns:
        str: "high", "medium", hoặc "low".
    """
    if not results:
        return "low"

    avg_score = sum(r.score for r in results) / len(results)

    if avg_score >= 0.7:
        return "high"
    elif avg_score >= 0.5:
        return "medium"
    else:
        return "low"
