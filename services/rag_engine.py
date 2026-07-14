"""
services/rag_engine.py
----------------------
Luồng RAG chính với Query Router:
  1. Router: phân loại câu hỏi → CHIT_CHAT hoặc RAG_REQUIRED.
  2. Nếu CHIT_CHAT → LLM trả lời giao tiếp bình thường (không RAG).
  3. Nếu RAG_REQUIRED → Semantic Search (Global) → Prompt → LLM → Citations.

Phiên bản v2:
- Thêm bước Query Router (Structured Output, temperature=0).
- Bỏ filter subject_id → Global Search toàn bộ Vector DB.
- Chuyển sang Single-turn (bỏ history trong prompt).
- Citation bao gồm chapter và section từ heading metadata.
"""

import json
import logging
import re
from typing import Any

from qdrant_client import models

from core.config import get_settings
from core.database import get_qdrant_client
from core.llm_setup import get_embedding_model, get_llm, get_router_llm
from models.schemas import (
    Citation,
    QueryIntent,
    QueryRequest,
    QueryResponse,
)

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════
# BƯỚC 1: QUERY ROUTER — Phân loại ý định câu hỏi
# ══════════════════════════════════════════════════════════════════

async def _classify_intent(question: str) -> QueryIntent:
    """
    Sử dụng LLM (temperature=0) để phân loại ý định câu hỏi.
    LLM được ép trả về Structured Output theo schema QueryIntent.

    Strict prompt đảm bảo LLM KHÔNG trả lời câu hỏi,
    chỉ phân loại là CHIT_CHAT hoặc RAG_REQUIRED.

    Args:
        question: Câu hỏi của người dùng.

    Returns:
        QueryIntent: Kết quả phân loại intent.
    """
    router_llm = get_router_llm()

    # Prompt ép buộc LLM chỉ phân loại, KHÔNG trả lời
    router_prompt = (
        "Bạn là một bộ phân loại câu hỏi. Nhiệm vụ DUY NHẤT của bạn là xác định "
        "ý định của câu hỏi dưới đây.\n\n"
        "QUY TẮC:\n"
        "- Chỉ phân loại câu hỏi là CHIT_CHAT hay RAG_REQUIRED.\n"
        "- TUYỆT ĐỐI KHÔNG trả lời câu hỏi.\n"
        "- TUYỆT ĐỐI KHÔNG giải thích lý do.\n"
        "- Chỉ trả về JSON object duy nhất.\n\n"
        "ĐỊNH NGHĨA:\n"
        "- CHIT_CHAT: Câu chào hỏi, cảm ơn, tạm biệt, hỏi thăm sức khoẻ, "
        "nói chuyện phiếm, câu không cần tra cứu tài liệu.\n"
        "- RAG_REQUIRED: Câu hỏi về kiến thức, bài học, khái niệm, "
        "yêu cầu giải thích nội dung học thuật, cần tra cứu tài liệu.\n\n"
        "RESPONSE FORMAT (JSON):\n"
        '{"intent": "CHIT_CHAT"} hoặc {"intent": "RAG_REQUIRED"}\n\n'
        f'Câu hỏi: "{question}"\n\n'
        "Trả về JSON:"
    )

    try:
        response = await router_llm.acomplete(router_prompt)
        response_text = response.text.strip()

        # Parse JSON response từ LLM
        # Xử lý trường hợp LLM bọc trong ```json ... ```
        if "```" in response_text:
            json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response_text, re.DOTALL)
            if json_match:
                response_text = json_match.group(1)

        parsed = json.loads(response_text)
        intent = QueryIntent(**parsed)

        logger.info("Query Router phân loại: %s", intent.intent)
        return intent

    except (json.JSONDecodeError, Exception) as e:
        # Fallback: nếu parse lỗi, mặc định là RAG_REQUIRED (an toàn hơn)
        logger.warning(
            "Không parse được intent từ LLM response: '%s'. "
            "Fallback → RAG_REQUIRED. Lỗi: %s",
            response_text if 'response_text' in dir() else "N/A",
            str(e),
        )
        return QueryIntent(intent="RAG_REQUIRED")


# ══════════════════════════════════════════════════════════════════
# BƯỚC 1B: XỬ LÝ CHIT_CHAT — Trả lời giao tiếp bình thường
# ══════════════════════════════════════════════════════════════════

async def _handle_chit_chat(question: str) -> QueryResponse:
    """
    Xử lý câu hỏi giao tiếp bình thường (CHIT_CHAT).
    LLM trả lời tự nhiên, thân thiện, KHÔNG cần tra cứu tài liệu.

    Args:
        question: Câu hỏi giao tiếp của người dùng.

    Returns:
        QueryResponse với câu trả lời giao tiếp (không có citations).
    """
    llm = get_llm()

    chit_chat_prompt = (
        "Bạn là trợ lý giáo dục thân thiện tên là EduBot. "
        "Hãy trả lời câu hỏi giao tiếp sau một cách tự nhiên, "
        "vui vẻ và ngắn gọn bằng tiếng Việt.\n\n"
        f"Câu hỏi: {question}"
    )

    try:
        response = await llm.acomplete(chit_chat_prompt)
        answer = response.text.strip()
    except Exception as e:
        logger.error("Lỗi khi xử lý CHIT_CHAT: %s", str(e))
        raise

    logger.info("CHIT_CHAT → Trả lời giao tiếp (%d ký tự)", len(answer))

    return QueryResponse(
        answer=answer,
        citations=[],
        confidence="high",
        no_answer=False,
    )


# ══════════════════════════════════════════════════════════════════
# HÀM CHÍNH: PROCESS QUERY
# ══════════════════════════════════════════════════════════════════

async def process_query(request: QueryRequest) -> QueryResponse:
    """
    Xử lý truy vấn với Query Router:

    Bước 1: Router — Phân loại intent (CHIT_CHAT / RAG_REQUIRED).
    Bước 2: Nếu CHIT_CHAT → trả lời giao tiếp, kết thúc.
    Bước 3: Nếu RAG_REQUIRED → Embedding câu hỏi.
    Bước 4: Global Search trong Qdrant (không filter subject_id).
    Bước 5: Guardrail — kiểm tra similarity threshold.
    Bước 6: Xây dựng prompt + gọi Gemini LLM.
    Bước 7: Trích xuất citations (bao gồm chapter/section).
    Bước 8: Trả về QueryResponse.

    Args:
        request: QueryRequest chứa câu hỏi và conversation_id.

    Returns:
        QueryResponse với câu trả lời, trích dẫn, và đánh giá confidence.
    """
    settings = get_settings()

    logger.info(
        "Bắt đầu xử lý query: conversation_id=%s, question='%s'",
        request.conversation_id,
        request.question[:100],
    )

    # ── Bước 1: Query Router — Phân loại ý định ───────────────────
    intent = await _classify_intent(request.question)

    # ── Bước 2: Rẽ nhánh theo intent ──────────────────────────────
    if intent.intent == "CHIT_CHAT":
        logger.info("Intent = CHIT_CHAT → Chuyển sang xử lý giao tiếp")
        return await _handle_chit_chat(request.question)

    # ── Từ đây trở xuống: RAG_REQUIRED ────────────────────────────
    logger.info("Intent = RAG_REQUIRED → Bắt đầu luồng RAG")

    # ── Bước 3: Embedding câu hỏi ─────────────────────────────────
    embed_model = get_embedding_model()

    try:
        question_vector = await embed_model.aget_text_embedding(request.question)
    except Exception as e:
        logger.error("Lỗi khi tạo embedding cho câu hỏi: %s", str(e))
        raise

    # ── Bước 4: Global Search trong Qdrant ─────────────────────────
    # KHÔNG filter theo subject_id → tìm toàn bộ Vector DB
    client = await get_qdrant_client()

    try:
        search_results = client.query_points(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            query=question_vector,
            limit=settings.TOP_K,
            with_payload=True,
        )
    except Exception as e:
        logger.error("Lỗi khi truy vấn Qdrant: %s", str(e))
        raise

    results = search_results.points
    logger.info("Tìm thấy %d kết quả từ Qdrant", len(results))

    # ── Bước 5: Guardrail — kiểm tra similarity threshold ─────────
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
                   "trong tài liệu hiện có.",
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

    # ── Bước 6: Xây dựng prompt + gọi LLM ─────────────────────────
    context_text = _build_context(filtered_results)
    prompt = _build_rag_prompt(
        question=request.question,
        context=context_text,
    )

    llm = get_llm()

    try:
        llm_response = await llm.acomplete(prompt)
        answer_text = llm_response.text.strip()
    except Exception as e:
        logger.error("Lỗi khi gọi Gemini LLM: %s", str(e))
        raise

    logger.info("LLM đã sinh câu trả lời (%d ký tự)", len(answer_text))

    # ── Bước 7: Trích xuất citations ──────────────────────────────
    citations = _extract_citations(answer_text, filtered_results)

    # ── Bước 8: Đánh giá confidence + trả về ─────────────────────
    confidence = _evaluate_confidence(filtered_results)

    return QueryResponse(
        answer=answer_text,
        citations=citations,
        confidence=confidence,
        no_answer=False,
    )


# ══════════════════════════════════════════════════════════════════
# HÀM PHỤ TRỢ (PRIVATE HELPERS)
# ══════════════════════════════════════════════════════════════════

def _build_context(results: list[Any]) -> str:
    """
    Xây dựng đoạn context từ danh sách kết quả Qdrant.
    Đánh số mỗi chunk [1], [2], ... để LLM có thể trích dẫn.
    Bao gồm chapter/section nếu có trong metadata.

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
        chapter = payload.get("chapter", "")
        section = payload.get("section", "")

        # Xây dựng dòng metadata với heading hierarchy
        meta_parts = [f"Tài liệu: {doc_id}", f"Trang: {page}"]
        if chapter:
            meta_parts.append(f"Chương: {chapter}")
        if section:
            meta_parts.append(f"Mục: {section}")

        meta_line = ", ".join(meta_parts)
        context_parts.append(f"[{idx}] ({meta_line})\n{text}")

    return "\n\n---\n\n".join(context_parts)


def _build_rag_prompt(question: str, context: str) -> str:
    """
    Xây dựng prompt RAG (Single-turn, không có history).

    RÀNG BUỘC:
    - LLM chỉ được dùng context đã cung cấp.
    - Bắt buộc trích dẫn nguồn [1], [2], ... sau mỗi ý.
    - Không được tự bịa thông tin.

    Args:
        question: Câu hỏi của người dùng.
        context: Đoạn context từ Qdrant chunks.

    Returns:
        str: Prompt đầy đủ gửi tới LLM.
    """
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

    full_prompt = (
        f"{system_instruction}\n"
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
    Tìm các pattern [1], [2], ... trong answer và khớp với metadata.
    Bao gồm chapter và section từ heading hierarchy.

    Args:
        answer: Câu trả lời từ LLM (chứa [1], [2], ...).
        results: Danh sách ScoredPoint từ Qdrant (đã filter).

    Returns:
        List[Citation]: Danh sách trích dẫn nguồn.
    """
    cited_indices = set()
    matches = re.findall(r"\[(\d+)\]", answer)

    for match in matches:
        idx = int(match)
        if 1 <= idx <= len(results):
            cited_indices.add(idx)

    citations = []
    for idx in sorted(cited_indices):
        result = results[idx - 1]  # Convert 1-indexed → 0-indexed
        payload = result.payload

        citations.append(
            Citation(
                doc_id=payload.get("doc_id", "unknown"),
                page_number=payload.get("page_number", 0),
                snippet=payload.get("text", "")[:200],
                chapter=payload.get("chapter"),
                section=payload.get("section"),
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
