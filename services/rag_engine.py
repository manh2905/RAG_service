"""
services/rag_engine.py
----------------------
Luồng RAG chính với Query Router:
  1. Router: phân loại câu hỏi → CHIT_CHAT hoặc RAG_REQUIRED.
  2. Nếu CHIT_CHAT → LLM trả lời giao tiếp bình thường (không RAG).
  3. Nếu RAG_REQUIRED → Semantic Search (READY+VISIBLE) → Prompt → LLM → Citations.

Phiên bản v3:
- Nhận history từ Node.js → đưa vào prompt (multi-turn context).
- Filter is_hidden != true khi search (chỉ search READY + VISIBLE).
- Usage tracking: prompt_tokens, completion_tokens, total_tokens, model.
"""

import json
import logging
import re
from typing import Any

# pyrefly: ignore [missing-import]
from qdrant_client import models

from core.config import get_settings
from core.database import get_qdrant_client
from core.llm_setup import get_embedding_model, get_llm, get_router_llm
from models.schemas import (
    Citation,
    ChatMessage,
    QueryIntent,
    QueryRequest,
    QueryResponse,
    UsageInfo,
)

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════
# BƯỚC 1: QUERY ROUTER — Phân loại ý định câu hỏi
# ══════════════════════════════════════════════════════════════════

async def _classify_intent(question: str) -> QueryIntent:
    """
    Sử dụng LLM (temperature=0) để phân loại ý định câu hỏi.
    LLM được ép trả về Structured Output theo schema QueryIntent.
    """
    router_llm = get_router_llm()

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

        if "```" in response_text:
            json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response_text, re.DOTALL)
            if json_match:
                response_text = json_match.group(1)

        parsed = json.loads(response_text)
        intent = QueryIntent(**parsed)

        logger.info("Query Router phân loại: %s", intent.intent)
        return intent

    except (json.JSONDecodeError, Exception) as e:
        logger.warning(
            "Không parse được intent từ LLM response. "
            "Fallback → RAG_REQUIRED. Lỗi: %s",
            str(e),
        )
        return QueryIntent(intent="RAG_REQUIRED")


# ══════════════════════════════════════════════════════════════════
# BƯỚC 1B: XỬ LÝ CHIT_CHAT
# ══════════════════════════════════════════════════════════════════

async def _handle_chit_chat(
    question: str,
    history: list[ChatMessage],
) -> QueryResponse:
    """Xử lý câu hỏi giao tiếp bình thường (CHIT_CHAT)."""
    settings = get_settings()
    llm = get_llm()

    # Xây dựng context từ history
    history_text = _format_history(history)

    chit_chat_prompt = (
        "Bạn là trợ lý giáo dục thân thiện tên là EduBot. "
        "Hãy trả lời câu hỏi giao tiếp sau một cách tự nhiên, "
        "vui vẻ và ngắn gọn bằng tiếng Việt.\n\n"
    )

    if history_text:
        chit_chat_prompt += f"Lịch sử hội thoại:\n{history_text}\n\n"

    chit_chat_prompt += f"Câu hỏi: {question}"

    try:
        response = await llm.acomplete(chit_chat_prompt)
        answer = response.text.strip()

        # Usage tracking
        usage = _extract_usage(response, settings.GEMINI_LLM_MODEL)

    except Exception as e:
        logger.error("Lỗi khi xử lý CHIT_CHAT: %s", str(e))
        raise

    return QueryResponse(
        answer=answer,
        citations=[],
        confidence="high",
        no_answer=False,
        usage=usage,
    )


# ══════════════════════════════════════════════════════════════════
# HÀM CHÍNH: PROCESS QUERY
# ══════════════════════════════════════════════════════════════════

async def process_query(request: QueryRequest) -> QueryResponse:
    """
    Xử lý truy vấn với Query Router:

    Bước 1: Router — Phân loại intent.
    Bước 2: Nếu CHIT_CHAT → trả lời giao tiếp.
    Bước 3: Nếu RAG_REQUIRED → Embedding → Search (filter is_hidden) → LLM → Citations.
    """
    settings = get_settings()
    history = request.history or []

    logger.info(
        "Bắt đầu xử lý query: conversation_id=%s, question='%s', history_len=%d",
        request.conversation_id,
        request.question[:100],
        len(history),
    )

    # ── Bước 1: Query Router ──────────────────────────────────────
    intent = await _classify_intent(request.question)

    # ── Bước 2: CHIT_CHAT ────────────────────────────────────────
    if intent.intent == "CHIT_CHAT":
        logger.info("Intent = CHIT_CHAT → Chuyển sang xử lý giao tiếp")
        return await _handle_chit_chat(request.question, history)

    # ── Từ đây: RAG_REQUIRED ─────────────────────────────────────
    logger.info("Intent = RAG_REQUIRED → Bắt đầu luồng RAG")

    # ── Bước 3: Embedding câu hỏi ────────────────────────────────
    embed_model = get_embedding_model()

    try:
        question_vector = await embed_model.aget_text_embedding(request.question)
    except Exception as e:
        logger.error("Lỗi khi tạo embedding cho câu hỏi: %s", str(e))
        raise

    # ── Bước 4: Search Qdrant (filter is_hidden != true) ──────────
    client = await get_qdrant_client()

    try:
        search_results = client.query_points(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            query=question_vector,
            query_filter=models.Filter(
                must_not=[
                    models.FieldCondition(
                        key="is_hidden",
                        match=models.MatchValue(value=True),
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
    logger.info("Tìm thấy %d kết quả (đã filter is_hidden)", len(results))

    # ── Bước 5: Guardrail — similarity threshold ──────────────────
    filtered_results = [
        r for r in results
        if r.score >= settings.SIMILARITY_THRESHOLD
    ]

    if not filtered_results:
        logger.warning(
            "Không có chunk nào vượt ngưỡng similarity %.2f",
            settings.SIMILARITY_THRESHOLD,
        )
        return QueryResponse(
            answer="Không đủ dữ liệu/Không tìm thấy thông tin liên quan "
                   "trong tài liệu hiện có.",
            citations=[],
            confidence="low",
            no_answer=True,
            usage=None,
        )

    # ── Bước 6: Build prompt + gọi LLM ───────────────────────────
    context_text = _build_context(filtered_results)
    prompt = _build_rag_prompt(
        question=request.question,
        context=context_text,
        history=history,
    )

    llm = get_llm()

    try:
        llm_response = await llm.acomplete(prompt)
        answer_text = llm_response.text.strip()
        usage = _extract_usage(llm_response, settings.GEMINI_LLM_MODEL)
    except Exception as e:
        logger.error("Lỗi khi gọi Gemini LLM: %s", str(e))
        raise

    # ── Bước 7: Trích xuất và chuẩn hóa citations ─────────────────
    citation_result = _extract_citations(
        answer=answer_text,
        results=filtered_results,
        question=request.question,
    )

    # Một câu trả lời RAG không có marker, hoặc có marker không ánh xạ
    # được về retrieval result, không được phép tạo citation giả.
    if citation_result is None:
        logger.warning(
            "Câu trả lời không ánh xạ an toàn được về retrieval results; "
            "chuyển sang no_answer"
        )
        return QueryResponse(
            answer="Không đủ dữ liệu để tạo câu trả lời có nguồn trích dẫn đáng tin cậy.",
            citations=[],
            confidence="low",
            no_answer=True,
            usage=usage,
        )

    answer_text, citations = citation_result
    confidence = _evaluate_confidence(filtered_results)

    return QueryResponse(
        answer=answer_text,
        citations=citations,
        confidence=confidence,
        no_answer=False,
        usage=usage,
    )


# ══════════════════════════════════════════════════════════════════
# HÀM PHỤ TRỢ
# ══════════════════════════════════════════════════════════════════

def _format_history(history: list[ChatMessage]) -> str:
    """Format history thành text cho prompt."""
    if not history:
        return ""

    parts = []
    for msg in history[-6:]:  # Chỉ lấy 6 tin nhắn gần nhất
        role_label = "Người dùng" if msg.role == "user" else "Trợ lý"
        parts.append(f"{role_label}: {msg.content}")

    return "\n".join(parts)


def _build_context(results: list[Any]) -> str:
    """Xây dựng đoạn context từ kết quả Qdrant."""
    context_parts = []

    for idx, result in enumerate(results, start=1):
        payload = result.payload
        text = payload.get("text", "")
        doc_id = payload.get("doc_id", "N/A")
        page = payload.get("page_number", 0)
        chapter = payload.get("chapter", "")
        section = payload.get("section", "")

        meta_parts = [f"Tài liệu: {doc_id}", f"Trang: {page}"]
        if chapter:
            meta_parts.append(f"Chương: {chapter}")
        if section:
            meta_parts.append(f"Mục: {section}")

        meta_line = ", ".join(meta_parts)
        context_parts.append(f"[{idx}] ({meta_line})\n{text}")

    return "\n\n---\n\n".join(context_parts)


def _build_rag_prompt(
    question: str,
    context: str,
    history: list[ChatMessage],
) -> str:
    """Xây dựng prompt RAG với history support."""
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

    # Thêm history nếu có
    history_text = _format_history(history)
    history_section = ""
    if history_text:
        history_section = (
            f"--- LỊCH SỬ HỘI THOẠI ---\n"
            f"{history_text}\n"
            f"--- HẾT LỊCH SỬ ---\n\n"
        )

    full_prompt = (
        f"{system_instruction}\n"
        f"{history_section}"
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
    question: str = "",
) -> tuple[str, list[Citation]] | None:
    """
    Ánh xạ marker của LLM về retrieval result và đánh lại marker liên tục.

    Ví dụ LLM dùng result [1] và [3] thì response cuối sẽ dùng [1] và [2],
    đúng với thứ tự của mảng citations mà Node.js lưu. Nếu có bất kỳ marker
    nào không hợp lệ, hoặc không có marker, kết quả bị từ chối.
    """
    raw_matches = re.findall(r"\[(\d+)\]", answer)
    if not raw_matches:
        return None

    referenced_indices: list[int] = []
    seen: set[int] = set()
    for match in raw_matches:
        idx = int(match)
        if not 1 <= idx <= len(results):
            return None
        if idx not in seen:
            seen.add(idx)
            referenced_indices.append(idx)

    marker_map = {
        original_idx: citation_idx
        for citation_idx, original_idx in enumerate(referenced_indices, start=1)
    }

    def replace_marker(match: re.Match[str]) -> str:
        return f"[{marker_map[int(match.group(1))]}]"

    normalized_answer = re.sub(r"\[(\d+)\]", replace_marker, answer)

    citations = []
    for idx in referenced_indices:
        result = results[idx - 1]
        payload = result.payload
        chunk_text = payload.get("text", "")

        citations.append(
            Citation(
                vector_node_id=str(result.id),
                doc_id=payload.get("doc_id", "unknown"),
                page_number=payload.get("page_number"),
                snippet=_select_relevant_snippet(
                    text=chunk_text,
                    question=question,
                    answer=answer,
                ),
                chapter=payload.get("chapter"),
                section=payload.get("section"),
            )
        )

    return normalized_answer, citations


def _select_relevant_snippet(
    text: str,
    question: str,
    answer: str,
    max_chars: int = 500,
) -> str:
    """
    Chọn đoạn nguồn gần nhất với câu hỏi/câu trả lời thay vì luôn lấy đầu chunk.

    Đây là trích xuất trực tiếp từ chunk đã retrieval, không sinh lại nội dung.
    """
    text = text.strip()
    if not text:
        return ""
    if len(text) <= 220:
        return text

    segments = [
        segment.strip()
        for segment in re.split(r"(?<=[.!?])\s+|\n+", text)
        if segment.strip()
    ]
    if not segments:
        return text[:max_chars].strip()

    stop_words = {
        "các", "cho", "của", "được", "hay", "khi", "là", "một", "này",
        "những", "phần", "qua", "sau", "theo", "thì", "trong", "trên",
        "vào", "và", "với", "tài", "liệu",
    }

    def keywords(value: str) -> set[str]:
        return {
            word
            for word in re.findall(r"\w+", value.lower(), flags=re.UNICODE)
            if len(word) > 2 and word not in stop_words and not word.isdigit()
        }

    question_words = keywords(question)
    answer_words = keywords(re.sub(r"\[\d+\]", "", answer))

    def score(segment: str) -> tuple[int, int]:
        segment_words = keywords(segment)
        relevance = (
            3 * len(segment_words & question_words)
            + len(segment_words & answer_words)
        )
        return relevance, -len(segment)

    best_idx = max(range(len(segments)), key=lambda idx: score(segments[idx]))
    selected = segments[best_idx]

    if len(selected) > max_chars:
        relevant_words = question_words | answer_words
        match_positions = [
            match.start()
            for match in re.finditer(r"\w+", selected.lower(), flags=re.UNICODE)
            if match.group(0) in relevant_words
        ]
        center = match_positions[0] if match_positions else 0
        start = max(0, center - max_chars // 3)
        end = min(len(selected), start + max_chars)
        start = max(0, end - max_chars)
        selected = selected[start:end].strip()

    # Thêm ngữ cảnh liền kề nhưng luôn giữ đoạn liên quan ở trong giới hạn.
    left = best_idx - 1
    right = best_idx + 1
    while len(selected) < 220 and (left >= 0 or right < len(segments)):
        candidates = []
        if left >= 0:
            candidates.append(("left", segments[left]))
        if right < len(segments):
            candidates.append(("right", segments[right]))

        side, candidate = max(candidates, key=lambda item: score(item[1]))
        combined = (
            f"{candidate}\n{selected}"
            if side == "left"
            else f"{selected}\n{candidate}"
        )
        if len(combined) > max_chars:
            break
        selected = combined
        if side == "left":
            left -= 1
        else:
            right += 1

    return selected[:max_chars].strip()


def _evaluate_confidence(results: list[Any]) -> str:
    """Đánh giá mức độ tin cậy dựa trên similarity score."""
    if not results:
        return "low"

    avg_score = sum(r.score for r in results) / len(results)

    if avg_score >= 0.7:
        return "high"
    elif avg_score >= 0.5:
        return "medium"
    else:
        return "low"


def _extract_usage(llm_response: Any, model_name: str) -> UsageInfo:
    """
    Trích xuất usage info từ LLM response.
    Gemini qua LlamaIndex có thể có hoặc không có token counts.
    """
    try:
        raw = getattr(llm_response, "raw", None)
        if raw and hasattr(raw, "usage_metadata"):
            metadata = raw.usage_metadata
            return UsageInfo(
                prompt_tokens=getattr(metadata, "prompt_token_count", 0),
                completion_tokens=getattr(metadata, "candidates_token_count", 0),
                total_tokens=getattr(metadata, "total_token_count", 0),
                model=model_name,
            )
    except Exception:
        pass

    # Fallback: không có usage data
    return UsageInfo(
        prompt_tokens=0,
        completion_tokens=0,
        total_tokens=0,
        model=model_name,
    )
