"""
core/llm_setup.py
-----------------
Khởi tạo Embedding Model và LLM (Google Gemini) thông qua LlamaIndex.
Sử dụng pattern Singleton để tránh khởi tạo lại nhiều lần.

Phiên bản v2:
- Thêm get_router_llm() với temperature=0 cho Query Router (Structured Output).
"""

import logging
from functools import lru_cache

from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding
from core.config import get_settings

logger = logging.getLogger(__name__)


@lru_cache()
def get_embedding_model() -> GeminiEmbedding:
    """
    Khởi tạo và cache Gemini Embedding model.
    Model này biến văn bản thành vector số để lưu/truy vấn trong Qdrant.

    Returns:
        GeminiEmbedding: Instance embedding model đã sẵn sàng sử dụng.
    """
    settings = get_settings()

    logger.info(
        "Đang khởi tạo Gemini Embedding model: %s",
        settings.GEMINI_EMBEDDING_MODEL,
    )

    embed_model = GeminiEmbedding(
        model_name=settings.GEMINI_EMBEDDING_MODEL,
        api_key=settings.GOOGLE_API_KEY,
    )

    logger.info("Gemini Embedding model sẵn sàng ✓")
    return embed_model


@lru_cache()
def get_llm() -> Gemini:
    """
    Khởi tạo và cache Gemini LLM dùng cho việc sinh câu trả lời RAG.
    Cấu hình temperature thấp để đảm bảo câu trả lời bám sát context.

    Returns:
        Gemini: Instance LLM đã sẵn sàng sử dụng.
    """
    settings = get_settings()

    logger.info(
        "Đang khởi tạo Gemini LLM: %s",
        settings.GEMINI_LLM_MODEL,
    )

    llm = Gemini(
        model=settings.GEMINI_LLM_MODEL,
        api_key=settings.GOOGLE_API_KEY,
        temperature=0.1,  # Giữ thấp để trả lời chính xác, ít sáng tạo
    )

    logger.info("Gemini LLM sẵn sàng ✓")
    return llm


@lru_cache()
def get_router_llm() -> Gemini:
    """
    Khởi tạo và cache Gemini LLM riêng cho Query Router.
    Temperature=0 để phân loại intent một cách deterministic.
    Model này CHỈ dùng để phân loại câu hỏi (CHIT_CHAT / RAG_REQUIRED),
    KHÔNG dùng để sinh câu trả lời.

    Returns:
        Gemini: Instance LLM cho router với temperature=0.
    """
    settings = get_settings()

    logger.info(
        "Đang khởi tạo Router LLM (temperature=0): %s",
        settings.GEMINI_LLM_MODEL,
    )

    router_llm = Gemini(
        model=settings.GEMINI_LLM_MODEL,
        api_key=settings.GOOGLE_API_KEY,
        temperature=0.0,  # Deterministic — luôn cho kết quả nhất quán
    )

    logger.info("Router LLM sẵn sàng ✓")
    return router_llm
