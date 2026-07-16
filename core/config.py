"""
core/config.py
--------------
Quản lý biến môi trường cho toàn bộ ứng dụng.
Sử dụng pydantic-settings để tự động load từ file .env.

Phiên bản v3:
- Thêm LLAMA_CLOUD_API_KEY cho LlamaParse.
- Thêm INTERNAL_SECRET cho callback auth.
- Thêm CALLBACK_TIMEOUT, CALLBACK_MAX_RETRIES cho callback mechanism.
"""

from functools import lru_cache
# pyrefly: ignore [missing-import]
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Cấu hình ứng dụng — tất cả giá trị được đọc từ biến môi trường
    hoặc file .env ở thư mục gốc dự án.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # === Google Gemini API ===
    GOOGLE_API_KEY: str = ""

    # === LlamaParse (LlamaIndex Cloud) ===
    LLAMA_CLOUD_API_KEY: str = ""

    # === Tên model Gemini ===
    GEMINI_LLM_MODEL: str = "models/gemini-2.0-flash"
    GEMINI_EMBEDDING_MODEL: str = "models/text-embedding-004"

    # === Qdrant Vector Database ===
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_API_KEY: str | None = None
    QDRANT_COLLECTION_NAME: str = "education_docs"

    # === RAG Parameters ===
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50
    TOP_K: int = 5
    SIMILARITY_THRESHOLD: float = 0.35

    # === CORS — cho phép Node.js backend gọi tới ===
    CORS_ORIGINS: list[str] = ["http://localhost:3000"]

    # === Embedding Dimension (text-embedding-004 trả về 768 chiều) ===
    EMBEDDING_DIMENSION: int = 768

    # === Callback — gọi ngược Node.js sau khi xử lý xong ===
    INTERNAL_SECRET: str = "change-me-in-production"
    CALLBACK_TIMEOUT: int = 30      # seconds
    CALLBACK_MAX_RETRIES: int = 3


@lru_cache()
def get_settings() -> Settings:
    """
    Singleton pattern: chỉ khởi tạo Settings một lần duy nhất
    rồi cache lại cho các lần gọi sau.
    """
    return Settings()
