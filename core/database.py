"""
core/database.py
----------------
Khởi tạo và quản lý kết nối Singleton tới Qdrant Vector Database.
Tự động tạo collection và payload indexes nếu chưa tồn tại.

Phiên bản v3:
- Thêm payload index cho doc_id (tăng tốc filter/delete).
- Thêm payload index cho is_hidden (tăng tốc filter khi search).
"""

import logging
# pyrefly: ignore [missing-import]
from qdrant_client import QdrantClient, models
from core.config import get_settings

logger = logging.getLogger(__name__)

# ── Biến toàn cục lưu trữ singleton client ──────────────────────
_qdrant_client: QdrantClient | None = None


async def get_qdrant_client() -> QdrantClient:
    """
    Lấy hoặc tạo QdrantClient singleton.
    Nếu client chưa tồn tại, khởi tạo kết nối mới tới Qdrant server
    và tạo collection + indexes nếu cần.
    """
    global _qdrant_client

    if _qdrant_client is not None:
        return _qdrant_client

    settings = get_settings()

    try:
        logger.info("Đang khởi tạo kết nối tới Qdrant tại: %s", settings.QDRANT_URL)

        _qdrant_client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY,
            timeout=30,
        )

        # Kiểm tra và tạo collection nếu chưa có
        await _ensure_collection_exists(
            client=_qdrant_client,
            collection_name=settings.QDRANT_COLLECTION_NAME,
            vector_size=settings.EMBEDDING_DIMENSION,
        )

        logger.info("Kết nối Qdrant thành công ✓")
        return _qdrant_client

    except Exception as e:
        logger.error("Lỗi khi kết nối tới Qdrant: %s", str(e))
        _qdrant_client = None
        raise


async def _ensure_collection_exists(
    client: QdrantClient,
    collection_name: str,
    vector_size: int,
) -> None:
    """
    Kiểm tra collection đã tồn tại chưa.
    Nếu chưa → tạo mới với Cosine similarity + payload indexes.
    """
    try:
        collections = client.get_collections().collections
        existing_names = [col.name for col in collections]

        if collection_name not in existing_names:
            logger.info("Collection '%s' chưa tồn tại — đang tạo mới...", collection_name)

            client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE,
                ),
            )

            # Tạo payload indexes để tăng tốc filter/delete
            _create_payload_indexes(client, collection_name)

            logger.info("Đã tạo collection '%s' với %d chiều vector ✓", collection_name, vector_size)
        else:
            logger.info("Collection '%s' đã tồn tại ✓", collection_name)

    except Exception as e:
        logger.error("Lỗi khi kiểm tra/tạo collection: %s", str(e))
        raise


def _create_payload_indexes(client: QdrantClient, collection_name: str) -> None:
    """Tạo payload indexes cho các field thường dùng để filter."""
    indexes = [
        ("doc_id", models.PayloadSchemaType.KEYWORD),
        ("subject_id", models.PayloadSchemaType.KEYWORD),
        ("is_hidden", models.PayloadSchemaType.BOOL),
    ]

    for field_name, schema_type in indexes:
        try:
            client.create_payload_index(
                collection_name=collection_name,
                field_name=field_name,
                field_schema=schema_type,
            )
            logger.info("Đã tạo index cho field '%s' ✓", field_name)
        except Exception as e:
            logger.warning("Không tạo được index cho '%s': %s", field_name, str(e))


async def close_qdrant_client() -> None:
    """Đóng kết nối tới Qdrant khi shutdown app."""
    global _qdrant_client

    if _qdrant_client is not None:
        try:
            _qdrant_client.close()
            logger.info("Đã đóng kết nối Qdrant ✓")
        except Exception as e:
            logger.warning("Lỗi khi đóng kết nối Qdrant: %s", str(e))
        finally:
            _qdrant_client = None
