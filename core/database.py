"""
core/database.py
----------------
Khởi tạo và quản lý kết nối Singleton tới Qdrant Vector Database.
Đảm bảo chỉ có một instance QdrantClient duy nhất trong toàn bộ app,
và tự động tạo collection nếu chưa tồn tại.
"""

import logging
from qdrant_client import QdrantClient, models
from core.config import get_settings

logger = logging.getLogger(__name__)

# ── Biến toàn cục lưu trữ singleton client ──────────────────────
_qdrant_client: QdrantClient | None = None


async def get_qdrant_client() -> QdrantClient:
    """
    Lấy hoặc tạo QdrantClient singleton.
    Nếu client chưa tồn tại, khởi tạo kết nối mới tới Qdrant server
    và tạo collection nếu cần.

    Returns:
        QdrantClient: Instance kết nối tới Qdrant.
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
    Nếu chưa → tạo mới với cấu hình Cosine similarity.

    Args:
        client: QdrantClient instance.
        collection_name: Tên collection cần kiểm tra/tạo.
        vector_size: Số chiều của vector embedding.
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

            # Tạo index cho trường subject_id để tăng tốc filter
            client.create_payload_index(
                collection_name=collection_name,
                field_name="subject_id",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )

            logger.info("Đã tạo collection '%s' với %d chiều vector ✓", collection_name, vector_size)
        else:
            logger.info("Collection '%s' đã tồn tại ✓", collection_name)

    except Exception as e:
        logger.error("Lỗi khi kiểm tra/tạo collection: %s", str(e))
        raise


async def close_qdrant_client() -> None:
    """
    Đóng kết nối tới Qdrant khi shutdown app.
    Giải phóng tài nguyên một cách an toàn.
    """
    global _qdrant_client

    if _qdrant_client is not None:
        try:
            _qdrant_client.close()
            logger.info("Đã đóng kết nối Qdrant ✓")
        except Exception as e:
            logger.warning("Lỗi khi đóng kết nối Qdrant: %s", str(e))
        finally:
            _qdrant_client = None
