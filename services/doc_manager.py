"""
services/doc_manager.py
-----------------------
Quản lý trạng thái tài liệu trong Qdrant.

Theo sơ đồ 6 (State Machines) + sơ đồ 8 (Hide/Show/Delete Flow):
- Hide:   Set is_hidden=true trên tất cả points có doc_id → callback SUCCEEDED
- Unhide: Set is_hidden=false trên tất cả points có doc_id → callback SUCCEEDED
- Delete: Xóa tất cả points có doc_id khỏi Qdrant → callback SUCCEEDED

Tất cả đều chạy background + callback (async pattern).
"""

import logging

# pyrefly: ignore [missing-import]
from qdrant_client import models

from core.config import get_settings
from core.database import get_qdrant_client
from services.callback import (
    send_progress,
    send_succeeded_visibility,
    send_succeeded_delete,
    send_failed,
)

logger = logging.getLogger(__name__)


async def hide_document_background(
    doc_id: str,
    job_id: str,
    callback_url: str,
) -> None:
    """
    Ẩn tài liệu khỏi RAG: set is_hidden=true.
    Khi search, filter is_hidden != true sẽ bỏ qua các chunks này.
    """
    try:
        await send_progress(callback_url, job_id, "hiding")

        settings = get_settings()
        client = await get_qdrant_client()

        # Set payload is_hidden=true cho tất cả points có doc_id
        result = client.set_payload(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            payload={"is_hidden": True},
            points=models.Filter(
                must=[
                    models.FieldCondition(
                        key="doc_id",
                        match=models.MatchValue(value=doc_id),
                    )
                ]
            ),
        )

        # Đếm số points đã update
        count = _count_points_by_doc_id(client, settings.QDRANT_COLLECTION_NAME, doc_id)

        logger.info("[DOC_MANAGER] Đã ẩn doc_id=%s (%d chunks)", doc_id, count)

        await send_succeeded_visibility(callback_url, job_id, updated_count=count)

    except Exception as e:
        logger.exception("[DOC_MANAGER] Lỗi khi ẩn doc_id=%s", doc_id)
        await send_failed(callback_url, job_id, "HIDE_ERROR", str(e))


async def unhide_document_background(
    doc_id: str,
    job_id: str,
    callback_url: str,
) -> None:
    """
    Hiện lại tài liệu trong RAG: set is_hidden=false.
    """
    try:
        await send_progress(callback_url, job_id, "unhiding")

        settings = get_settings()
        client = await get_qdrant_client()

        client.set_payload(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            payload={"is_hidden": False},
            points=models.Filter(
                must=[
                    models.FieldCondition(
                        key="doc_id",
                        match=models.MatchValue(value=doc_id),
                    )
                ]
            ),
        )

        count = _count_points_by_doc_id(client, settings.QDRANT_COLLECTION_NAME, doc_id)

        logger.info("[DOC_MANAGER] Đã hiện lại doc_id=%s (%d chunks)", doc_id, count)

        await send_succeeded_visibility(callback_url, job_id, updated_count=count)

    except Exception as e:
        logger.exception("[DOC_MANAGER] Lỗi khi hiện lại doc_id=%s", doc_id)
        await send_failed(callback_url, job_id, "UNHIDE_ERROR", str(e))


async def delete_document_background(
    doc_id: str,
    job_id: str,
    callback_url: str,
) -> None:
    """
    Xóa toàn bộ vectors của tài liệu khỏi Qdrant.
    Theo sơ đồ: xóa vẫn giữ file gốc và lịch sử MySQL (Node.js xử lý).
    """
    try:
        await send_progress(callback_url, job_id, "deleting")

        settings = get_settings()
        client = await get_qdrant_client()

        # Đếm trước khi xóa
        count = _count_points_by_doc_id(client, settings.QDRANT_COLLECTION_NAME, doc_id)

        # Xóa tất cả points có doc_id
        client.delete(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="doc_id",
                            match=models.MatchValue(value=doc_id),
                        )
                    ]
                )
            ),
        )

        logger.info("[DOC_MANAGER] Đã xóa doc_id=%s (%d vectors)", doc_id, count)

        await send_succeeded_delete(callback_url, job_id, deleted_count=count)

    except Exception as e:
        logger.exception("[DOC_MANAGER] Lỗi khi xóa doc_id=%s", doc_id)
        await send_failed(callback_url, job_id, "DELETE_ERROR", str(e))


def _count_points_by_doc_id(client, collection_name: str, doc_id: str) -> int:
    """Đếm số points trong collection có doc_id tương ứng."""
    try:
        result = client.count(
            collection_name=collection_name,
            count_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="doc_id",
                        match=models.MatchValue(value=doc_id),
                    )
                ]
            ),
        )
        return result.count
    except Exception:
        return 0
