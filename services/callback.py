"""
services/callback.py
--------------------
Service gọi callback tới Node.js Internal API sau khi xử lý xong.

Chức năng:
- Gửi HTTP POST tới callback_url kèm Authorization header.
- Retry logic với exponential backoff.
- Hỗ trợ các eventType: PROGRESS, SUCCEEDED, FAILED, CANCELLED.

Theo sơ đồ 5 (Callback Flow):
- Python gửi {jobId, attemptCount, eventType} → Node.js
- Node.js xác thực internal token → trả 200 ACK
"""

import logging
import asyncio
from typing import Optional, Any

import httpx

from core.config import get_settings
from models.schemas import CallbackPayload

logger = logging.getLogger(__name__)


async def send_callback(
    callback_url: str,
    payload: CallbackPayload,
) -> bool:
    """
    Gửi callback tới Node.js Internal API.

    Args:
        callback_url: URL endpoint của Node.js để nhận callback.
        payload: CallbackPayload chứa job_id, event_type, data.

    Returns:
        True nếu gửi thành công (nhận 200 ACK), False nếu thất bại.
    """
    settings = get_settings()
    max_retries = settings.CALLBACK_MAX_RETRIES
    timeout = settings.CALLBACK_TIMEOUT

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {settings.INTERNAL_SECRET}",
    }

    for retry_idx in range(1, max_retries + 1):
        try:
            logger.info(
                "[CALLBACK] Gửi callback (retry %d/%d): job_id=%s, event=%s, attempt_count=%d → %s",
                retry_idx,
                max_retries,
                payload.job_id,
                payload.event_type,
                payload.attempt_count,
                callback_url,
            )

            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(
                    callback_url,
                    json=payload.model_dump(exclude_none=True),
                    headers=headers,
                )

            if response.status_code == 200:
                logger.info(
                    "[CALLBACK] Thành công: job_id=%s, event=%s (retry %d)",
                    payload.job_id,
                    payload.event_type,
                    retry_idx,
                )
                return True

            logger.warning(
                "[CALLBACK] Node.js trả status %d: job_id=%s (retry %d)",
                response.status_code,
                payload.job_id,
                retry_idx,
            )

        except httpx.TimeoutException:
            logger.warning(
                "[CALLBACK] Timeout sau %ds: job_id=%s (retry %d/%d)",
                timeout,
                payload.job_id,
                retry_idx,
                max_retries,
            )

        except httpx.ConnectError:
            logger.warning(
                "[CALLBACK] Không kết nối được tới %s (retry %d/%d)",
                callback_url,
                retry_idx,
                max_retries,
            )

        except Exception as e:
            logger.error(
                "[CALLBACK] Lỗi không xác định: %s (retry %d/%d)",
                str(e),
                retry_idx,
                max_retries,
            )

        # Exponential backoff: 1s, 2s, 4s
        if retry_idx < max_retries:
            wait_time = 2 ** (retry_idx - 1)
            logger.info("[CALLBACK] Chờ %ds trước khi retry...", wait_time)
            await asyncio.sleep(wait_time)

    logger.error(
        "[CALLBACK] THẤT BẠI sau %d lần thử: job_id=%s, event=%s",
        max_retries,
        payload.job_id,
        payload.event_type,
    )
    return False


# ══════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS — Tạo callback nhanh
# ══════════════════════════════════════════════════════════════════

async def send_progress(callback_url: str, job_id: str, attempt_count: int, stage: str) -> bool:
    """Gửi callback PROGRESS với stage hiện tại."""
    payload = CallbackPayload(
        job_id=job_id,
        attempt_count=attempt_count,
        event_type="PROGRESS",
        stage=stage,
    )
    return await send_callback(callback_url, payload)


async def send_succeeded_ingest(
    callback_url: str,
    job_id: str,
    attempt_count: int,
    chunks_count: int,
    chunk_manifest: list,
) -> bool:
    """Gửi callback SUCCEEDED cho ingest (kèm chunk manifest)."""
    payload = CallbackPayload(
        job_id=job_id,
        attempt_count=attempt_count,
        event_type="SUCCEEDED",
        chunks_count=chunks_count,
        chunk_manifest=chunk_manifest,
    )
    return await send_callback(callback_url, payload)


async def send_succeeded_visibility(
    callback_url: str,
    job_id: str,
    attempt_count: int,
    updated_count: int,
) -> bool:
    """Gửi callback SUCCEEDED cho hide/unhide."""
    payload = CallbackPayload(
        job_id=job_id,
        attempt_count=attempt_count,
        event_type="SUCCEEDED",
        updated_count=updated_count,
    )
    return await send_callback(callback_url, payload)


async def send_succeeded_delete(
    callback_url: str,
    job_id: str,
    attempt_count: int,
    deleted_count: int,
) -> bool:
    """Gửi callback SUCCEEDED cho delete."""
    payload = CallbackPayload(
        job_id=job_id,
        attempt_count=attempt_count,
        event_type="SUCCEEDED",
        deleted_count=deleted_count,
    )
    return await send_callback(callback_url, payload)


async def send_failed(
    callback_url: str,
    job_id: str,
    attempt_count: int,
    error_code: str,
    error_message: str,
    stage: Optional[str] = None,
) -> bool:
    """Gửi callback FAILED với thông tin lỗi và giai đoạn."""
    payload = CallbackPayload(
        job_id=job_id,
        attempt_count=attempt_count,
        event_type="FAILED",
        stage=stage,
        error={"code": error_code, "message": error_message},
    )
    return await send_callback(callback_url, payload)
