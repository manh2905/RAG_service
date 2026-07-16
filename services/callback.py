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

    for attempt in range(1, max_retries + 1):
        # Cập nhật attempt_count trong payload
        payload.attempt_count = attempt

        try:
            logger.info(
                "[CALLBACK] Gửi callback (attempt %d/%d): job_id=%s, event=%s → %s",
                attempt,
                max_retries,
                payload.job_id,
                payload.event_type,
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
                    "[CALLBACK] Thành công: job_id=%s, event=%s (attempt %d)",
                    payload.job_id,
                    payload.event_type,
                    attempt,
                )
                return True

            logger.warning(
                "[CALLBACK] Node.js trả status %d: job_id=%s (attempt %d)",
                response.status_code,
                payload.job_id,
                attempt,
            )

        except httpx.TimeoutException:
            logger.warning(
                "[CALLBACK] Timeout sau %ds: job_id=%s (attempt %d/%d)",
                timeout,
                payload.job_id,
                attempt,
                max_retries,
            )

        except httpx.ConnectError:
            logger.warning(
                "[CALLBACK] Không kết nối được tới %s (attempt %d/%d)",
                callback_url,
                attempt,
                max_retries,
            )

        except Exception as e:
            logger.error(
                "[CALLBACK] Lỗi không xác định: %s (attempt %d/%d)",
                str(e),
                attempt,
                max_retries,
            )

        # Exponential backoff: 1s, 2s, 4s
        if attempt < max_retries:
            wait_time = 2 ** (attempt - 1)
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

async def send_progress(callback_url: str, job_id: str, stage: str) -> bool:
    """Gửi callback PROGRESS với stage hiện tại."""
    payload = CallbackPayload(
        job_id=job_id,
        event_type="PROGRESS",
        stage=stage,
    )
    return await send_callback(callback_url, payload)


async def send_succeeded_ingest(
    callback_url: str,
    job_id: str,
    chunks_count: int,
    chunk_manifest: list,
) -> bool:
    """Gửi callback SUCCEEDED cho ingest (kèm chunk manifest)."""
    payload = CallbackPayload(
        job_id=job_id,
        event_type="SUCCEEDED",
        chunks_count=chunks_count,
        chunk_manifest=chunk_manifest,
    )
    return await send_callback(callback_url, payload)


async def send_succeeded_visibility(
    callback_url: str,
    job_id: str,
    updated_count: int,
) -> bool:
    """Gửi callback SUCCEEDED cho hide/unhide."""
    payload = CallbackPayload(
        job_id=job_id,
        event_type="SUCCEEDED",
        updated_count=updated_count,
    )
    return await send_callback(callback_url, payload)


async def send_succeeded_delete(
    callback_url: str,
    job_id: str,
    deleted_count: int,
) -> bool:
    """Gửi callback SUCCEEDED cho delete."""
    payload = CallbackPayload(
        job_id=job_id,
        event_type="SUCCEEDED",
        deleted_count=deleted_count,
    )
    return await send_callback(callback_url, payload)


async def send_failed(
    callback_url: str,
    job_id: str,
    error_code: str,
    error_message: str,
) -> bool:
    """Gửi callback FAILED với thông tin lỗi."""
    payload = CallbackPayload(
        job_id=job_id,
        event_type="FAILED",
        error={"code": error_code, "message": error_message},
    )
    return await send_callback(callback_url, payload)
