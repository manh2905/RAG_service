"""
main.py
-------
File khởi chạy chính cho FastAPI RAG Microservice.

Chức năng:
- Cấu hình CORS cho phép Node.js backend gọi tới.
- Mount tất cả API routes.
- Quản lý lifecycle: khởi tạo kết nối Qdrant khi startup,
  đóng kết nối khi shutdown.
- Cấu hình logging cho toàn bộ ứng dụng.

Chạy server:
    uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router as api_router
from core.config import get_settings
from core.database import get_qdrant_client, close_qdrant_client

# ── Cấu hình logging ────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════
# LIFESPAN — Quản lý vòng đời của ứng dụng
# ══════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Quản lý startup và shutdown của ứng dụng.

    Startup:
    - Load cấu hình từ .env
    - Khởi tạo kết nối tới Qdrant (tạo collection nếu chưa có)

    Shutdown:
    - Đóng kết nối Qdrant an toàn
    """
    # ── STARTUP ──────────────────────────────────────────────────
    settings = get_settings()
    logger.info("=" * 60)
    logger.info("🚀 RAG Education Service đang khởi động...")
    logger.info("   Qdrant URL   : %s", settings.QDRANT_URL)
    logger.info("   Collection   : %s", settings.QDRANT_COLLECTION_NAME)
    logger.info("   LLM Model    : %s", settings.GEMINI_LLM_MODEL)
    logger.info("   Embed Model  : %s", settings.GEMINI_EMBEDDING_MODEL)
    logger.info("   Top-K        : %d", settings.TOP_K)
    logger.info("   Sim Threshold: %.2f", settings.SIMILARITY_THRESHOLD)
    logger.info("=" * 60)

    # Khởi tạo kết nối Qdrant sớm để fail-fast nếu có lỗi
    try:
        await get_qdrant_client()
        logger.info("Khởi tạo thành công — Service sẵn sàng phục vụ ✓")
    except Exception as e:
        logger.error("⚠️  Không thể kết nối Qdrant: %s", str(e))
        logger.warning("Service vẫn khởi động nhưng các API sẽ lỗi cho đến khi Qdrant sẵn sàng")

    yield  # ← App chạy ở đây

    # ── SHUTDOWN ─────────────────────────────────────────────────
    logger.info("Đang tắt service...")
    await close_qdrant_client()
    logger.info("Service đã tắt an toàn ✓")


# ══════════════════════════════════════════════════════════════════
# KHỞI TẠO FASTAPI APP
# ══════════════════════════════════════════════════════════════════

app = FastAPI(
    title="RAG Education Service",
    description=(
        "Microservice xử lý RAG (Retrieval-Augmented Generation) "
        "cho hệ thống Trợ lý Giáo dục. Giao tiếp nội bộ với Node.js backend."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",       # Swagger UI tại /docs
    redoc_url="/redoc",     # ReDoc tại /redoc
)


# ── Cấu hình CORS — cho phép Node.js backend gọi tới ───────────
settings = get_settings()

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Mount tất cả API routes ─────────────────────────────────────
app.include_router(api_router)


# ══════════════════════════════════════════════════════════════════
# ROOT ENDPOINT
# ══════════════════════════════════════════════════════════════════

@app.get("/", tags=["Root"])
async def root():
    """Endpoint gốc — hiển thị thông tin cơ bản về service."""
    return {
        "service": "RAG Education Service",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/health",
    }


# ── Chạy trực tiếp bằng: python main.py ─────────────────────────
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
