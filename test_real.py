import asyncio
from services.ingestion import ingest_document_background
from models.schemas import IngestRequest
import os
import logging

logging.basicConfig(level=logging.INFO)

async def main():
    cwd = os.getcwd()
    pdf_path = os.path.join(cwd, "tests", "file.pdf")
    req = IngestRequest(
        doc_id="doc_real_test",
        job_id="job_real_1",
        attempt_count=1,
        subject_id="sub_test",
        file_path=pdf_path,
        callback_url="http://localhost:3000/callback" 
    )
    print(f"Bắt đầu xử lý file: {pdf_path}")
    await ingest_document_background(req)
    print("Tiến trình đã chạy xong!")

if __name__ == "__main__":
    asyncio.run(main())
