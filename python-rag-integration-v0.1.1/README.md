# NodeJS–Python RAG Integration v0.1.1

Package này dùng để thống nhất boundary giữa NodeJS/Core và Python Data/RAG trước khi chạy remote integration.

## 1. Mục tiêu và ownership

- Client chỉ gọi public API của NodeJS.
- NodeJS quản lý authentication, authorization, document metadata/file gốc, processing job, chat history, citation snapshot, usage log và MySQL.
- Python quản lý parsing, chunking, embedding, Qdrant, retrieval và LLM generation.
- Python không ghi MySQL. NodeJS không truy cập trực tiếp Qdrant.

## 2. Các thay đổi cần thống nhất trước remote integration

| Vấn đề | Hiện trạng Python | Target cần thống nhất | Lý do |
|---|---|---|---|
| Processing attempt | Callback retry đang có thể thay đổi `attempt_count` | Nhận từ request và giữ nguyên trong mọi callback | Node dùng `job_id + attempt_count` để chống stale callback |
| Chunk manifest | Chỉ có `text_preview` | Trả full `chunk_text`, `content_hash` và actual Qdrant point ID | Node cần lưu chunk metadata và kiểm tra citation |
| Citation identity | Chưa có vector/point ID | Mỗi citation trả `vector_node_id` | Node resolve citation về đúng indexed chunk |
| Internal authentication | Python gửi Bearer khi callback nhưng chưa kiểm tra request inbound | Ingest, visibility, delete và query xác thực Bearer token | Chỉ NodeJS được gọi internal RAG API |

Chi tiết field và payload nằm trong `API_CONTRACT.md`.

## 3. Điểm cần chốt khi chạy remote E2E

Đã thống nhất:

- Node `RAG_INTERNAL_TOKEN` và Python `INTERNAL_SECRET` dùng cùng một secret, tối thiểu 32 ký tự.
- Python chỉ đọc file; NodeJS sở hữu việc lưu và quản lý file gốc.
- Chỉ Python sở hữu Qdrant.
- NodeJS gửi `teacher_metadata: {}`; identity/authorization không được đưa vào Qdrant metadata.
- `subject_id=mvp-global` chỉ là compatibility shim của MVP, không phải business scope hoặc phân quyền.

Cần chốt theo môi trường triển khai:

- shared volume hoặc physical path để Python đọc được `file_path`;
- `RAG_SERVICE_URL` và `callback_url` mà hai phía truy cập được;
- Python chạy trên host hay cùng Docker network với NodeJS.

## 4. Checklist trước remote E2E

- Bearer token hợp lệ được chấp nhận; thiếu hoặc sai token bị từ chối.
- Processing `attempt_count` không đổi khi callback HTTP retry.
- Complete manifest có full text, đúng SHA-256 và actual Qdrant point ID.
- Citation map được về indexed chunk qua `vector_node_id`.
- No-answer trả HTTP `200`, `no_answer=true` và zero citations.
- Contract tests phía Python pass mà không cần gọi Gemini/Qdrant thật.

## 5. Nội dung package

- `API_CONTRACT.md`: contract chính thức cần đối chiếu.
- `examples/ingest.json`: ingest request, accepted response và callbacks.
- `examples/document-operations.json`: hide, unhide và delete.
- `examples/query.json`: query, answer, no-answer và error.

Mỗi file example gom các message của cùng một luồng. Từng property con như `ingest_request` hoặc `success_callback` là một HTTP body độc lập; không gửi toàn bộ object ngoài cùng.

## 6. Ngoài phạm vi v0.1.1

OCR, PPTX, object storage, durable queue, public reprocess, callback batching, RAG quality tuning và production infrastructure chưa thuộc lần tích hợp này.
