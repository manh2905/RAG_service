# Internal RAG API Contract v0.1.1

## 1. Quy ước chung

- Transport: HTTP + JSON.
- Boundary field naming: `snake_case`.
- Mọi NodeJS → Python request, trừ health check, phải có:

  `Authorization: Bearer <INTERNAL_SECRET>`

- Python → NodeJS callback dùng cùng secret.
- `attempt_count` luôn là processing-job attempt, không phải callback delivery retry.
- NodeJS tạo document ID, job ID, request ID và conversation ID. Tên field document ID tại Python boundary là `doc_id`.
- Python tạo `chunk_id`; giá trị này phải đồng thời là Qdrant point ID.

## 2. Operations

| Operation | Method và path | Kiểu xử lý |
|---|---|---|
| Health | `GET /api/health` | Synchronous, có thể public |
| Ingest | `POST /api/ingest` | `202 Accepted`, kết quả qua callback |
| Visibility | `PATCH /api/docs/{doc_id}/visibility` | `202 Accepted`, kết quả qua callback |
| Delete vectors | `DELETE /api/ingest/{doc_id}` | `202 Accepted`, kết quả qua callback |
| Query | `POST /api/query` | Synchronous `200` |
| Processing callback | `POST /api/internal/rag/processing-callback` | Python gọi NodeJS |

## 3. Ingest

### Request

| Field | Bắt buộc | Ý nghĩa |
|---|---|---|
| `doc_id` | Có | String của NodeJS document ID |
| `job_id` | Có | String của processing job ID |
| `attempt_count` | Có | Processing attempt hiện tại, số nguyên từ 1 |
| `subject_id` | Có trong MVP | Giá trị shim `mvp-global` |
| `file_path` | Có | Absolute path mà Python có thể đọc |
| `callback_url` | Có | NodeJS internal callback URL |
| `teacher_metadata` | Không | NodeJS gửi `{}`; không dùng cho authorization |

### Accepted response

HTTP `202` với `status="accepted"` và matching `job_id`. Response này chỉ xác nhận Python đã nhận job, không có nghĩa indexing đã hoàn tất.

### Callback

Python gửi progress hoặc terminal callback về `callback_url`.

Common fields:

- `job_id`;
- `attempt_count`;
- `event_type`: `PROGRESS`, `SUCCEEDED`, `FAILED` hoặc `CANCELLED`;
- optional `stage`;
- optional `error: {code, message}`.

Successful ingest callback cần:

- `chunks_count`;
- `chunk_manifest[]`.

Mỗi manifest item bắt buộc:

| Field | Yêu cầu |
|---|---|
| `chunk_index` | Số nguyên, zero-based và không trùng trong document |
| `chunk_id` | UUID thực dùng làm Qdrant point ID |
| `chunk_text` | Toàn bộ text đã được embedding/index |
| `content_hash` | SHA-256 lowercase hex của exact UTF-8 `chunk_text` |

Optional: `token_count`, `page_number`, `chapter`, `section`, `source_locator`.

PDF `page_number` dùng 1-based. DOCX/TXT không có physical page đáng tin cậy; khi không xác định có thể omit hoặc dùng `0`, NodeJS sẽ normalize thành `null`.

`text_preview` không thay thế được `chunk_text`. Nếu chưa có complete manifest, Python phải gửi failure thay vì success.

## 4. Visibility

`PATCH /api/docs/{doc_id}/visibility`

Body:

- `job_id`;
- `attempt_count`;
- `action`: `hide` hoặc `unhide`;
- `callback_url`.

Python trả `202 Accepted`, sau đó callback:

- terminal `event_type`;
- matching `job_id` và `attempt_count`;
- optional `updated_count`.

`hide` chỉ loại document khỏi retrieval; không xóa point. `unhide` cho phép retrieval trở lại.

## 5. Delete vectors

`DELETE /api/ingest/{doc_id}`

Body:

- `job_id`;
- `attempt_count`;
- `callback_url`.

Python xóa Qdrant points theo `doc_id`, sau đó callback terminal với optional `deleted_count`. Python không xóa file gốc hoặc dữ liệu MySQL.

## 6. Callback handling

NodeJS đối chiếu callback bằng `job_id + attempt_count`:

- Callback đúng attempt và job đang chạy: xử lý bình thường.
- Callback terminal gửi lại: ACK idempotent.
- Callback của attempt cũ: ACK nhưng không thay đổi dữ liệu.
- Payload hoặc token sai: trả `400` hoặc `401`.

Python có thể retry HTTP callback, nhưng mọi lần retry phải giữ nguyên processing `attempt_count`.

## 7. Query

### Request

| Field | Bắt buộc | Ý nghĩa |
|---|---|---|
| `conversation_id` | Có | Chat session ID do NodeJS tạo |
| `question` | Có | Câu hỏi hiện tại |
| `history` | Không | Các message gần nhất với role `user` hoặc `assistant` |
| `request_id` | Không | Correlation/idempotency extension do NodeJS tạo |
| `user_id` | Không | Correlation context; không dùng làm authorization tại Python |

NodeJS hiện có thể gửi đủ cả năm field. Python có thể khai báo hoặc bỏ qua các correlation extension không dùng đến.

### Answer response

- `answer`: string;
- `no_answer`: boolean;
- `confidence`: `high`, `medium` hoặc `low`;
- `citations[]`;
- optional `usage`.

Mỗi citation cần:

| Field | Bắt buộc | Ý nghĩa |
|---|---|---|
| `vector_node_id` | Có | Qdrant point ID của retrieved chunk |
| `doc_id` | Có | Document ID từ Qdrant payload |
| `snippet` hoặc `source_text` | Có | Source fragment thuộc retrieved chunk đã được đưa vào answer context |
| `page_number` | Không | Trang nguồn nếu có |
| `chapter`, `section` | Không | Locator theo heading nếu có |

NodeJS dùng `vector_node_id` để resolve chunk và lưu immutable citation snapshot. Không parse marker `[1]` để suy ra ID và không match bằng doc/page/text.

`usage` hiện có thể gồm:

- `prompt_tokens`;
- `completion_tokens`;
- `total_tokens`;
- `model`.

### No-answer

No-answer là kết quả nghiệp vụ hợp lệ:

- HTTP `200`;
- `no_answer=true`;
- `citations=[]`;
- `answer` có thể chứa thông báo không đủ dữ liệu;
- `usage` có thể là `null`.

## 8. Errors

Error chủ động nên dùng:

- `error_code`;
- `message`;
- `timestamp`.

FastAPI validation error dạng `detail` vẫn được NodeJS hỗ trợ.

Status cơ bản:

- `401`: thiếu hoặc sai internal token;
- `422`: request validation error;
- `500`: lỗi xử lý nội bộ;
- `502/503` do NodeJS sử dụng khi Python trả response không hợp lệ, timeout hoặc không kết nối được.

No-answer không được trả như HTTP error.

## 9. Giới hạn payload

NodeJS callback body limit hiện là 10 MB. Complete manifest phải nằm trong giới hạn này. Nếu tài liệu hợp lệ có thể vượt giới hạn, hai team cần thảo luận batching hoặc thay đổi giới hạn trước khi remote E2E; không tự thay contract trong v0.1.1.
