"""
tests/test_api.py
-----------------
Test các API endpoints bằng FastAPI TestClient.
Chúng ta sẽ mock các background tasks để không cần Qdrant thực.
"""

from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import sys

# Mock external heavy dependencies before importing app
sys.modules['qdrant_client'] = MagicMock()
sys.modules['qdrant_client.models'] = MagicMock()

# Mock llama_parse
sys.modules['llama_parse'] = MagicMock()

# Mock llama_index and its submodules
llama_index_mock = MagicMock()
sys.modules['llama_index'] = llama_index_mock
sys.modules['llama_index.core'] = MagicMock()
sys.modules['llama_index.core.node_parser'] = MagicMock()
sys.modules['llama_index.core.schema'] = MagicMock()
sys.modules['llama_index.llms'] = MagicMock()
sys.modules['llama_index.llms.gemini'] = MagicMock()
sys.modules['llama_index.llms.google_genai'] = MagicMock()
sys.modules['llama_index.embeddings'] = MagicMock()
sys.modules['llama_index.embeddings.gemini'] = MagicMock()
sys.modules['llama_index.embeddings.google_genai'] = MagicMock()

from main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

@patch("api.routes.ingest_document_background")
def test_ingest_accepted(mock_bg_task):
    payload = {
        "doc_id": "doc1",
        "job_id": "job1",
        "attempt_count": 1,
        "subject_id": "sub1",
        "file_path": "/tests/file.pdf",
        "callback_url": "http://test/cb"
    }
    response = client.post("/api/ingest", json=payload)
    assert response.status_code == 202
    assert response.json()["status"] == "accepted"
    assert response.json()["job_id"] == "job1"
    # Verify background task was called
    mock_bg_task.assert_called_once()

@patch("api.routes.hide_document_background")
def test_hide_document_accepted(mock_bg_task):
    payload = {
        "job_id": "job2",
        "attempt_count": 1,
        "action": "hide",
        "callback_url": "http://test/cb"
    }
    response = client.patch("/api/docs/doc1/visibility", json=payload)
    assert response.status_code == 202
    assert response.json()["job_id"] == "job2"
    mock_bg_task.assert_called_once()

@patch("api.routes.delete_document_background")
def test_delete_document_accepted(mock_bg_task):
    payload = {
        "job_id": "job3",
        "attempt_count": 1,
        "callback_url": "http://test/cb"
    }
    # Using json via request body is correct for DELETE here (httpx supports it)
    response = client.request("DELETE", "/api/ingest/doc1", json=payload)
    assert response.status_code == 202
    assert response.json()["job_id"] == "job3"
    mock_bg_task.assert_called_once()

@patch("api.routes.process_query")
def test_query_endpoint(mock_process_query):
    # Mock return value for process_query
    from models.schemas import QueryResponse
    mock_process_query.return_value = QueryResponse(
        answer="Mocked answer",
        citations=[],
        confidence="high",
        no_answer=False,
    )
    
    payload = {
        "question": "What is AI?",
        "conversation_id": "conv1",
        "history": []
    }
    
    import asyncio
    
    # We need to mock the async function properly. TestClient runs async endpoints using run_in_threadpool.
    # So we can just mock it as an async function.
    async def mock_pq(*args, **kwargs):
        return mock_process_query.return_value
        
    with patch("api.routes.process_query", new=mock_pq):
        response = client.post("/api/query", json=payload)
        
    assert response.status_code == 200
    assert response.json()["answer"] == "Mocked answer"


def test_citations_are_renumbered_to_match_returned_order():
    from services.rag_engine import _extract_citations

    result_1 = MagicMock(
        id="point-1",
        payload={
            "doc_id": "doc-1",
            "page_number": 7,
            "text": "Page citation trong API công khai bắt đầu từ 1.",
        },
    )
    result_2 = MagicMock(
        id="point-2",
        payload={"doc_id": "doc-1", "page_number": 9, "text": "Không được dùng."},
    )
    result_3 = MagicMock(
        id="point-3",
        payload={
            "doc_id": "doc-1",
            "page_number": 14,
            "text": "Bảng đáp án xác nhận page là 1-based.",
        },
    )

    extracted = _extract_citations(
        answer="Trang đầu tiên là trang 1 [1], không phải trang 0 [3].",
        results=[result_1, result_2, result_3],
        question="Page citation bắt đầu từ mấy?",
    )

    assert extracted is not None
    normalized_answer, citations = extracted
    assert normalized_answer == "Trang đầu tiên là trang 1 [1], không phải trang 0 [2]."
    assert [citation.vector_node_id for citation in citations] == ["point-1", "point-3"]


def test_citation_snippet_selects_relevant_text_inside_chunk():
    from services.rag_engine import _select_relevant_snippet

    text = (
        "Metadata của mỗi chunk gồm document_id và chunk_index. " * 8
        + "\nPage citation trong API công khai bắt đầu từ 1 (1-based). "
        + "Adapter PDF phải chuyển chỉ số từ 0-based sang 1-based."
    )

    snippet = _select_relevant_snippet(
        text=text,
        question="Page citation bắt đầu từ mấy?",
        answer="Page citation bắt đầu từ 1.",
    )

    assert "bắt đầu từ 1" in snippet


def test_invalid_citation_marker_is_rejected():
    from services.rag_engine import _extract_citations

    result = MagicMock(
        id="point-1",
        payload={"doc_id": "doc-1", "page_number": 1, "text": "Nội dung nguồn."},
    )

    assert _extract_citations("Câu trả lời [2].", [result], "Câu hỏi") is None
