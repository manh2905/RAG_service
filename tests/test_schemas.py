"""
tests/test_schemas.py
---------------------
Test validation cho các schema Pydantic.
"""

from models.schemas import (
    IngestRequest,
    QueryRequest,
    VisibilityRequest,
    CallbackPayload,
)
# pyrefly: ignore [missing-import]
import pytest
from pydantic import ValidationError

def test_ingest_request_valid():
    data = {
        "doc_id": "123",
        "job_id": "job1",
        "subject_id": "sub1",
        "file_path": "/tmp/file.pdf",
        "callback_url": "http://localhost/cb",
    }
    req = IngestRequest(**data)
    assert req.doc_id == "123"
    assert req.job_id == "job1"
    assert req.teacher_metadata == {}

def test_query_request_valid():
    data = {
        "question": "test",
        "conversation_id": "conv1",
        "history": [{"role": "user", "content": "hi"}]
    }
    req = QueryRequest(**data)
    assert req.question == "test"
    assert len(req.history) == 1
    assert req.history[0].role == "user"

def test_visibility_request_invalid_action():
    data = {
        "job_id": "job1",
        "action": "delete", # Invalid action
        "callback_url": "http://localhost/cb",
    }
    with pytest.raises(ValidationError):
        VisibilityRequest(**data)

def test_callback_payload_valid():
    data = {
        "job_id": "job1",
        "event_type": "PROGRESS",
        "stage": "parsing"
    }
    payload = CallbackPayload(**data)
    assert payload.event_type == "PROGRESS"
    assert payload.stage == "parsing"
