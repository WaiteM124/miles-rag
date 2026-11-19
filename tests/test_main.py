"""Unit tests for main FastAPI app."""

import os
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def mock_retriever():
    with patch("main.retriever") as mock:
        mock.search.return_value = [MagicMock(content="Test content", source="test.md", score=0.9)]
        yield mock


@pytest.fixture
def mock_answer():
    with patch("main.answer") as mock:
        mock.return_value = "Test answer based on documentation."
        yield mock


@pytest.fixture
def client(mock_retriever, mock_answer):
    from main import app

    return TestClient(app)


@pytest.fixture
def admin_client(mock_retriever, mock_answer):
    with patch.dict(os.environ, {"ADMIN_TOKEN": "test-secret"}):
        import importlib

        import main

        importlib.reload(main)
        from main import app

        yield TestClient(app)


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_query_endpoint(client, mock_retriever, mock_answer):
    response = client.post("/query", json={"question": "How do I post a journal?"})
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "refusal" in data
    assert "sources" in data
    assert data["answer"] == "Test answer based on documentation."
    assert data["refusal"] is False


def test_query_endpoint_refusal(client, mock_retriever, mock_answer):
    mock_answer.return_value = "REFUSAL: NOT IN DOCUMENTATION"
    response = client.post("/query", json={"question": "Unknown topic?"})
    assert response.status_code == 200
    data = response.json()
    assert data["refusal"] is True


def test_cache_clear_requires_auth(client):
    response = client.post("/api/health/cache/clear")
    assert response.status_code == 503


def test_cache_clear_rejects_bad_token(admin_client):
    response = admin_client.post(
        "/api/health/cache/clear", headers={"X-Admin-Token": "wrong-token"}
    )
    assert response.status_code == 401


def test_cache_clear_accepts_valid_token(admin_client):
    response = admin_client.post(
        "/api/health/cache/clear", headers={"X-Admin-Token": "test-secret"}
    )
    assert response.status_code == 200
    assert "cleared" in response.json()
