"""Unit tests for custom_judges module."""

from custom_judges import _parse_json_response


def test_parse_json_response_clean():
    result = _parse_json_response('{"score": 1, "reason": "Good answer"}')
    assert result["score"] == 1
    assert result["reason"] == "Good answer"


def test_parse_json_response_with_whitespace():
    result = _parse_json_response('  {"score": 0, "reason": "Bad answer"}  ')
    assert result["score"] == 0


def test_parse_json_response_embedded_json():
    result = _parse_json_response('Here is the result: {"score": 1, "reason": "test"} done.')
    assert result["score"] == 1
    assert result["reason"] == "test"


def test_parse_json_response_invalid_returns_default():
    result = _parse_json_response("This is not JSON at all")
    assert result["score"] == 0
    assert "Failed to parse" in result["reason"]


def test_parse_json_response_empty():
    result = _parse_json_response("")
    assert result["score"] == 0
