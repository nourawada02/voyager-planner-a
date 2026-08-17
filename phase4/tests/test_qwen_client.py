"""Hermetic tests for the planner-owned Qwen decision provider
(Checkpoint Phase 4 D.0 §7). `urllib.request.urlopen` is monkeypatched
throughout -- no real socket is ever opened."""

from __future__ import annotations

import json
import secrets
import urllib.error
import urllib.request

import pytest

from phase4.qwen_client import (
    QwenConfigurationError,
    QwenDecisionProvider,
    QwenTransportError,
    qwen_api_key_present,
)


def _ephemeral_test_secret() -> str:
    return f"ephemeral-test-token-{secrets.token_hex(16)}"


class _FakeUrlopenResponse:
    def __init__(self, body: bytes):
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *args) -> bool:
        return False


def test_missing_base_url_raises_configuration_error(monkeypatch):
    monkeypatch.delenv("QWEN_BASE_URL", raising=False)
    provider = QwenDecisionProvider()
    with pytest.raises(QwenConfigurationError):
        provider.generate("system", "user")


def test_non_https_base_url_raises_configuration_error(monkeypatch):
    monkeypatch.setenv("QWEN_BASE_URL", "http://insecure.example.com/v1")
    provider = QwenDecisionProvider()
    with pytest.raises(QwenConfigurationError):
        provider.generate("system", "user")


def test_key_presence_check_never_reads_the_value(monkeypatch):
    secret_value = _ephemeral_test_secret()
    monkeypatch.setenv("QWEN_API_KEY", secret_value)
    assert qwen_api_key_present() is True
    monkeypatch.delenv("QWEN_API_KEY", raising=False)
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    assert qwen_api_key_present() is False


def test_dashscope_api_key_is_an_accepted_fallback_name(monkeypatch):
    monkeypatch.delenv("QWEN_API_KEY", raising=False)
    monkeypatch.setenv("DASHSCOPE_API_KEY", _ephemeral_test_secret())
    assert qwen_api_key_present() is True


def test_key_sent_only_via_authorization_header_never_in_body_or_url(monkeypatch):
    secret_value = _ephemeral_test_secret()
    monkeypatch.setenv("QWEN_API_KEY", secret_value)
    monkeypatch.setenv("QWEN_BASE_URL", "https://qwen.example.com/v1")
    captured = {}

    def _fake_urlopen(request, timeout=None):
        captured["url"] = request.full_url
        captured["headers"] = dict(request.headers)
        captured["body"] = request.data.decode("utf-8")
        return _FakeUrlopenResponse(json.dumps({"choices": [{"message": {"content": "{}"}}]}).encode("utf-8"))

    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)
    provider = QwenDecisionProvider()
    provider.generate("system prompt", "user prompt")

    assert secret_value not in captured["url"]
    assert secret_value not in captured["body"]
    assert captured["headers"].get("Authorization") == f"Bearer {secret_value}"


def test_request_uses_json_object_response_format_and_no_thinking(monkeypatch):
    monkeypatch.setenv("QWEN_API_KEY", _ephemeral_test_secret())
    monkeypatch.setenv("QWEN_BASE_URL", "https://qwen.example.com/v1")
    captured = {}

    def _fake_urlopen(request, timeout=None):
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return _FakeUrlopenResponse(json.dumps({"choices": [{"message": {"content": "{}"}}]}).encode("utf-8"))

    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)
    QwenDecisionProvider().generate("system", "user")

    assert captured["body"]["response_format"] == {"type": "json_object"}
    assert captured["body"]["enable_thinking"] is False
    assert captured["body"]["temperature"] == 0


def test_default_model_is_qwen3_7_flash(monkeypatch):
    monkeypatch.delenv("QWEN_MODEL", raising=False)
    monkeypatch.delenv("QWEN_GENERATOR_MODEL", raising=False)
    assert QwenDecisionProvider().model == "qwen3.7-flash"


def test_model_override_via_environment(monkeypatch):
    monkeypatch.setenv("QWEN_MODEL", "qwen-custom-test-model")
    assert QwenDecisionProvider().model == "qwen-custom-test-model"


def test_http_error_maps_to_transport_error_with_only_status_code(monkeypatch):
    monkeypatch.setenv("QWEN_API_KEY", _ephemeral_test_secret())
    monkeypatch.setenv("QWEN_BASE_URL", "https://qwen.example.com/v1")

    def _fake_urlopen(request, timeout=None):
        raise urllib.error.HTTPError(url="https://qwen.example.com/v1/chat/completions", code=401, msg="Unauthorized", hdrs=None, fp=None)

    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)
    with pytest.raises(QwenTransportError) as excinfo:
        QwenDecisionProvider().generate("system", "user")
    assert "401" in str(excinfo.value)


def test_network_error_maps_to_transport_error(monkeypatch):
    monkeypatch.setenv("QWEN_API_KEY", _ephemeral_test_secret())
    monkeypatch.setenv("QWEN_BASE_URL", "https://qwen.example.com/v1")

    def _fake_urlopen(request, timeout=None):
        raise urllib.error.URLError("simulated DNS failure")

    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)
    with pytest.raises(QwenTransportError):
        QwenDecisionProvider().generate("system", "user")


def test_provider_repr_never_contains_the_key(monkeypatch):
    secret_value = _ephemeral_test_secret()
    monkeypatch.setenv("QWEN_API_KEY", secret_value)
    provider = QwenDecisionProvider()
    assert secret_value not in repr(provider)
    assert secret_value not in str(provider)
