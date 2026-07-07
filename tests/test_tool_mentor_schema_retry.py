"""Guardrails for mentor structured-output reliability.

Thinking models can spend 100+ completion tokens on reasoning before emitting
the JSON body, so a too-small max_tokens budget truncates the JSON mid-string
and the empty-parse fallback then fails key validation. These tests pin the
mentor token budget and the one-retry recovery path.
"""

import json

import pytest

import tau2.voice.audio_native.mentor.tool_boundary_mentor as tbm
from tau2.data_model.message import ToolCall, ToolMessage
from tau2.data_model.simulation import AudioNativeConfig
from tau2.registry import registry


class _FakeMessage:
    def __init__(self, content):
        self.content = content
        self.refusal = None


class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content):
        self.choices = [_FakeChoice(content)]


def _mentor():
    env = registry.get_env_constructor("retail")()
    cfg = AudioNativeConfig(
        provider="openai",
        tool_mentor_enabled=True,
        tool_mentor_mode="llm",
        tool_mentor_model="gemini/gemini-3.5-flash",
        tool_mentor_reasoning_effort="low",
    )
    return tbm.ToolBoundaryMentor(config=cfg, environment=env)


TOOL_CALL = ToolCall(
    id="c1",
    name="get_order_details",
    arguments={"order_id": "#W1"},
    requestor="assistant",
)
TOOL_RESULT = ToolMessage(
    id="c1",
    role="tool",
    requestor="assistant",
    content=json.dumps({"status": "delivered"}),
    error=False,
)
TRANSCRIPT = [{"role": "user", "text": "hello, checking on my order"}]


def test_truncated_then_valid_response_recovers_via_retry(monkeypatch):
    calls = []

    def fake_completion(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return _FakeResponse('{\n  "agent_note": "truncated mid stri')
        return _FakeResponse(json.dumps({"agent_note": "Order is delivered."}))

    monkeypatch.setattr(tbm, "completion", fake_completion)
    note = _mentor()._llm_post_read(TOOL_CALL, TOOL_RESULT, TRANSCRIPT)
    assert note == "Order is delivered."
    assert len(calls) == 2


def test_persistent_invalid_response_raises_with_truncation_diagnostics(monkeypatch):
    calls = []

    def fake_completion(**kwargs):
        calls.append(kwargs)
        return _FakeResponse('{"unexpected": true}')

    monkeypatch.setattr(tbm, "completion", fake_completion)
    with pytest.raises(ValueError) as exc_info:
        _mentor()._llm_post_read(TOOL_CALL, TOOL_RESULT, TRANSCRIPT)
    message = str(exc_info.value)
    assert "missing keys" in message
    assert "content_len" in message
    assert len(calls) == tbm.MENTOR_VALIDATION_ATTEMPTS


def test_json_budget_covers_thinking_tokens():
    assert tbm.MENTOR_JSON_MAX_TOKENS >= 2048
