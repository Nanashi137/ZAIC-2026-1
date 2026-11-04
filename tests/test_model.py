from __future__ import annotations

from typing import Any

import pytest

from lifemate.model import (
    InferenceSettings,
    ModelClientError,
    ModelRequest,
    OpenAICompatibleClient,
    RetrySettings,
)


class RecordingTransport:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def post(self, url, payload, headers, timeout_seconds):
        self.calls.append(
            {
                "url": url,
                "payload": payload,
                "headers": headers,
                "timeout_seconds": timeout_seconds,
            }
        )
        return {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "alarm__list", "arguments": "{}"},
                            }
                        ],
                    }
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
        }


def test_openai_compatible_client_builds_sequential_tool_request() -> None:
    transport = RecordingTransport()
    client = OpenAICompatibleClient(
        base_url="http://localhost:1234/v1",
        api_key="secret",
        timeout_seconds=12,
        transport=transport,
    )
    request = ModelRequest(
        messages=({"role": "user", "content": "test"},),
        tools=({"type": "function", "function": {"name": "alarm__list", "parameters": {}}},),
        settings=InferenceSettings(
            model="local-model", temperature=0.2, seed=12345, max_output_tokens=333
        ),
    )

    response = client.complete(request)

    sent = transport.calls[0]
    assert sent["url"] == "http://localhost:1234/v1/chat/completions"
    assert sent["headers"]["Authorization"] == "Bearer secret"
    assert sent["payload"]["parallel_tool_calls"] is False
    assert sent["payload"]["tool_choice"] == "required"
    assert sent["payload"]["temperature"] == 0.2
    assert sent["payload"]["seed"] == 12345
    assert sent["payload"]["max_tokens"] == 333
    assert response.tool_calls[0].name == "alarm__list"
    assert response.usage.as_dict() == {
        "input_tokens": 10,
        "output_tokens": 4,
        "total_tokens": 14,
        "requests": 1,
    }


def test_openai_compatible_client_can_restore_auto_tool_choice() -> None:
    transport = RecordingTransport()
    client = OpenAICompatibleClient(base_url="http://localhost:1234/v1", transport=transport)
    client.complete(
        ModelRequest(
            messages=(),
            tools=({"type": "function", "function": {"name": "fixture"}},),
            settings=InferenceSettings(model="fixture", tool_choice="auto"),
        )
    )

    assert transport.calls[0]["payload"]["tool_choice"] == "auto"


def test_inference_seed_must_be_non_negative() -> None:
    with pytest.raises(ValueError, match="seed must be non-negative"):
        InferenceSettings(model="fixture", seed=-1)


def test_official_openai_uses_current_token_field_and_can_omit_temperature() -> None:
    transport = RecordingTransport()
    client = OpenAICompatibleClient(base_url="https://api.openai.com/v1", transport=transport)
    request = ModelRequest(
        messages=({"role": "user", "content": "test"},),
        tools=(),
        settings=InferenceSettings(model="gpt-4.1-mini", temperature=None, max_output_tokens=222),
    )

    client.complete(request)

    payload = transport.calls[0]["payload"]
    assert payload["max_completion_tokens"] == 222
    assert "max_tokens" not in payload
    assert "temperature" not in payload


def test_transient_model_errors_use_bounded_exponential_backoff() -> None:
    class FlakyTransport(RecordingTransport):
        def __init__(self) -> None:
            super().__init__()
            self.attempts = 0

        def post(self, url, payload, headers, timeout_seconds):
            self.attempts += 1
            if self.attempts < 3:
                raise ModelClientError("rate limited", status_code=429, transient=True)
            return super().post(url, payload, headers, timeout_seconds)

    transport = FlakyTransport()
    sleeps: list[float] = []
    client = OpenAICompatibleClient(
        base_url="http://localhost:1234/v1",
        transport=transport,
        retry_settings=RetrySettings(max_retries=3, base_delay_seconds=1, max_delay_seconds=2),
        sleep=sleeps.append,
    )
    request = ModelRequest(messages=(), tools=(), settings=InferenceSettings(model="fixture"))

    client.complete(request)

    assert transport.attempts == 3
    assert sleeps == [1, 2]


def test_non_transient_model_error_is_not_retried() -> None:
    class BrokenTransport:
        attempts = 0

        def post(self, url, payload, headers, timeout_seconds):
            del url, payload, headers, timeout_seconds
            self.attempts += 1
            raise ModelClientError("bad request", status_code=400, transient=False)

    transport = BrokenTransport()
    client = OpenAICompatibleClient(base_url="http://localhost:1234/v1", transport=transport)

    with pytest.raises(ModelClientError, match="bad request"):
        client.complete(ModelRequest(messages=(), tools=(), settings=InferenceSettings(model="fixture")))
    assert transport.attempts == 1


def test_request_rate_limit_is_shared_by_client_calls() -> None:
    class FakeTime:
        value = 0.0

        def clock(self) -> float:
            return self.value

        def sleep(self, seconds: float) -> None:
            self.value += seconds

    fake_time = FakeTime()
    transport = RecordingTransport()
    client = OpenAICompatibleClient(
        base_url="http://localhost:1234/v1",
        transport=transport,
        requests_per_minute=60,
        sleep=fake_time.sleep,
        clock=fake_time.clock,
    )
    request = ModelRequest(messages=(), tools=(), settings=InferenceSettings(model="fixture"))

    client.complete(request)
    client.complete(request)

    assert fake_time.value == 1.0


def test_response_format_is_forwarded_when_present() -> None:
    transport = RecordingTransport()
    client = OpenAICompatibleClient(base_url="http://localhost:1234/v1", transport=transport)
    request = ModelRequest(
        messages=(),
        tools=(),
        settings=InferenceSettings(model="fixture", agent_output_mode="constrained-json"),
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "next_action",
                "schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "enum": ["alarm.list"]},
                        "arguments": {"type": "object"},
                    },
                    "required": ["name", "arguments"],
                    "additionalProperties": False,
                },
            },
        },
    )

    client.complete(request)

    payload = transport.calls[0]["payload"]
    assert payload["response_format"]["type"] == "json_schema"
    assert "tools" not in payload
    assert "tool_choice" not in payload
    assert "parallel_tool_calls" not in payload
