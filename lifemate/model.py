"""Configurable OpenAI-compatible chat-completions client.

The production client uses only the standard library. Tests use ScriptedModelClient,
which performs no network access.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol
from urllib.parse import urlparse

from .compliance import SampleDeadline


class ModelClientError(RuntimeError):
    """A model request failed or returned an unusable envelope."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        transient: bool = False,
        retry_after_seconds: float | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.transient = transient
        self.retry_after_seconds = retry_after_seconds


@dataclass(frozen=True)
class InferenceSettings:
    model: str
    temperature: float | None = 0.0
    max_output_tokens: int = 1024
    max_tokens_field: Literal["auto", "max_tokens", "max_completion_tokens"] = "auto"
    tool_choice: Literal["auto", "required"] = "required"

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("model must be non-empty")
        if self.temperature is not None and not 0 <= self.temperature <= 2:
            raise ValueError("temperature must be between 0 and 2")
        if self.max_output_tokens < 1:
            raise ValueError("max_output_tokens must be positive")
        if self.tool_choice not in {"auto", "required"}:
            raise ValueError("tool_choice must be auto or required")


@dataclass(frozen=True)
class RetrySettings:
    max_retries: int = 3
    base_delay_seconds: float = 1.0
    max_delay_seconds: float = 8.0

    def __post_init__(self) -> None:
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if self.base_delay_seconds < 0 or self.max_delay_seconds < 0:
            raise ValueError("retry delays must be non-negative")
        if self.max_delay_seconds < self.base_delay_seconds:
            raise ValueError("max retry delay must be at least the base delay")


@dataclass(frozen=True)
class ModelToolCall:
    id: str
    name: str
    arguments: str


@dataclass(frozen=True)
class ModelUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    requests: int = 1

    def __add__(self, other: ModelUsage) -> ModelUsage:
        return ModelUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            requests=self.requests + other.requests,
        )

    def as_dict(self) -> dict[str, int]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "requests": self.requests,
        }


@dataclass(frozen=True)
class ModelResponse:
    content: str | None = None
    tool_calls: tuple[ModelToolCall, ...] = ()
    refusal: str | None = None
    usage: ModelUsage = field(default_factory=ModelUsage)


@dataclass(frozen=True)
class ModelRequest:
    messages: tuple[Mapping[str, Any], ...]
    tools: tuple[Mapping[str, Any], ...]
    settings: InferenceSettings
    deadline: SampleDeadline | None = None


class ModelClient(Protocol):
    name: str

    def complete(self, request: ModelRequest) -> ModelResponse: ...


class JsonTransport(Protocol):
    def post(
        self,
        url: str,
        payload: Mapping[str, Any],
        headers: Mapping[str, str],
        timeout_seconds: float,
    ) -> Mapping[str, Any]: ...


class UrllibJsonTransport:
    """Small injectable JSON POST transport for OpenAI-compatible endpoints."""

    def post(
        self,
        url: str,
        payload: Mapping[str, Any],
        headers: Mapping[str, str],
        timeout_seconds: float,
    ) -> Mapping[str, Any]:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(url, data=body, headers=dict(headers), method="POST")
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = _safe_http_error(exc)
            transient = exc.code == 429 or exc.code in {408, 409, 500, 502, 503, 504}
            raise ModelClientError(
                f"model endpoint returned HTTP {exc.code}: {detail}",
                status_code=exc.code,
                transient=transient,
                retry_after_seconds=_retry_after_seconds(exc),
            ) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise ModelClientError(
                f"model transport failed: {type(exc).__name__}: {exc}", transient=True
            ) from exc
        try:
            decoded = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ModelClientError("model endpoint returned invalid JSON") from exc
        if not isinstance(decoded, dict):
            raise ModelClientError("model endpoint response must be a JSON object")
        return decoded


class OpenAICompatibleClient:
    """Chat-completions client usable with OpenAI-compatible providers/base URLs."""

    name = "openai-compatible"

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str | None = None,
        timeout_seconds: float = 60.0,
        retry_settings: RetrySettings | None = None,
        requests_per_minute: float | None = None,
        headers: Mapping[str, str] | None = None,
        transport: JsonTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        base_url = base_url.rstrip("/")
        if not base_url.startswith(("http://", "https://")):
            raise ValueError("base_url must use http:// or https://")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if requests_per_minute is not None and requests_per_minute <= 0:
            raise ValueError("requests_per_minute must be positive when configured")
        self.endpoint = f"{base_url}/chat/completions"
        self.timeout_seconds = timeout_seconds
        self.retry_settings = retry_settings or RetrySettings()
        self.requests_per_minute = requests_per_minute
        request_headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if api_key:
            request_headers["Authorization"] = f"Bearer {api_key}"
        request_headers.update(headers or {})
        self.headers = request_headers
        self.transport = transport or UrllibJsonTransport()
        self._sleep = sleep
        self._clock = clock
        self._rate_lock = threading.Lock()
        self._next_request_at = 0.0
        self._official_openai = (urlparse(base_url).hostname or "").lower() == "api.openai.com"

    def complete(self, request: ModelRequest) -> ModelResponse:
        payload: dict[str, Any] = {
            "model": request.settings.model,
            "messages": [dict(message) for message in request.messages],
            "tools": [dict(tool) for tool in request.tools],
            "tool_choice": request.settings.tool_choice,
            "parallel_tool_calls": False,
        }
        token_field = request.settings.max_tokens_field
        if token_field == "auto":
            token_field = "max_completion_tokens" if self._official_openai else "max_tokens"
        payload[token_field] = request.settings.max_output_tokens
        if request.settings.temperature is not None:
            payload["temperature"] = request.settings.temperature
        response = self._post_with_retries(payload, request.deadline)
        return _parse_chat_response(response)

    def _post_with_retries(
        self,
        payload: Mapping[str, Any],
        deadline: SampleDeadline | None,
    ) -> Mapping[str, Any]:
        for attempt in range(self.retry_settings.max_retries + 1):
            if deadline is not None:
                deadline.check("model request")
            self._wait_for_rate_limit(deadline)
            request_timeout = self.timeout_seconds
            if deadline is not None:
                request_timeout = deadline.timeout_for(self.timeout_seconds, "model request")
            try:
                response = self.transport.post(self.endpoint, payload, self.headers, request_timeout)
                if deadline is not None:
                    deadline.check("model request")
                return response
            except ModelClientError as exc:
                if not exc.transient or attempt >= self.retry_settings.max_retries:
                    raise
                delay = min(
                    self.retry_settings.max_delay_seconds,
                    self.retry_settings.base_delay_seconds * (2**attempt),
                )
                if exc.retry_after_seconds is not None:
                    delay = min(
                        self.retry_settings.max_delay_seconds,
                        max(delay, exc.retry_after_seconds),
                    )
                if deadline is not None:
                    deadline.require_duration(delay, "model retry backoff")
                self._sleep(delay)
        raise AssertionError("unreachable")

    def _wait_for_rate_limit(self, deadline: SampleDeadline | None) -> None:
        if self.requests_per_minute is None:
            return
        interval = 60.0 / self.requests_per_minute
        with self._rate_lock:
            now = self._clock()
            delay = self._next_request_at - now
            if delay > 0:
                if deadline is not None:
                    deadline.require_duration(delay, "model rate-limit wait")
                self._sleep(delay)
                now = self._clock()
            self._next_request_at = max(now, self._next_request_at) + interval


class ScriptedModelClient:
    """Deterministic no-network client for unit and integration smoke tests."""

    name = "scripted"

    def __init__(self, responses: list[ModelResponse] | tuple[ModelResponse, ...]) -> None:
        self._responses = list(responses)
        self.requests: list[ModelRequest] = []

    def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        if not self._responses:
            raise ModelClientError("scripted model has no response remaining")
        return self._responses.pop(0)


def _parse_chat_response(response: Mapping[str, Any]) -> ModelResponse:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise ModelClientError("model response has no choices[0]")
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise ModelClientError("model response has no assistant message")

    content = message.get("content")
    if content is not None and not isinstance(content, str):
        raise ModelClientError("assistant content must be a string or null")
    refusal = message.get("refusal")
    if refusal is not None and not isinstance(refusal, str):
        refusal = str(refusal)

    calls: list[ModelToolCall] = []
    raw_calls = message.get("tool_calls") or []
    if not isinstance(raw_calls, list):
        raise ModelClientError("assistant tool_calls must be an array")
    for index, raw_call in enumerate(raw_calls):
        if not isinstance(raw_call, dict) or not isinstance(raw_call.get("function"), dict):
            raise ModelClientError(f"assistant tool_calls[{index}] is malformed")
        function = raw_call["function"]
        call_id = raw_call.get("id")
        name = function.get("name")
        arguments = function.get("arguments")
        if not isinstance(call_id, str) or not isinstance(name, str) or not isinstance(arguments, str):
            raise ModelClientError(f"assistant tool_calls[{index}] has invalid fields")
        calls.append(ModelToolCall(call_id, name, arguments))

    raw_usage = response.get("usage") or {}
    if not isinstance(raw_usage, dict):
        raw_usage = {}
    input_tokens = _non_negative_int(raw_usage.get("prompt_tokens", raw_usage.get("input_tokens", 0)))
    output_tokens = _non_negative_int(raw_usage.get("completion_tokens", raw_usage.get("output_tokens", 0)))
    total_tokens = _non_negative_int(raw_usage.get("total_tokens", input_tokens + output_tokens))
    return ModelResponse(
        content=content,
        tool_calls=tuple(calls),
        refusal=refusal,
        usage=ModelUsage(input_tokens, output_tokens, total_tokens, 1),
    )


def _non_negative_int(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


def _safe_http_error(exc: urllib.error.HTTPError) -> str:
    try:
        raw = exc.read(4096).decode("utf-8", errors="replace")
        decoded = json.loads(raw)
        message = decoded.get("error", {}).get("message") if isinstance(decoded, dict) else None
        if isinstance(message, str):
            return message[:300]
    except (OSError, ValueError, AttributeError):
        pass
    return exc.reason if isinstance(exc.reason, str) else "request rejected"


def _retry_after_seconds(exc: urllib.error.HTTPError) -> float | None:
    value = exc.headers.get("Retry-After") if exc.headers is not None else None
    if value is None:
        return None
    try:
        seconds = float(value)
    except ValueError:
        return None
    return seconds if seconds >= 0 else None
