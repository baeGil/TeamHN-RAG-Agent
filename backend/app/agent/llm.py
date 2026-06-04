"""Thin OpenAI chat wrapper with token accounting and JSON helpers."""
import json
import logging
import time
from typing import Any, Iterator, Optional

from ..config import get_settings

logger = logging.getLogger("rag.flow")


def _message_chars(messages: list[dict[str, Any]]) -> int:
    total = 0
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            total += len(content)
        elif isinstance(content, list):
            total += sum(len(str(part.get("text", ""))) for part in content if isinstance(part, dict))
    return total


class LLM:
    _quota_exceeded = False

    def __init__(self) -> None:
        self.settings = get_settings()
        self._client = None
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI

            if not self.settings.has_openai:
                raise RuntimeError(
                    "OPENAI_API_KEY chưa được cấu hình. Vui lòng điền vào backend/.env"
                )
            self._client = OpenAI(
                api_key=self.settings.openai_api_key,
                base_url=self.settings.openai_base_url,
            )
        return self._client

    def _track(self, resp) -> None:
        u = getattr(resp, "usage", None)
        if u:
            self.usage["prompt_tokens"] += getattr(u, "prompt_tokens", 0) or 0
            self.usage["completion_tokens"] += getattr(u, "completion_tokens", 0) or 0
        self.usage["calls"] += 1

    def chat(
        self,
        messages: list[dict[str, str]],
        fast: bool = False,
        temperature: float = 0.0,
        json_mode: bool = False,
        node: str = "unknown",
    ) -> str:
        if getattr(LLM, "_quota_exceeded", False):
            raise RuntimeError("OpenAI API Quota Exceeded (Short-circuited)")
        model = self.settings.llm_model_fast if fast else self.settings.llm_model
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        started = time.perf_counter()
        try:
            resp = self.client.chat.completions.create(**kwargs)
            duration_ms = (time.perf_counter() - started) * 1000
            self._track(resp)
            u = getattr(resp, "usage", None)
            content = resp.choices[0].message.content or ""
            logger.info(
                "RAG_FLOW llm_call node=%s mode=chat model=%s fast=%s json=%s duration_ms=%.1f "
                "prompt_tokens=%s completion_tokens=%s input_chars=%s output_chars=%s",
                node,
                model,
                fast,
                json_mode,
                duration_ms,
                getattr(u, "prompt_tokens", None) if u else None,
                getattr(u, "completion_tokens", None) if u else None,
                _message_chars(messages),
                len(content),
            )
            return content
        except Exception as e:
            duration_ms = (time.perf_counter() - started) * 1000
            err_msg = str(e).lower()
            if "quota" in err_msg or "exceeded" in err_msg or "429" in err_msg:
                LLM._quota_exceeded = True
            logger.exception(
                "RAG_FLOW llm_error node=%s mode=chat model=%s fast=%s json=%s duration_ms=%.1f",
                node,
                model,
                fast,
                json_mode,
                duration_ms,
            )
            raise

    def chat_vision(
        self, system: str, instruction: str, image_b64: str, temperature: float = 0.0, node: str = "vision"
    ) -> str:
        """Single-image vision transcription using the configured VLM model."""
        if getattr(LLM, "_quota_exceeded", False):
            raise RuntimeError("OpenAI API Quota Exceeded (Short-circuited)")
        messages = [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": instruction},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                    },
                ],
            },
        ]
        started = time.perf_counter()
        try:
            resp = self.client.chat.completions.create(
                model=self.settings.vlm_model,
                messages=messages,
                temperature=temperature,
            )
            duration_ms = (time.perf_counter() - started) * 1000
            self._track(resp)
            u = getattr(resp, "usage", None)
            content = resp.choices[0].message.content or ""
            logger.info(
                "RAG_FLOW llm_call node=%s mode=vision model=%s duration_ms=%.1f "
                "prompt_tokens=%s completion_tokens=%s output_chars=%s",
                node,
                self.settings.vlm_model,
                duration_ms,
                getattr(u, "prompt_tokens", None) if u else None,
                getattr(u, "completion_tokens", None) if u else None,
                len(content),
            )
            return content
        except Exception as e:
            duration_ms = (time.perf_counter() - started) * 1000
            err_msg = str(e).lower()
            if "quota" in err_msg or "exceeded" in err_msg or "429" in err_msg:
                LLM._quota_exceeded = True
            logger.exception(
                "RAG_FLOW llm_error node=%s mode=vision model=%s duration_ms=%.1f",
                node,
                self.settings.vlm_model,
                duration_ms,
            )
            raise

    def chat_json(self, messages: list[dict[str, str]], fast: bool = True, node: str = "unknown") -> Any:
        raw = self.chat(messages, fast=fast, json_mode=True, node=node)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            start, end = raw.find("{"), raw.rfind("}")
            if start != -1 and end != -1:
                return json.loads(raw[start : end + 1])
            raise

    def stream(
        self, messages: list[dict[str, str]], fast: bool = False, temperature: float = 0.0, node: str = "answer"
    ) -> Iterator[str]:
        if getattr(LLM, "_quota_exceeded", False):
            raise RuntimeError("OpenAI API Quota Exceeded (Short-circuited)")
        model = self.settings.llm_model_fast if fast else self.settings.llm_model
        started = time.perf_counter()
        first_token_ms = None
        chunks = 0
        output_chars = 0
        prompt_tokens = None
        completion_tokens = None
        try:
            stream = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                stream=True,
                stream_options={"include_usage": True},
            )
            self.usage["calls"] += 1
            for chunk in stream:
                if getattr(chunk, "usage", None):
                    u = chunk.usage
                    prompt_tokens = getattr(u, "prompt_tokens", None)
                    completion_tokens = getattr(u, "completion_tokens", None)
                    self.usage["prompt_tokens"] += getattr(u, "prompt_tokens", 0) or 0
                    self.usage["completion_tokens"] += getattr(u, "completion_tokens", 0) or 0
                if chunk.choices and chunk.choices[0].delta.content:
                    if first_token_ms is None:
                        first_token_ms = (time.perf_counter() - started) * 1000
                    text = chunk.choices[0].delta.content
                    chunks += 1
                    output_chars += len(text)
                    yield text
            duration_ms = (time.perf_counter() - started) * 1000
            logger.info(
                "RAG_FLOW llm_call node=%s mode=stream model=%s fast=%s duration_ms=%.1f "
                "first_token_ms=%s prompt_tokens=%s completion_tokens=%s input_chars=%s output_chars=%s chunks=%s",
                node,
                model,
                fast,
                duration_ms,
                round(first_token_ms, 1) if first_token_ms is not None else None,
                prompt_tokens,
                completion_tokens,
                _message_chars(messages),
                output_chars,
                chunks,
            )
        except Exception as e:
            duration_ms = (time.perf_counter() - started) * 1000
            err_msg = str(e).lower()
            if "quota" in err_msg or "exceeded" in err_msg or "429" in err_msg:
                LLM._quota_exceeded = True
            logger.exception(
                "RAG_FLOW llm_error node=%s mode=stream model=%s fast=%s duration_ms=%.1f "
                "first_token_ms=%s output_chars=%s chunks=%s",
                node,
                model,
                fast,
                duration_ms,
                round(first_token_ms, 1) if first_token_ms is not None else None,
                output_chars,
                chunks,
            )
            raise
