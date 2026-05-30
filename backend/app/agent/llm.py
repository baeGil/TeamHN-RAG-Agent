"""Thin OpenAI chat wrapper with token accounting and JSON helpers."""
import json
from typing import Any, Iterator, Optional

from ..config import get_settings


class LLM:
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
    ) -> str:
        model = self.settings.llm_model_fast if fast else self.settings.llm_model
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        resp = self.client.chat.completions.create(**kwargs)
        self._track(resp)
        return resp.choices[0].message.content or ""

    def chat_vision(
        self, system: str, instruction: str, image_b64: str, temperature: float = 0.0
    ) -> str:
        """Single-image vision transcription using the configured VLM model."""
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
        resp = self.client.chat.completions.create(
            model=self.settings.vlm_model,
            messages=messages,
            temperature=temperature,
        )
        self._track(resp)
        return resp.choices[0].message.content or ""

    def chat_json(self, messages: list[dict[str, str]], fast: bool = True) -> Any:
        raw = self.chat(messages, fast=fast, json_mode=True)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            start, end = raw.find("{"), raw.rfind("}")
            if start != -1 and end != -1:
                return json.loads(raw[start : end + 1])
            raise

    def stream(
        self, messages: list[dict[str, str]], fast: bool = False, temperature: float = 0.0
    ) -> Iterator[str]:
        model = self.settings.llm_model_fast if fast else self.settings.llm_model
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
                self.usage["prompt_tokens"] += getattr(u, "prompt_tokens", 0) or 0
                self.usage["completion_tokens"] += getattr(u, "completion_tokens", 0) or 0
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
