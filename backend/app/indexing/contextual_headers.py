"""Contextual Chunk Headers for index, rerank, and generation context."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


HEADER_VERSION = 1


@dataclass(frozen=True)
class ContextualHeader:
    title: Optional[str] = None
    section: Optional[str] = None
    page: Optional[int] = None

    @property
    def is_empty(self) -> bool:
        return not any((self.title, self.section, self.page))

    def format(self, *, include_page: bool = True) -> str:
        parts: list[str] = []
        if self.title:
            parts.append(f"Document: {self.title}")
        if self.section:
            parts.append(f"Section: {self.section}")
        if include_page and self.page:
            parts.append(f"Page: {self.page}")
        return " | ".join(parts)


def build_header(
    title: Optional[str],
    section: Optional[str],
    page: Optional[int] = None,
) -> ContextualHeader:
    return ContextualHeader(
        title=(title or "").strip() or None,
        section=(section or "").strip() or None,
        page=page,
    )


def apply_contextual_header(
    text: str,
    *,
    title: Optional[str],
    section: Optional[str],
    page: Optional[int] = None,
    enabled: bool = True,
    include_page: bool = True,
) -> str:
    if not enabled:
        return text
    header = build_header(title, section, page)
    if header.is_empty:
        return text
    return f"{header.format(include_page=include_page)}\n{text}"


def signature(enabled: bool, include_page: bool) -> str:
    return f"cch:v{HEADER_VERSION}:enabled={int(enabled)}:page={int(include_page)}"
