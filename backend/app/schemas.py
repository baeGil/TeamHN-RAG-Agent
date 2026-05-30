from typing import Optional

from pydantic import BaseModel


class UrlIn(BaseModel):
    url: str


class TextIn(BaseModel):
    text: str
    title: Optional[str] = "Văn bản"


class ChatIn(BaseModel):
    session_id: Optional[str] = None
    message: str


class SessionIn(BaseModel):
    title: Optional[str] = None
