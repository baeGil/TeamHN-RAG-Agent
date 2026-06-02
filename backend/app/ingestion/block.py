from dataclasses import dataclass
from typing import Optional


@dataclass
class Block:
    page: Optional[int] = None
    section: Optional[str] = None
    text: str = ""
    embed_text: Optional[str] = None