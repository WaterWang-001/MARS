import os
import re
from typing import List

from .base_adapter import BasePlatformAdapter


class WeiboAdapter(BasePlatformAdapter):
    platform_name = "weibo"

    def __init__(self, fandom_words_file: str | None = None):
        self.fandom_words = set()
        if fandom_words_file and os.path.exists(fandom_words_file):
            with open(fandom_words_file, "r", encoding="utf-8") as f:
                self.fandom_words = {line.strip() for line in f if line.strip()}

    def extract_tags(self, text: str, post: dict | None = None) -> List[str]:
        tags = re.findall(r"#([^#\n]+)#", str(text or ""))
        cleaned = []
        seen = set()
        for t in tags:
            tag = t.strip().strip(".,!?;:。，！？；：'\"”’")
            if not tag:
                continue
            lower_tag = tag.lower()
            if lower_tag in seen:
                continue
            seen.add(lower_tag)
            cleaned.append(tag)
        return cleaned

    def is_fandom_text(self, text: str) -> bool:
        if not self.fandom_words:
            return False
        low = str(text or "").lower()
        return any(w.lower() in low for w in self.fandom_words)
