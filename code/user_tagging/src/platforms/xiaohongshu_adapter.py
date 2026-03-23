import re
from typing import List

from .base_adapter import BasePlatformAdapter


class XiaohongshuAdapter(BasePlatformAdapter):
    platform_name = "xiaohongshu"

    def extract_tags(self, text: str, post: dict | None = None) -> List[str]:
        tags = re.findall(r"#([^#\s\n]+)", str(text or ""))
        cleaned = []
        seen = set()
        for t in tags:
            tag = t.strip().strip("#.,!?;:。，！？；：'\"”’")
            if not tag:
                continue
            key = tag.lower()
            if key in seen:
                continue
            seen.add(key)
            cleaned.append(tag)
        return cleaned
