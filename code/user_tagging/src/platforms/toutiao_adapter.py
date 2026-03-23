import re
from typing import List

from .base_adapter import BasePlatformAdapter


class ToutiaoAdapter(BasePlatformAdapter):
    platform_name = "toutiao"

    def extract_tags(self, text: str, post: dict | None = None) -> List[str]:
        result: list[str] = []
        seen = set()

        # 优先读取结构化字段（不同数据源字段名可能不同）
        if post:
            for key in ("topics", "topic_names", "labels", "tag_list"):
                values = post.get(key)
                if not values:
                    continue
                if isinstance(values, str):
                    values = [x.strip() for x in re.split(r"[,，;；|/]", values) if x.strip()]
                for value in values:
                    tag = str(value).strip().strip("#")
                    if tag and tag.lower() not in seen:
                        seen.add(tag.lower())
                        result.append(tag)

        # 兼容正文中的 hashtag 形式
        inline_tags = re.findall(r"#([^#\n]+)#?", str(text or ""))
        for tag in inline_tags:
            clean_tag = tag.strip().strip("#.,!?;:。，！？；：'\"”’")
            if clean_tag and clean_tag.lower() not in seen:
                seen.add(clean_tag.lower())
                result.append(clean_tag)

        return result
