import re
from typing import List

from .base_adapter import BasePlatformAdapter


class DoubanAdapter(BasePlatformAdapter):
    platform_name = "douban"

    def extract_tags(self, text: str, post: dict | None = None) -> List[str]:
        result: list[str] = []
        seen = set()

        # 豆瓣常见结构化字段：group_name / subject / tags
        if post:
            candidates = []
            for key in ("group_name", "subject", "category", "tags", "tag_list"):
                value = post.get(key)
                if not value:
                    continue
                if isinstance(value, list):
                    candidates.extend([str(x).strip() for x in value if str(x).strip()])
                else:
                    candidates.extend([x.strip() for x in re.split(r"[,，;；|/]", str(value)) if x.strip()])

            for item in candidates:
                clean_item = item.strip().strip("#")
                if clean_item and clean_item.lower() not in seen:
                    seen.add(clean_item.lower())
                    result.append(clean_item)

        # 兼容正文 #话题# 形式
        inline_tags = re.findall(r"#([^#\n]+)#", str(text or ""))
        for tag in inline_tags:
            clean_tag = tag.strip().strip("#.,!?;:。，！？；：'\"”’")
            if clean_tag and clean_tag.lower() not in seen:
                seen.add(clean_tag.lower())
                result.append(clean_tag)

        return result
