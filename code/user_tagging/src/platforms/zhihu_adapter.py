import re
from typing import List

from .base_adapter import BasePlatformAdapter


class ZhihuAdapter(BasePlatformAdapter):
    platform_name = "zhihu"

    def extract_tags(self, text: str, post: dict | None = None) -> List[str]:
        result: list[str] = []
        seen = set()

        if post:
            topics = post.get("topics") or post.get("topic_names") or []
            if isinstance(topics, str):
                topics = [x.strip() for x in re.split(r"[,，;；|]", topics) if x.strip()]
            for topic in topics:
                topic_str = str(topic).strip()
                if topic_str and topic_str.lower() not in seen:
                    seen.add(topic_str.lower())
                    result.append(topic_str)

        inline_topics = re.findall(r"#([^#\n]+)#", str(text or ""))
        for topic in inline_topics:
            topic_str = topic.strip()
            if topic_str and topic_str.lower() not in seen:
                seen.add(topic_str.lower())
                result.append(topic_str)

        return result
