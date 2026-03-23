import re
from abc import ABC, abstractmethod
from typing import Dict, List


class BasePlatformAdapter(ABC):
    platform_name: str = "base"

    @abstractmethod
    def extract_tags(self, text: str, post: dict | None = None) -> List[str]:
        raise NotImplementedError

    def normalize_user(self, user_data: dict) -> Dict:
        return {
            "user_id": str(user_data.get("user_id", "")).strip(),
            "username": user_data.get("display_name") or user_data.get("username") or "Unknown",
            "bio": str(user_data.get("bio", "") or "").strip(),
            "platform": self.platform_name,
        }

    def clean_text(self, text: str) -> str:
        text = str(text or "")
        text = re.sub(r"http[s]?://\S+", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def normalize_post(self, post: dict) -> dict:
        content = self.clean_text(post.get("content", ""))
        quote_content = self.clean_text(post.get("quote_content", ""))
        return {
            **post,
            "content": content,
            "quote_content": quote_content,
            "valid_tags": self.extract_tags(content, post=post),
        }
