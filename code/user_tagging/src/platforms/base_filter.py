import re
from abc import ABC, abstractmethod

import numpy as np

from ..shared_components.content_filter import ContentFilter


class BasePlatformFilter(ABC):
    def __init__(self, config: dict, blacklist_file: str | None = None):
        self.config = config or {}
        self.blacklist_file = blacklist_file
        self.shared_filter = ContentFilter(blacklist_file=blacklist_file)
        self.quality_rules = self.config.get("quality_rules", {})

    def is_user_qualified(self, user_profile: dict) -> tuple[bool, str]:
        return True, "Pass"

    @abstractmethod
    def get_hash_text(self, post: dict) -> str:
        raise NotImplementedError

    @abstractmethod
    def check_fatal_risk(self, post: dict) -> tuple[bool, str]:
        raise NotImplementedError

    @abstractmethod
    def extract_anchors(self, post: dict) -> list[str]:
        raise NotImplementedError

    @abstractmethod
    def clean_text(self, post: dict) -> str:
        raise NotImplementedError

    def is_post_meaningful(self, clean_text: str, valid_tags: list[str]) -> tuple[bool, str]:
        return self.shared_filter.is_post_meaningful(clean_text, valid_tags)

    def check_tag_dependent_fatal_risk(
        self,
        text: str,
        valid_tags: list[str],
        max_density_ratio: float = 0.5,
    ) -> tuple[bool, str]:
        return self.shared_filter.check_tag_dependent_fatal_risk(
            text,
            valid_tags,
            max_density_ratio=max_density_ratio,
        )

    def validate_batch_behavior(self, posts_list: list[dict], quality_config: dict) -> tuple[bool, str]:
        return self.shared_filter.validate_batch_behavior(posts_list, quality_config)

    def check_behavior_risk(self, posts: list[dict], api_client=None) -> tuple[bool, str]:
        behavior_cfg = self.quality_rules.get("behavior_risk", {})
        coherence_cfg = behavior_cfg.get("coherence", self.quality_rules.get("coherence", {}))

        min_personality_ratio = float(behavior_cfg.get("min_personality_ratio", 0.1))
        low_threshold = float(coherence_cfg.get("low_threshold", 0.15))
        high_threshold = float(coherence_cfg.get("high_threshold", 0.85))

        if min_personality_ratio > 0:
            is_human, reason = self._check_personality_ratio(posts, min_personality_ratio)
            if not is_human:
                return False, reason

        is_coherent, reason = self._check_semantic_coherence(
            posts=posts,
            api_client=api_client,
            low_threshold=low_threshold,
            high_threshold=high_threshold,
        )
        if not is_coherent:
            return False, reason

        return True, "Pass"

    def _check_personality_ratio(self, posts: list[dict], min_ratio: float) -> tuple[bool, str]:
        if not posts:
            return True, "Empty"
        if len(posts) < 3:
            return True, "Too Few Posts (Safe Pass)"

        hit_count = 0
        for post in posts:
            content = str(post.get("content", "") or "").strip()
            if self.shared_filter._has_strong_personality(content):
                hit_count += 1

        ratio = hit_count / len(posts)
        if ratio < min_ratio:
            return False, f"Non-Human: Low Personality Ratio ({hit_count}/{len(posts)} = {ratio:.2f})"
        return True, "Pass"

    def _check_semantic_coherence(
        self,
        posts: list[dict],
        api_client=None,
        low_threshold: float = 0.15,
        high_threshold: float = 0.85,
    ) -> tuple[bool, str]:
        if len(posts) < 3:
            return True, "pass_too_few"

        texts = [str(p.get("content", "") or "") for p in posts[-10:]]
        hashtag_start_count = sum(1 for t in texts if t.strip().startswith("#"))
        if texts and hashtag_start_count / len(texts) > 0.8:
            return False, "rejected_format_marketing_style"

        if api_client is None:
            return True, "pass_no_api_client"

        try:
            embeddings = api_client.get_embeddings(texts)
            if not embeddings:
                return True, "pass_embed_fail"

            vecs = np.array(embeddings)
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            normalized_vecs = vecs / (norms + 1e-9)
            sim_matrix = np.dot(normalized_vecs, normalized_vecs.T)

            n = len(texts)
            rows, cols = np.tril_indices(n, -1)
            if len(rows) == 0:
                return True, "pass_single"

            avg_sim = float(np.mean(sim_matrix[rows, cols]))
            if avg_sim < low_threshold:
                return False, f"rejected_incoherent (too_scattered: {avg_sim:.3f})"
            if avg_sim > high_threshold:
                return False, f"rejected_repetitive (too_similar: {avg_sim:.3f})"

            return True, f"pass_coherence (sim={avg_sim:.3f})"
        except Exception:
            return True, "pass_error"

    def _normalize_spaces(self, text: str) -> str:
        text = str(text or "")
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _clean_urls(self, text: str) -> str:
        return re.sub(r"http[s]?://\S+", "", str(text or ""))

    def _strip_html(self, text: str) -> str:
        text = re.sub(r"<[^>]+>", " ", str(text or ""))
        text = re.sub(r"<[^>]*$", " ", text)  # 清除末尾未闭合的标签
        return self._normalize_spaces(text)

    def _clean_tags(self, tags: list[str]) -> list[str]:
        result = []
        seen = set()
        for tag in tags:
            t = str(tag or "").strip().strip("#").strip(".,!?;:。，！？；：'\"”’")
            if not t:
                continue
            key = t.lower()
            if key in seen:
                continue
            if t.isdigit() or len(t) < 2 or len(t) > 60:
                continue
            seen.add(key)
            result.append(t)
        return result

    SYMBOL_SPAM_RE = re.compile(r"[!！]{4,}|[~～]{4,}|[.。]{6,}")

    def _is_symbol_spam(self, text: str) -> bool:
        if not text:
            return False
        return bool(self.SYMBOL_SPAM_RE.search(text))

    def is_post_content_cleanable(self, text: str):
        if self._is_symbol_spam(text):
            return False, "Symbol Spam"

        return True, None
