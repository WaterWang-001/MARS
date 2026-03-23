import re
import os
import difflib
import gzip
from datetime import datetime
import numpy as np
import pandas as pd


class ContentFilter:
    def __init__(self, blacklist_file: str | None = None, **kwargs):
        self.hashtag_blacklist = set()
        if blacklist_file and os.path.exists(blacklist_file):
            with open(blacklist_file, "r", encoding="utf-8") as f:
                for line in f:
                    tag = line.strip().replace("#", "")
                    if tag:
                        self.hashtag_blacklist.add(tag)

        # 平台可注入词库（默认给跨平台兜底）
        self.fandom_keywords = set(kwargs.get("fandom_keywords", set()))
        self.marketing_keywords = set(kwargs.get("marketing_keywords", {
            "领券", "优惠券", "点击链接", "私信我", "加购", "下单"
        }))
        self.junk_hashtag_keywords = set(kwargs.get("junk_hashtag_keywords", {
            "签到", "打卡", "互粉", "回关", "日常"
        }))
        self.fandom_abbr = set(kwargs.get("fandom_abbr", set()))
        self.junk_tag_suffixes = list(kwargs.get("junk_tag_suffixes", []))

        # 通用规则
        self.symbol_spam_pattern = re.compile(r"[!！]{4,}|[~～]{4,}|[.。]{6,}")
        self.emoji_pattern = re.compile(r"[\U00010000-\U0010ffff]")

        self.first_person_pronouns = {"我", "俺", "本人", "博主", "主包"}
        self.implicit_personality_words = {"觉得", "打算", "好累", "崩溃", "无语", "救命", "真香", "离谱"}
        self.fake_first_person_patterns = ["我的评分", "我的主页", "关注我", "私信我", "联系我", "主页有惊喜"]

    def _has_strong_personality(self, text):
        text = str(text or "").strip()
        if not text:
            return False

        processed_text = text
        for pattern in self.fake_first_person_patterns:
            processed_text = processed_text.replace(pattern, "")

        if any(pronoun in processed_text for pronoun in self.first_person_pronouns):
            return True

        if any(word in processed_text for word in self.implicit_personality_words):
            return True

        return False

    def is_post_meaningful(self, clean_text, valid_tags=None):
        text = str(clean_text or "").strip()
        tags = valid_tags or []

        if not tags:
            if len(text) < 5:
                return False, "No Valid Tags & Text Too Short"

        if len(text) > 30:
            original_len = len(text.encode("utf-8"))
            compressed_len = len(gzip.compress(text.encode("utf-8")))
            if original_len > 0 and (compressed_len / original_len) < 0.2:
                return False, "Low Entropy"

        return True, "Pass"

    def filter_hashtags(self, raw_hashtags):
        if not raw_hashtags:
            return []
        valid_tags, seen_tags = [], set()
        for tag in raw_hashtags:
            clean_tag = str(tag).strip().replace("#", "")
            for suffix in self.junk_tag_suffixes:
                clean_tag = clean_tag.replace(suffix, "")
            clean_tag = clean_tag.strip()
            low = clean_tag.lower()

            if not clean_tag or low in seen_tags:
                continue

            is_junk = len(clean_tag) < 2 or len(clean_tag) > 30 or clean_tag.isdigit()
            if not is_junk:
                is_junk = bool(re.match(r"^[\W_]+$", clean_tag))
            if not is_junk:
                is_junk = any(k in low for k in self.junk_hashtag_keywords)

            if not is_junk:
                valid_tags.append(clean_tag)
                seen_tags.add(low)
        return valid_tags

    def check_tag_dependent_fatal_risk(self, text, valid_tags, max_density_ratio=0.5):
        text = str(text or "")
        if not valid_tags:
            return False, "Safe"

        for tag in valid_tags:
            clean_tag = str(tag).replace("#", "").strip()
            if clean_tag in self.hashtag_blacklist:
                return True, f"Fatal: Blacklisted Tag ({clean_tag})"

        total_len = max(len(text), 1)
        if sum(len(t) for t in valid_tags) / total_len > max_density_ratio:
            return True, "Fatal: Low Post Density"
        return False, "Safe"

    def is_post_content_cleanable(self, text):
        text = str(text or "").strip()
        if self.symbol_spam_pattern.search(text):
            return False, "Symbol Spam"
        return True, "Clean"

    def check_semantic_coherence(self, posts, api_client=None, low_threshold=0.15, high_threshold=0.85):
        if len(posts) < 3:
            return True, "pass_too_few"
        texts = [p.get("content", "") for p in posts[-10:]]
        hashtag_start_count = sum(1 for t in texts if str(t).strip().startswith("#"))
        if hashtag_start_count / len(texts) > 0.8:
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
            avg_sim = np.mean(sim_matrix[rows, cols])
            if avg_sim < low_threshold:
                return False, f"rejected_incoherent (too_scattered: {avg_sim:.3f})"
            if avg_sim > high_threshold:
                return False, f"rejected_repetitive (too_similar: {avg_sim:.3f})"
            return True, f"pass_coherence (sim={avg_sim:.3f})"
        except Exception:
            return True, "pass_error"

    def check_aggregator_behavior(self, posts_list, min_ratio=0.1):
        if not posts_list:
            return True, "Empty"
        total = len(posts_list)
        if total < 3:
            return True, "Too Few Posts (Safe Pass)"
        hit_count = sum(
            1 for post in posts_list
            if self._has_strong_personality(str(post.get("content", "")).strip())
        )
        ratio = hit_count / total
        if ratio < min_ratio:
            return False, f"Non-Human: Low Personality Ratio ({hit_count}/{total} = {ratio:.2f})"
        return True, "Pass"

    def check_behavior_risk(
        self,
        posts,
        min_personality_ratio=0.1,
        api_client=None,
        low_thresh=0.15,
        high_thresh=0.85,
    ):
        is_human, h_reason = self.check_aggregator_behavior(posts, min_ratio=min_personality_ratio)
        if not is_human:
            return False, h_reason

        is_coherent, c_reason = self.check_semantic_coherence(
            posts, api_client=api_client, low_threshold=low_thresh, high_threshold=high_thresh
        )
        if not is_coherent:
            return False, c_reason
        return True, "Pass"

    def _calculate_fuzzy_repetition(self, contents):
        if len(contents) < 2:
            return 0.0

        sample = contents[:10]
        total_sim = 0.0
        pairs = 0
        for i in range(len(sample)):
            for j in range(i + 1, len(sample)):
                sim = difflib.SequenceMatcher(None, sample[i], sample[j]).ratio()
                total_sim += sim
                pairs += 1

        return (total_sim / pairs) if pairs > 0 else 0.0

    def _extract_primary_content(self, post):
        if not isinstance(post, dict):
            return ""
        return str(post.get("content", "") or "").strip()

    def validate_batch_behavior(self, posts_list, quality_config=None):
        if not posts_list:
            return True, "Empty Batch"

        quality_config = quality_config or {}
        valid_contents = []
        timestamps = []

        for post in posts_list:
            c = self._extract_primary_content(post)
            if c:
                valid_contents.append(c)
            if post.get("created_at"):
                try:
                    timestamps.append(pd.to_datetime(post["created_at"]))
                except Exception:
                    pass

        rep_threshold = float(quality_config.get("repetition_threshold", 0.3))
        fuzzy_rep_threshold = float(quality_config.get("fuzzy_repetition_threshold", 0.8))
        if len(valid_contents) >= 3:
            unique_count = len(set(valid_contents))
            exact_rep_rate = 1.0 - (unique_count / len(valid_contents))
            if exact_rep_rate > rep_threshold:
                return False, f"SPAM: High Exact Repetition ({exact_rep_rate:.2f})"

            fuzzy_rep_rate = self._calculate_fuzzy_repetition(valid_contents)
            if fuzzy_rep_rate > fuzzy_rep_threshold:
                return False, f"SPAM: High Fuzzy Repetition ({fuzzy_rep_rate:.2f})"

        gap_threshold = float(quality_config.get("min_avg_time_gap", 60))
        if len(timestamps) > 4:
            timestamps.sort()
            diffs = [(timestamps[i + 1] - timestamps[i]).total_seconds() for i in range(len(timestamps) - 1)]
            burst_diffs = [d for d in diffs if d < 3600]
            if len(burst_diffs) > 3:
                avg_gap = sum(burst_diffs) / len(burst_diffs)
                if avg_gap < gap_threshold:
                    return False, f"BOT: Abnormal Frequency ({avg_gap:.1f}s)"

        return True, "Pass"

    def _extract_post_timestamp(self, post):
        candidate_keys = [
            "created_at",
            "create_time",
            "publish_time",
            "timestamp",
            "time",
        ]

        raw_value = None
        for key in candidate_keys:
            if key in post and post.get(key) not in (None, ""):
                raw_value = post.get(key)
                break

        if raw_value is None:
            return None

        if isinstance(raw_value, (int, float)):
            try:
                value = float(raw_value)
                if value > 1e12:
                    value /= 1000.0
                return datetime.fromtimestamp(value)
            except Exception:
                return None

        text = str(raw_value).strip()
        if not text:
            return None

        try:
            value = float(text)
            if value > 1e12:
                value /= 1000.0
            return datetime.fromtimestamp(value)
        except Exception:
            pass

        normalized = text.replace("Z", "+00:00")
        try:
            return datetime.fromisoformat(normalized)
        except Exception:
            return None
