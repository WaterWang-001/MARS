import re

from .base_filter import BasePlatformFilter


class DoubanFilter(BasePlatformFilter):
    def __init__(self, config: dict, blacklist_file: str | None = None):
        super().__init__(config=config, blacklist_file=blacklist_file)
        self.anchor_tag_pattern = re.compile(r"<a\b[^>]*>.*?</a>", flags=re.IGNORECASE | re.DOTALL)
        self.href_pattern = re.compile(r'href=["\']([^"\']+)["\']', flags=re.IGNORECASE)
        self.title_attr_pattern = re.compile(r'title=["\']([^"\']+)["\']', flags=re.IGNORECASE)
        self.anchor_text_pattern = re.compile(r">\s*(.*?)\s*</a>", flags=re.IGNORECASE | re.DOTALL)
        self.subject_link_pattern = re.compile(
            r"https?://(movie|book|music)\.douban\.com/subject/(\d+)/?",
            flags=re.IGNORECASE,
        )
        self.catalog_item_pattern = re.compile(
            r"(?P<title>[^\n/。！？!?，,]{2,80}?)\s+"
            r"(?:(?:\[[^\]]{1,12}\]|【[^】]{1,12}】|［[^］]{1,12}］|\([^\)]{1,12}\)|（[^）]{1,12}）)\s*)?"
            r"[^/\n]{1,80}\s*/\s*"
            r"(?P<meta>\d{4}(?:[-.]\d{1,2}(?:[-.]\d{1,2})?)?|单曲|专辑|选集|album|ep)",
            flags=re.IGNORECASE,
        )
        self.action_tokens = ["看过", "想看", "在看", "读过", "想读", "在读", "听过", "想听", "在听"]
        self.domain_map = {"movie": "影视", "book": "图书", "music": "音乐"}
        self.domain_action_map = {
            "movie": {"想看": "想看", "看过": "看过", "在看": "在看"},
            "book": {
                "想读": "想读",
                "读过": "读过",
                "在读": "在读",
                "想看": "想读",
                "看过": "读过",
                "在看": "在读",
            },
            "music": {
                "想听": "想听",
                "听过": "听过",
                "在听": "在听",
                "想看": "想听",
                "看过": "听过",
                "在看": "在听",
            },
        }
        self.marketing_keywords = {
            "公号",
            "关注我",
            "私信",
            "引流",
            "推广",
            "豆邮",
            "加Q",
            "兼职",
            "代写",
            "互粉",
            "无费",
            "刷分",
            "水军",
            "好价",
            "作业",
            "京东自营",
            "旗舰店",
            "羊毛",
            "拼单",
            "捡漏",
            "代下",
        }
        self.broadcast_prefix_pattern = re.compile(
            r"^.*?的广播\s+[^\s]+\s+(想看|看过|在看|读过|想读|在读|听过|想听|在听)",
            flags=re.IGNORECASE,
        )
        self.metadata_suffix_pattern = re.compile(r"(?:\(\d{4}\))?\s*\d{4}\s*/\s*[^/]+\s*/.*$", flags=re.IGNORECASE)
        self.min_original_text_len = int((self.config or {}).get("min_original_text_len", 15))

    def get_hash_text(self, post: dict) -> str:
        return f"{post.get('title', '')} {post.get('content', '')} {post.get('quote_content', '')}"

    def get_original_text(self, post: dict) -> str:
        title = str(post.get("title", "") or "")
        content = str(post.get("content", "") or "")
        quote_content = str(post.get("quote_content", "") or "")

        excluded_titles = set()
        for source in (title, content, quote_content):
            for anchor in self._extract_subject_anchors_from_text(source):
                item_title = self._normalize_spaces(str(anchor.get("title", "") or ""))
                if item_title:
                    excluded_titles.add(item_title)
            for anchor in self._extract_text_anchors_from_text(source):
                item_title = self._normalize_spaces(str(anchor.get("title", "") or ""))
                if item_title:
                    excluded_titles.add(item_title)

        if quote_content:
            text = self._strip_html(content)
        else:
            text = self._strip_html(f"{title} {content}")
        text = self.broadcast_prefix_pattern.sub("", text)
        text = re.sub(r"[★☆]+", "", text)
        text = self.metadata_suffix_pattern.sub("", text)

        for item_title in sorted(excluded_titles, key=len, reverse=True):
            if len(item_title) < 2:
                continue
            escaped_title = re.escape(item_title)
            text = re.sub(rf"[《\[]?{escaped_title}[》\]]?", " ", text)

        text = self._normalize_spaces(text)
        return text.strip()

    def check_fatal_risk(self, post: dict) -> tuple[bool, str]:
        if not isinstance(post, dict):
            return True, "Invalid Post Type"

        text = self.get_hash_text(post)
        for kw in self.marketing_keywords:
            if kw in text:
                return True, f"Marketing Keyword ({kw})"

        original_text = self.get_original_text(post)
        if len(original_text) < self.min_original_text_len:
            return True, "Fatal: System Broadcast or Text too short (Low Entropy)"

        return False, "Safe"

    def extract_anchors(self, post: dict) -> list[str]:
        raw_content = str(post.get("content", "") or "")
        quote_content = str(post.get("quote_content", "") or "")

        candidates: list[str] = []
        seen_subject_keys: set[str] = set()
        seen_text_keys: set[str] = set()

        for source in (raw_content, quote_content):
            source_anchors = self._extract_subject_anchors_from_text(source)
            for anchor in source_anchors:
                key = f"{anchor['domain']}:{anchor['subject_id']}"
                if key in seen_subject_keys:
                    continue
                seen_subject_keys.add(key)
                candidates.append(self._build_anchor_tag(anchor))

            text_anchors = self._extract_text_anchors_from_text(source)
            for anchor in text_anchors:
                title = self._normalize_spaces(str(anchor.get("title", "") or ""))
                domain = str(anchor.get("domain", "") or "")
                if not title or not domain:
                    continue
                text_key = f"{domain}:{title.lower()}"
                if text_key in seen_text_keys:
                    continue
                seen_text_keys.add(text_key)
                candidates.append(self._build_anchor_tag(anchor))

        return self._clean_tags(candidates)

    def _extract_text_anchors_from_text(self, text: str) -> list[dict]:
        anchors: list[dict] = []
        normalized = self._strip_html(str(text or ""))
        if not normalized:
            return anchors

        seen_titles: set[str] = set()

        def _append(domain: str, title: str, context: str = ""):
            clean_title = self._normalize_spaces(title).strip("《》")
            if not clean_title or len(clean_title) < 2:
                return
            key = f"{domain}:{clean_title.lower()}"
            if key in seen_titles:
                return
            seen_titles.add(key)
            anchors.append(
                {
                    "domain": domain,
                    "subject_id": "",
                    "title": clean_title,
                    "action": self._extract_action_token(context or normalized, domain=domain),
                }
            )

        for match in self.catalog_item_pattern.finditer(normalized):
            title = match.group("title") or ""
            meta = (match.group("meta") or "").lower()
            window_start = max(0, match.start() - 60)
            context = normalized[window_start : match.end()]
            if meta in {"单曲", "专辑", "选集", "album", "ep"} or "数字(digital)" in normalized.lower() or "audio cd" in normalized.lower():
                _append("music", title, context)
            else:
                _append("book", title, context)

        for quoted in re.findall(r"《([^》]{2,80})》", normalized):
            lowered_context = normalized.lower()
            if any(k in lowered_context for k in ["单曲", "专辑", "album", "音乐", "听过", "想听", "在听"]):
                _append("music", quoted, normalized)
            elif any(k in normalized for k in ["电影", "影片", "短片", "看过", "想看", "在看", "第一季", "第二季"]):
                _append("movie", quoted, normalized)
            else:
                _append("book", quoted, normalized)

        tail_candidates = [seg.strip() for seg in re.split(r"[。！？!?]\s*", normalized) if seg.strip()]
        if tail_candidates:
            tail = tail_candidates[-1]
            has_rating_prefix = bool(re.match(r"^\s*\d{1,2}(?:\.\d)?[，,]", normalized))
            has_movie_context = any(k in normalized for k in ["看过", "想看", "在看", "电影", "影片", "短片"])
            tail_is_title_like = (
                ("‎" in tail)
                or bool(re.search(r"[A-Za-z]{2,}", tail))
                or bool(re.search(r"[（(]\d{4}[）)]\s*$", tail))
                or bool(re.search(r"[\u4e00-\u9fff]{2,}", tail))
            )
            if (
                2 <= len(tail) <= 70
                and "/" not in tail
                and "广播" not in tail
                and "http" not in tail.lower()
                and tail_is_title_like
                and (has_movie_context or has_rating_prefix)
            ):
                tail = tail.strip("《》")
                _append("movie", tail, normalized)

        return anchors

    def _extract_subject_anchors_from_text(self, text: str) -> list[dict]:
        anchors: list[dict] = []
        raw_text = str(text or "")
        if not raw_text:
            return anchors

        for match in self.anchor_tag_pattern.finditer(raw_text):
            anchor_html = match.group(0)
            href_match = self.href_pattern.search(anchor_html)
            if not href_match:
                continue

            href = href_match.group(1)
            subject_match = self.subject_link_pattern.search(href)
            if not subject_match:
                continue

            domain = subject_match.group(1).lower()
            subject_id = subject_match.group(2)

            title_match = self.title_attr_pattern.search(anchor_html)
            text_match = self.anchor_text_pattern.search(anchor_html)
            item_title = ""
            if title_match and str(title_match.group(1)).strip():
                item_title = self._strip_html(title_match.group(1))
            elif text_match and str(text_match.group(1)).strip():
                item_title = self._strip_html(text_match.group(1))

            window_start = max(0, match.start() - 400)
            action_window = raw_text[window_start:match.start()]
            action = self._extract_action_token(action_window, domain=domain)
            if not action:
                normalized_full_text = self._strip_html(raw_text)
                action = self._extract_action_token(normalized_full_text, domain=domain)

            anchors.append(
                {
                    "domain": domain,
                    "subject_id": subject_id,
                    "title": item_title,
                    "action": action,
                }
            )

        return anchors

    def _extract_action_token(self, text: str, domain: str = "") -> str:
        normalized = self._strip_html(str(text or ""))
        for token in self.action_tokens:
            if token in normalized:
                return self._normalize_action_for_domain(token, domain)
        return ""

    def _normalize_action_for_domain(self, action: str, domain: str) -> str:
        domain_key = str(domain or "").lower()
        mapping = self.domain_action_map.get(domain_key, {})
        return mapping.get(action, action)

    def _build_anchor_tag(self, anchor: dict) -> str:
        domain = str(anchor.get("domain", "") or "").lower()
        item_type = self.domain_map.get(domain, "作品")
        action = str(anchor.get("action", "") or "")
        title = self._normalize_spaces(str(anchor.get("title", "") or ""))
        subject_id = str(anchor.get("subject_id", "") or "")

        if title:
            base = f"{action}{item_type}:{title}" if action else f"{item_type}:{title}"
        else:
            base = f"{action}{item_type}:{subject_id}" if action else f"{item_type}:{subject_id}"

        if len(base) > 60:
            return base[:60].rstrip()
        return base

    def clean_text(self, post: dict) -> str:
        original_text = self.get_original_text(post)
        return self._normalize_spaces(original_text)
