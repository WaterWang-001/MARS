import csv
import os
import re
import sqlite3
from collections import Counter
from typing import Callable


def _build_regex_anchor_extractor(hashtag_pattern: re.Pattern) -> Callable[[dict], list[str]]:
    def _extract(post: dict) -> list[str]:
        full_text = f"{post.get('title') or ''} {post.get('content') or ''} {post.get('quote_content') or ''}"
        if not full_text.strip():
            return []
        tags = hashtag_pattern.findall(full_text)
        return [str(t).strip() for t in tags if str(t).strip()]

    return _extract


def _sample_tag_counter(
    db_path: str,
    target_date: str,
    sample_size: int,
    anchor_extractor: Callable[[dict], list[str]],
):
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    sql = f"""
    SELECT title, content, quote_content
    FROM post
    WHERE created_at LIKE '{target_date}%'
    ORDER BY RANDOM()
    LIMIT {sample_size}
    """
    cursor.execute(sql)
    rows = cursor.fetchall()
    if not rows:
        conn.close()
        return 0, Counter()

    tag_counter = Counter()
    total_samples = len(rows)
    for row in rows:
        post = dict(row)
        anchors = anchor_extractor(post)
        unique_tags_in_post = {str(t).strip() for t in anchors if str(t).strip()}
        tag_counter.update(unique_tags_in_post)
    conn.close()
    return total_samples, tag_counter


def generate_tag_stats(
    db_path: str,
    target_date: str,
    output_dir: str,
    hashtag_pattern_str: str | None = None,
    sample_size: int = 500000,
    platform_filter=None,
):
    """仅生成频率抽样统计 CSV（不做 blacklist 计算），返回 stats_csv。"""
    if not os.path.exists(db_path):
        print(f"❌ Database not found: {db_path}")
        return None

    if platform_filter is None and not hashtag_pattern_str:
        print("❌ Either platform_filter or hashtag_pattern_str is required to generate sampled tag stats")
        return None

    anchor_extractor = None
    if platform_filter is not None:
        anchor_extractor = platform_filter.extract_anchors
    elif hashtag_pattern_str:
        hashtag_pattern = re.compile(hashtag_pattern_str)
        anchor_extractor = _build_regex_anchor_extractor(hashtag_pattern)

    os.makedirs(output_dir, exist_ok=True)
    csv_file = os.path.join(output_dir, f"sampled_tag_stats_{target_date}.csv")

    try:
        total_samples, tag_counter = _sample_tag_counter(
            db_path=db_path,
            target_date=target_date,
            sample_size=sample_size,
            anchor_extractor=anchor_extractor,
        )
        with open(csv_file, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Rank", "Tag", "Frequency", "Coverage (%)", "Status"])
            if total_samples > 0:
                for rank, (tag, count) in enumerate(tag_counter.most_common(), 1):
                    coverage = (count / total_samples) * 100
                    writer.writerow([rank, tag, count, f"{coverage:.3f}%", "SAMPLED"])

        print(f"✅ Sampled tag stats generated: {csv_file}")
        return csv_file
    except Exception as e:
        print(f"❌ Error generating sampled tag stats: {e}")
        return None


def generate_blacklist(
    db_path: str,
    target_date: str,
    output_dir: str,
    hashtag_pattern_str: str | None,
    sample_size: int = 500000,
    blacklist_threshold_percent: float = 0.02,
    platform_filter=None,
):
    """生成黑名单与统计 CSV，返回 (blacklist_txt, stats_csv)。"""
    if not os.path.exists(db_path):
        print(f"❌ Database not found: {db_path}")
        return None, None

    if platform_filter is None and not hashtag_pattern_str:
        print("❌ Either platform_filter or hashtag_pattern_str is required when blacklist is enabled")
        return None, None

    os.makedirs(output_dir, exist_ok=True)
    csv_file = os.path.join(output_dir, f"sampled_tag_stats_{target_date}.csv")
    txt_file = os.path.join(output_dir, f"blacklist_{target_date}.txt")

    try:
        if platform_filter is not None:
            anchor_extractor = platform_filter.extract_anchors
        else:
            hashtag_pattern = re.compile(hashtag_pattern_str)
            anchor_extractor = _build_regex_anchor_extractor(hashtag_pattern)

        total_samples, tag_counter = _sample_tag_counter(
            db_path=db_path,
            target_date=target_date,
            sample_size=sample_size,
            anchor_extractor=anchor_extractor,
        )

        if total_samples == 0:
            print("⚠️ No data found for this date.")
            return None, None

        abs_threshold = total_samples * (blacklist_threshold_percent / 100)

        blacklist_tags = []
        with open(csv_file, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Rank", "Tag", "Frequency", "Coverage (%)", "Status"])
            for rank, (tag, count) in enumerate(tag_counter.most_common(), 1):
                coverage = (count / total_samples) * 100
                is_spam = count >= abs_threshold
                status = "BLACKLIST" if is_spam else "SAFE"
                writer.writerow([rank, tag, count, f"{coverage:.3f}%", status])
                if is_spam:
                    blacklist_tags.append(tag)

        with open(txt_file, "w", encoding="utf-8") as f:
            for tag in sorted(blacklist_tags):
                f.write(f"{tag}\n")

        print(f"✅ Blacklist generated: {txt_file}")
        print(f"✅ Stats generated: {csv_file}")
        return txt_file, csv_file

    except Exception as e:
        print(f"❌ Error: {e}")
        return None, None
