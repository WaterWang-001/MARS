import csv
import os
import re
import threading
import time

from ...platforms.base_filter import BasePlatformFilter
from .benchmark_selector import BenchmarkSelector


class BenchGenerationService:
    def __init__(
        self,
        state_manager,
        prompt_manager,
        api_client,
        stopwords_path,
        output_dir,
        config,
        platform_filter: BasePlatformFilter,
        blacklist_tags_path=None,
    ):
        self.state_manager = state_manager
        self.prompt_manager = prompt_manager
        self.api_client = api_client
        self.output_dir = output_dir
        self.config = config
        self.platform_filter = platform_filter
        self.platform_name = self.platform_filter.config.get("platform_name", "unknown")

        buffer_rules = self.config.get("buffer_rules", {})
        quality_rules = self.config.get("quality_rules", {})
        behavior_risk_cfg = quality_rules.get("behavior_risk", {})

        self.trigger_threshold = int(buffer_rules.get("trigger_threshold", 5))
        self.max_posts_per_call = int(buffer_rules.get("max_posts_per_call", 50))
        self.quality_config = {
            "repetition_threshold": float(behavior_risk_cfg.get("repetition_threshold", 0.3)),
            "fuzzy_repetition_threshold": float(behavior_risk_cfg.get("fuzzy_repetition_threshold", 0.8)),
            "min_avg_time_gap": float(behavior_risk_cfg.get("min_avg_time_gap", 60)),
        }

        profile_bench_ids_path = self.config.get("profile_bench_ids_path") or self.config.get("profile_bench_csv_path")
        self.selector = BenchmarkSelector(
            output_dir=self.output_dir,
            platform_name=self.platform_name,
            profile_bench_ids_path=profile_bench_ids_path,
        )

        density_cfg = quality_rules.get("anchor_density", {})
        self.anchor_density_upper_bound = float(density_cfg.get("upper_bound", 1.0))
        self.anchor_density_lower_bound = float(density_cfg.get("lower_bound", 0.2))
        self.min_unique_tags = int(density_cfg.get("min_unique_tags", 0))
        stats_dir = self.output_dir
        if os.path.basename(os.path.normpath(stats_dir)) != self.platform_name:
            stats_dir = os.path.join(stats_dir, self.platform_name)
        os.makedirs(stats_dir, exist_ok=True)
        self.stats_file = os.path.join(stats_dir, f"density_distribution_{self.platform_name}.csv")
        self.stats_lock = threading.Lock()
        self.filter_slow_log_seconds = float(os.getenv("BENCH_FILTER_SLOW_SEC", "0.20"))
        self.user_process_slow_log_seconds = float(os.getenv("BENCH_USER_PROCESS_SLOW_SEC", "5.0"))
        self.profile_all_users = os.getenv("BENCH_PROFILE_ALL_USERS", "0") == "1"
        self.skip_empty_progress_update = os.getenv("BENCH_SKIP_EMPTY_PROGRESS_UPDATE", "1") == "1"
        if not os.path.exists(self.stats_file):
            with open(self.stats_file, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(["user_id", "total_posts", "total_anchors", "density_avg", "unique_tags", "status"])

    def _log_user_process_timing(
        self,
        user_id: str,
        status: str,
        timings: dict,
        raw_count: int,
        valid_count: int,
        combined_count: int,
    ):
        total = sum(float(v) for v in timings.values())
        if not self.profile_all_users and total < self.user_process_slow_log_seconds:
            return
        print(
            "⏱️ [Process Slow] "
            f"user={user_id} status={status} raw={raw_count} valid={valid_count} combined={combined_count} "
            f"| total={total:.3f}s "
            f"(state_get={timings.get('state_get', 0.0):.3f}s, "
            f"static_check={timings.get('static_check', 0.0):.3f}s, "
            f"clean_filter={timings.get('clean_filter', 0.0):.3f}s, "
            f"tags_upsert={timings.get('tags_upsert', 0.0):.3f}s, "
            f"behavior_risk={timings.get('behavior_risk', 0.0):.3f}s, "
            f"eligibility={timings.get('eligibility', 0.0):.3f}s, "
            f"buffer_update={timings.get('buffer_update', 0.0):.3f}s, "
            f"routing={timings.get('routing', 0.0):.3f}s, "
            f"progress_update={timings.get('progress_update', 0.0):.3f}s)"
        )

    def _evict_user(self, user_id, batch_date, reason_code):
        self.state_manager.clear_buffer_and_update_progress(user_id, batch_date, "normal")
        print(f"📉 [Evicted] User {user_id} removed from bench. Reason: {reason_code}")

    def _mark_risk_user(self, user_id, batch_date, reason_code):
        self.state_manager.clear_buffer_and_update_progress(user_id, batch_date, "risk_user")
        print(f"🚫 [Risk] User {user_id} blacklisted. Reason: {reason_code}")

    def process_user_incremental(self, user_data: dict, new_raw_posts: list[dict], batch_date: str):
        user_id = str(user_data.get("user_id", ""))
        raw_count = len(new_raw_posts or [])
        timings = {
            "state_get": 0.0,
            "static_check": 0.0,
            "clean_filter": 0.0,
            "tags_upsert": 0.0,
            "behavior_risk": 0.0,
            "eligibility": 0.0,
            "buffer_update": 0.0,
            "routing": 0.0,
            "progress_update": 0.0,
        }

        state_start = time.perf_counter()
        state = self.state_manager.get_user_state(user_id)
        timings["state_get"] = time.perf_counter() - state_start
        current_role = state.get("dataset_role")
        current_buffer = state.get("buffered_posts", [])
        is_trusted_user = current_role == "bench_hashtag"

        if current_role == "risk_user":
            self._log_user_process_timing(
                user_id=user_id,
                status="skipped_blacklist",
                timings=timings,
                raw_count=raw_count,
                valid_count=0,
                combined_count=len(current_buffer),
            )
            return {"status": "skipped_blacklist", "user_id": user_id, "reason": "previous_risk_user"}

        if not is_trusted_user:
            static_start = time.perf_counter()
            is_qualified, reason = self.platform_filter.is_user_qualified(user_data)
            timings["static_check"] = time.perf_counter() - static_start
            if not is_qualified:
                progress_start = time.perf_counter()
                self.state_manager.clear_buffer_and_update_progress(user_id, batch_date, "normal")
                timings["progress_update"] += time.perf_counter() - progress_start
                self._log_user_process_timing(
                    user_id=user_id,
                    status="skipped_static",
                    timings=timings,
                    raw_count=raw_count,
                    valid_count=0,
                    combined_count=0,
                )
                return {"status": "skipped_static", "reason": reason, "user_id": user_id}

        # 先做内容级清洗/致命风险检查
        clean_filter_start = time.perf_counter()
        valid_new_posts, fatal_reason = self._clean_and_filter_posts(new_raw_posts, user_id=user_id)
        timings["clean_filter"] = time.perf_counter() - clean_filter_start
        valid_count = len(valid_new_posts) if valid_new_posts else 0

        if valid_new_posts is None:
            # 恢复旧版语义：trusted 用户仅软跳过，不拉黑
            if is_trusted_user:
                progress_start = time.perf_counter()
                self.state_manager.update_user_progress(user_id, batch_date, current_role or "bench_hashtag")
                timings["progress_update"] += time.perf_counter() - progress_start
                self._log_user_process_timing(
                    user_id=user_id,
                    status="skipped_risk_soft",
                    timings=timings,
                    raw_count=raw_count,
                    valid_count=0,
                    combined_count=len(current_buffer),
                )
                return {
                    "status": "skipped_risk_soft",
                    "reason": fatal_reason or "fatal_risk_on_new_batch",
                    "user_id": user_id,
                }
            progress_start = time.perf_counter()
            self._mark_risk_user(user_id, batch_date, fatal_reason or "fatal_risk_on_new_batch")
            timings["progress_update"] += time.perf_counter() - progress_start
            self._log_user_process_timing(
                user_id=user_id,
                status="skipped_risk",
                timings=timings,
                raw_count=raw_count,
                valid_count=0,
                combined_count=0,
            )
            return {
                "status": "skipped_risk",
                "reason": fatal_reason or "fatal_risk_on_new_batch",
                "user_id": user_id,
            }

        if not valid_new_posts and not current_buffer:
            if self.skip_empty_progress_update and raw_count == 0:
                self._log_user_process_timing(
                    user_id=user_id,
                    status="empty_batch",
                    timings=timings,
                    raw_count=raw_count,
                    valid_count=0,
                    combined_count=0,
                )
                return {"status": "empty_batch", "user_id": user_id}

            progress_start = time.perf_counter()
            self.state_manager.update_user_progress(user_id, batch_date, current_role if current_role else "normal")
            timings["progress_update"] += time.perf_counter() - progress_start
            self._log_user_process_timing(
                user_id=user_id,
                status="empty_batch",
                timings=timings,
                raw_count=raw_count,
                valid_count=0,
                combined_count=0,
            )
            return {"status": "empty_batch", "user_id": user_id}

        combined_buffer = current_buffer + (valid_new_posts if valid_new_posts else [])

        if valid_new_posts:
            upsert_start = time.perf_counter()
            tags_to_save = []
            for p in valid_new_posts:
                tags_to_save.extend(p.get("valid_tags", []))
            self.state_manager.batch_upsert_tags(tags_to_save)
            timings["tags_upsert"] = time.perf_counter() - upsert_start

        if not is_trusted_user:
            behavior_start = time.perf_counter()
            is_safe, risk_reason = self.platform_filter.check_behavior_risk(combined_buffer, self.api_client)
            timings["behavior_risk"] = time.perf_counter() - behavior_start
            if not is_safe:
                progress_start = time.perf_counter()
                self._mark_risk_user(user_id, batch_date, risk_reason or "failed_behavior_risk")
                timings["progress_update"] += time.perf_counter() - progress_start
                self._log_user_process_timing(
                    user_id=user_id,
                    status="skipped_risk",
                    timings=timings,
                    raw_count=raw_count,
                    valid_count=valid_count,
                    combined_count=len(combined_buffer),
                )
                return {"status": "skipped_risk", "user_id": user_id, "reason": risk_reason}

        eligibility_start = time.perf_counter()
        if not self._check_strict_eligibility(user_id, combined_buffer):
            timings["eligibility"] = time.perf_counter() - eligibility_start
            progress_start = time.perf_counter()
            self._evict_user(user_id, batch_date, "failed_density_check")
            timings["progress_update"] += time.perf_counter() - progress_start
            self._log_user_process_timing(
                user_id=user_id,
                status="evicted_density_drop",
                timings=timings,
                raw_count=raw_count,
                valid_count=valid_count,
                combined_count=len(combined_buffer),
            )
            return {"status": "evicted_density_drop", "user_id": user_id}
        timings["eligibility"] = time.perf_counter() - eligibility_start

        buffer_update_start = time.perf_counter()
        buffered_role = None
        if len(combined_buffer) < self.trigger_threshold:
            buffered_role = current_role if is_trusted_user else "bench_hashtag"
        self.state_manager.update_user_buffer(user_id, combined_buffer, batch_date, dataset_role=buffered_role)
        timings["buffer_update"] = time.perf_counter() - buffer_update_start

        if len(combined_buffer) < self.trigger_threshold:
            self._log_user_process_timing(
                user_id=user_id,
                status="buffered_qualified",
                timings=timings,
                raw_count=raw_count,
                valid_count=valid_count,
                combined_count=len(combined_buffer),
            )
            return {
                "status": "buffered_qualified",
                "count": len(combined_buffer),
                "msg": f"Accumulating posts ({len(combined_buffer)}/{self.trigger_threshold})",
                "user_id": user_id,
            }

        process_buffer = combined_buffer[-self.max_posts_per_call :]
        enrich_user_data = user_data.copy()
        enrich_user_data["valid_tags"] = self._collect_tags(process_buffer)
        enrich_user_data["username"] = user_data.get("display_name") or user_data.get("username") or "Unknown"

        routing_start = time.perf_counter()
        final_role = self.selector.route_and_save(
            user_id=user_id,
            user_data=enrich_user_data,
            valid_posts=process_buffer,
            batch_date=batch_date,
        )
        timings["routing"] = time.perf_counter() - routing_start

        progress_start = time.perf_counter()
        self.state_manager.clear_buffer_and_update_progress(user_id, batch_date, final_role, interest_tree=None)
        timings["progress_update"] += time.perf_counter() - progress_start

        self._log_user_process_timing(
            user_id=user_id,
            status="success_skipped_llm",
            timings=timings,
            raw_count=raw_count,
            valid_count=valid_count,
            combined_count=len(combined_buffer),
        )
        return {"status": "success_skipped_llm", "category": final_role, "user_id": user_id}

    def _check_strict_eligibility(self, user_id, posts):
        is_whitelisted = str(user_id) in self.selector.profile_bench_ids
        total_posts = len(posts)
        if total_posts == 0:
            return False
        total_anchors_count = sum(len(p.get("valid_tags", [])) for p in posts)
        avg_anchors_per_post = total_anchors_count / total_posts

        # 跨帖去重后的 unique tag 数
        unique_tags = {t for p in posts for t in p.get("valid_tags", [])}
        unique_count = len(unique_tags)

        status = "qualified"
        if avg_anchors_per_post < self.anchor_density_lower_bound:
            status = "too_low"
        elif avg_anchors_per_post > self.anchor_density_upper_bound:
            status = "too_high"
        elif self.min_unique_tags > 0 and unique_count < self.min_unique_tags:
            status = "low_diversity"
        if is_whitelisted:
            status = "whitelist_pass"

        with self.stats_lock:
            with open(self.stats_file, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([str(user_id), total_posts, total_anchors_count, f"{avg_anchors_per_post:.4f}", unique_count, status])

        if is_whitelisted:
            return True
        return status not in {"too_low", "too_high", "low_diversity"}

    def _clean_and_filter_posts(self, raw_posts, user_id: str = ""):
        stage_start = time.perf_counter()
        dedup_elapsed = 0.0
        batch_check_elapsed = 0.0
        fatal_elapsed = 0.0
        anchor_elapsed = 0.0
        clean_elapsed = 0.0
        meaningful_elapsed = 0.0

        if not raw_posts:
            return [], None

        dedup_start = time.perf_counter()
        seen_content_hashes = set()
        unique_raw_posts = []
        for post in raw_posts:
            raw_content = str(post.get("content", "") or "")
            if not raw_content.strip():
                continue
            normalized_str = re.sub(r"\s+", "", raw_content)
            content_hash = hash(normalized_str)
            if content_hash in seen_content_hashes:
                continue
            seen_content_hashes.add(content_hash)
            unique_raw_posts.append(post)
        dedup_elapsed = time.perf_counter() - dedup_start

        if not unique_raw_posts:
            return [], None

        batch_check_start = time.perf_counter()
        is_batch_valid, _ = self.platform_filter.validate_batch_behavior(unique_raw_posts, self.quality_config)
        batch_check_elapsed = time.perf_counter() - batch_check_start
        if not is_batch_valid:
            return None, "Fatal: Bot-like Behavior Detected"

        valid_buffered_posts = []
        for post in unique_raw_posts:
            fatal_start = time.perf_counter()
            is_fatal, fatal_reason = self.platform_filter.check_fatal_risk(post)
            fatal_elapsed += time.perf_counter() - fatal_start
            if is_fatal:
                return None, f"Fatal(Raw): {fatal_reason}"

            raw_content = str(post.get("content", "") or "")
            is_cleanable, _ = self.platform_filter.is_post_content_cleanable(raw_content)
            if not is_cleanable:
                continue

            anchor_start = time.perf_counter()
            valid_tags = self.platform_filter.extract_anchors(post)
            anchor_elapsed += time.perf_counter() - anchor_start

            clean_start = time.perf_counter()
            clean_text = self.platform_filter.clean_text(post)
            clean_elapsed += time.perf_counter() - clean_start

            spam_fatal_start = time.perf_counter()
            is_fatal_spam, spam_reason = self.platform_filter.check_tag_dependent_fatal_risk(clean_text, valid_tags)
            fatal_elapsed += time.perf_counter() - spam_fatal_start
            if is_fatal_spam:
                return None, f"Fatal(Spam): {spam_reason}"

            meaningful_start = time.perf_counter()
            is_meaningful, _ = self.platform_filter.is_post_meaningful(clean_text, valid_tags)
            meaningful_elapsed += time.perf_counter() - meaningful_start
            if not is_meaningful:
                continue

            standardized_post = dict(post)
            standardized_post["title"] = self.platform_filter._strip_html(
                self.platform_filter._clean_urls(standardized_post.get("title") or "")
            )
            standardized_post["content"] = self.platform_filter._strip_html(
                self.platform_filter._clean_urls(standardized_post.get("content") or "")
            )
            standardized_post["quote_content"] = self.platform_filter._strip_html(
                self.platform_filter._clean_urls(standardized_post.get("quote_content") or "")
            )
            standardized_post["valid_tags"] = valid_tags
            valid_buffered_posts.append(standardized_post)

        total_elapsed = time.perf_counter() - stage_start
        if total_elapsed >= self.filter_slow_log_seconds:
            print(
                "⏱️ [Filter Slow] "
                f"user={user_id or 'unknown'} | raw={len(raw_posts)} unique={len(unique_raw_posts)} kept={len(valid_buffered_posts)} "
                f"| total={total_elapsed:.3f}s "
                f"(dedup={dedup_elapsed:.3f}s, batch_check={batch_check_elapsed:.3f}s, "
                f"fatal={fatal_elapsed:.3f}s, anchor={anchor_elapsed:.3f}s, "
                f"clean={clean_elapsed:.3f}s, meaningful={meaningful_elapsed:.3f}s)"
            )

        return valid_buffered_posts, None

    def _collect_tags(self, posts):
        tags = []
        for p in posts:
            tags.extend(p.get("valid_tags", []))
        return list(dict.fromkeys(tags))
