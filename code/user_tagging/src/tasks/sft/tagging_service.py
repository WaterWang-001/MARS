import logging
import re
import os
import concurrent.futures
import pandas as pd

from ...core.merge_util import merge_interest_trees, merge_profiles
from ...core.nlp_processor import NLPProcessor
from ...platforms.base_filter import BasePlatformFilter
from .dataset_manager import DatasetManager
from .state_manager import StateManager


class TaggingService:
    def __init__(self, db_client, prompt_manager, api_client, stopwords_path,
                 state_manager, output_dir, config,
                 platform_filter: BasePlatformFilter):
        self.db_client = db_client
        self.prompt_manager = prompt_manager
        self.api_client = api_client
        self.state_manager = state_manager
        self.output_dir = output_dir
        self.config = config
        self.platform_filter = platform_filter

        buffer_rules = self.config.get('buffer_rules', {})
        behavior_risk_cfg = self.config.get('quality_rules', {}).get('behavior_risk', {})

        self.trigger_threshold = int(buffer_rules.get('trigger_threshold', 5))
        self.max_posts_per_call = int(buffer_rules.get('max_posts_per_call', 50))
        self.quality_config = {
            "repetition_threshold": float(behavior_risk_cfg.get("repetition_threshold", 0.3)),
            "fuzzy_repetition_threshold": float(behavior_risk_cfg.get("fuzzy_repetition_threshold", 0.8)),
            "min_avg_time_gap": float(behavior_risk_cfg.get("min_avg_time_gap", 60)),
        }

        logging.info("[TaggingService] Initializing DatasetManager...")
        sft_quality_config = self.config.get('sft_quality', {})
        self.dataset_manager = DatasetManager(
            output_dir=self.output_dir,
            sft_quality_config=sft_quality_config
        )

        logging.info("[TaggingService] Initializing NLPProcessor...")
        self.nlp_processor = NLPProcessor(stopwords_path)

    # =========================================================================
    #  Helper Methods
    # =========================================================================

    def _normalize_gender(self, val):
        if pd.isna(val): return "NA"
        s = str(val).lower()
        if s in ['f', 'female', '女']: return "Female"
        if s in ['m', 'male', '男']: return "Male"
        return "NA"

    def _format_posts_for_llm(self, posts: list) -> str:
        if not posts: return ""
        formatted_lines = []
        for i, p in enumerate(posts, 1):
            content = str(p.get('content', '') or '').strip()
            quote = str(p.get('quote_content', '') or '').strip()
            if content.lower() == 'nan': content = ""
            if quote.lower() == 'nan': quote = ""
            if not content and not quote: continue
            parts = [f"[{i}] Content: {content}"]
            if quote: parts.append(f"| Quote: {quote}")
            formatted_lines.append(" ".join(parts))
        return "\n".join(formatted_lines)

    def _prepare_context(self, user_data: dict):
        user_id = str(user_data.get('user_id', '')).strip()
        display_name = user_data.get('display_name', '')
        username_raw = user_data.get('username', '')
        username_final = str(display_name).strip() if pd.notna(display_name) and str(display_name).strip() else str(username_raw).strip() or "Unknown"
        bio = str(user_data.get('bio', '')).strip() if pd.notna(user_data.get('bio', '')) else "NA"

        verified = user_data.get('verified', False)
        verified_type = str(user_data.get('verified_type', ''))
        verified_info = f"Verified: {verified} (Type: {verified_type})"

        is_org = str(verified_type) in ['1', '2', '3', '4', '5', '6', '7', '8']
        mapped_type_name = "机构" if is_org else "个人"

        prov = str(user_data.get('province', ''))
        city = str(user_data.get('city', ''))
        if prov.lower() == 'nan': prov = ''
        if city.lower() == 'nan': city = ''
        location_final = f"{prov} {city}".strip() or "NA"

        return {
            "user_id": user_id,
            "username": username_final,
            "bio": bio,
            "verified_info": verified_info,
            "verified_type": verified_type,
            "mapped_type_name": mapped_type_name,
            "is_org": is_org,
            "gender": self._normalize_gender(user_data.get('gender')),
            "reg_time": str(user_data.get('registration_time', 'NA')),
            "location": location_final
        }

    def _update_user_skipped(self, user_id, batch_date):
        state = self.state_manager.get_user_state(user_id)
        self.state_manager.update_user_state(user_id, batch_date, state["interest_tree"], state["profile"], persona_summary=state.get("persona_summary", ""))

    # =========================================================================
    #  Private Logic Blocks
    # =========================================================================

    def _check_static_filter(self, user_data, user_id, batch_date):
        is_user_qualified, reason = self.platform_filter.is_user_qualified(user_data)
        if not is_user_qualified:
            self._update_user_skipped(user_id, batch_date)
            return {"status": "filtered_user", "reason": reason, "user_id": user_id}
        return None

    def _fetch_and_clean_posts(self, user_id, state, batch_date, prefetched_posts=None):
        last_cursor_time = state["last_cursor_time"]

        # A. 获取 Raw Posts
        if prefetched_posts is not None:
            raw_posts = []
            for p in prefetched_posts:
                p_time = str(p.get('created_at', p.get('publish_time', '')))
                if p_time > last_cursor_time:
                    raw_posts.append(p)
        else:
            raw_posts = self.db_client.get_user_posts_incremental(user_id, last_cursor_time)

        # B. 批量行为检测 (bot/spam)
        is_batch_valid, batch_reason = self.platform_filter.validate_batch_behavior(
            raw_posts, self.quality_config
        )
        if not is_batch_valid:
            self._update_user_skipped(user_id, batch_date)
            return [], {"status": "filtered_behavior", "reason": batch_reason, "user_id": user_id}

        # C. 逐帖过滤 (使用平台 filter)
        valid_buffered_posts = []
        for post in raw_posts:
            raw_content = str(post.get('content', '') or '')

            # 致命风险检查 (平台特化: 传入完整 post dict)
            is_fatal, fatal_reason = self.platform_filter.check_fatal_risk(post)
            if is_fatal:
                self._update_user_skipped(user_id, batch_date)
                return [], {"status": "filtered_fatal_content", "reason": fatal_reason, "user_id": user_id}

            # 内容可清洗性检查
            is_cleanable, _ = self.platform_filter.is_post_content_cleanable(raw_content)
            if not is_cleanable:
                continue

            # 平台特化锚点提取 (替代硬编码 #tag# 正则)
            valid_tags = self.platform_filter.extract_anchors(post)

            # 平台特化文本清洗 (替代 clean_text_for_tagging)
            clean_text = self.platform_filter.clean_text(post)

            # 内容有意义性检查
            is_meaningful, _ = self.platform_filter.is_post_meaningful(clean_text, valid_tags)
            if not is_meaningful:
                continue

            post['content'] = clean_text
            post['valid_tags'] = valid_tags
            valid_buffered_posts.append(post)

        return valid_buffered_posts, None

    def _prepare_llm_input(self, all_buffered):
        if len(all_buffered) > self.max_posts_per_call:
            process_buffer = all_buffered[-self.max_posts_per_call:]
        else:
            process_buffer = all_buffered

        posts_text = self._format_posts_for_llm(process_buffer)

        current_valid_tags = []
        for p in process_buffer:
            if 'valid_tags' in p:
                current_valid_tags.extend(p['valid_tags'])
            else:
                # fallback: 用平台 filter 重新提取
                current_valid_tags.extend(self.platform_filter.extract_anchors(p))

        return process_buffer, posts_text, current_valid_tags

    def _execute_llm_inference(self, ctx, process_buffer, posts_text, state):
        raw_contents = [str(p.get('content', '') or '') for p in process_buffer]
        prev_persona = state.get("persona_summary", "")

        def run_interest_task():
            candidates = self.nlp_processor.extract_candidates([ctx['bio']] + raw_contents)
            prompt = self.prompt_manager.get_interest_prompt(
                username=ctx['username'], bio=ctx['bio'], posts=posts_text,
                candidate_entities=candidates, prev_persona=prev_persona
            )
            return self.api_client.call_api(prompt)

        prev_profile = state.get("profile", {})

        def run_profile_task():
            if ctx['is_org']:
                return self.api_client.call_api(self.prompt_manager.get_firmographic_prompt(
                    username=ctx['username'], bio=ctx['bio'], posts=posts_text,
                    verified_info=ctx['verified_info'], mapped_type_name=ctx['mapped_type_name'],
                    verified_type=ctx['verified_type'], prev_profile=prev_profile
                ))
            else:
                return self.api_client.call_api(self.prompt_manager.get_demographic_prompt(
                    username=ctx['username'], bio=ctx['bio'], posts=posts_text,
                    gender_reported=ctx['gender'], reg_time=ctx['reg_time'],
                    location_reported=ctx['location'], verified_info=ctx['verified_info'],
                    prev_profile=prev_profile
                ))

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                future_interest = executor.submit(run_interest_task)
                future_profile = executor.submit(run_profile_task)

                res_interest = future_interest.result() or {}
                new_partial_tree = res_interest.get("interest_tree", [])
                new_persona = res_interest.get("persona_summary", "")

                res_profile = future_profile.result() or {}

                if "firmographic_profile" in res_profile:
                    new_partial_profile = res_profile["firmographic_profile"]
                elif "demographic_profile" in res_profile:
                    new_partial_profile = res_profile["demographic_profile"]
                else:
                    new_partial_profile = res_profile

            # TreeMerge: 置信度竞争 + NA驱逐 + 拓扑并集
            merged_tree = merge_interest_trees(state["interest_tree"], new_partial_tree)
            # ProfileMerge: 置信度单调非减覆写
            merged_profile = merge_profiles(state["profile"], new_partial_profile)
            merged_profile["user_type"] = ctx['mapped_type_name']
            merged_profile["is_org"] = ctx['is_org']

            updated_persona = new_persona if new_persona else prev_persona

            return merged_tree, merged_profile, updated_persona

        except Exception as e:
            logging.error(f"[User {ctx['user_id']}] Parallel Task Error: {e}")
            return None, None, ""

    def _handle_downstream_ops(self, user_id, user_data, batch_date, posts_text, merged_tree, current_role, valid_tags):
        """数据集筛选落盘 (SFT vs Normal)"""
        enrich_user_data = user_data.copy()
        enrich_user_data['valid_tags'] = valid_tags
        enrich_user_data['batch_date'] = batch_date

        try:
            save_msg, final_role = self.dataset_manager.classify_and_save_user(
                user_id=user_id,
                user_data=enrich_user_data,
                posts_text=posts_text,
                interest_tree=merged_tree,
                current_role=current_role
            )
        except Exception as e:
            logging.error(f"[User {user_id}] Dataset Classify Error: {e}")
            save_msg = "Error in Classification"
            final_role = current_role

        return save_msg, final_role

    def _finalize_state(self, user_id, batch_date, tree, profile, role=None, persona_summary="", clear_buffer=False):
        if clear_buffer:
            self.state_manager.clear_user_buffer(user_id)
        self.state_manager.update_user_state(user_id, batch_date, tree, profile, dataset_role=role, persona_summary=persona_summary)

    # =========================================================================
    #  Main Logic
    # =========================================================================

    def process_user_incremental(self, user_data, batch_date, prefetched_posts=None):
        # 0. Context
        ctx = self._prepare_context(user_data)
        user_id = ctx['user_id']

        # 1. Level 1 Filter (平台特化)
        static_check = self._check_static_filter(user_data, user_id, batch_date)
        if static_check: return static_check

        # 2. Get State: S_{n-1}
        state = self.state_manager.get_user_state(user_id)

        # 3. Level 2-4 Fetch & Clean (平台特化过滤链)
        valid_new_posts, filter_response = self._fetch_and_clean_posts(user_id, state, batch_date, prefetched_posts)
        if filter_response: return filter_response

        # 4. Buffer Logic
        if valid_new_posts:
            self.state_manager.add_to_buffer(user_id, valid_new_posts)

        all_buffered = self.state_manager.get_buffered_posts(user_id)

        if len(all_buffered) < self.trigger_threshold:
            self._finalize_state(user_id, batch_date, state["interest_tree"], state["profile"], state.get("dataset_role"), persona_summary=state.get("persona_summary", ""))
            if not valid_new_posts:
                return {"status": "no_valid_posts", "user_id": user_id}
            return {"status": "buffered", "count": len(all_buffered), "user_id": user_id}

        process_buffer, posts_text, current_valid_tags = self._prepare_llm_input(all_buffered)

        print(f"🤖 [LLM Start] User {user_id} passed filters. Invoking LLM... (This may take 30-60s)")

        # 6. LLM Execute: {S_n, P_n} = f_θ(D_n, S_{n-1})
        merged_tree, merged_profile, updated_persona = self._execute_llm_inference(ctx, process_buffer, posts_text, state)

        if merged_tree is None:
            print(f"[LLM Fail] User {user_id} inference returned None.")
            return None

        print(f"✅ [LLM Done] User {user_id} inference complete.")

        # 7. Downstream Ops (SFT vs Normal 分类)
        save_status, final_dataset_role = self._handle_downstream_ops(
            user_id,
            user_data,
            batch_date,
            posts_text,
            merged_tree,
            state.get("dataset_role"),
            current_valid_tags
        )

        # 8. Finalize: 持久化 S_n, P_n
        self._finalize_state(user_id, batch_date, merged_tree, merged_profile, final_dataset_role, persona_summary=updated_persona, clear_buffer=True)

        return {
            "status": "updated",
            "user_id": user_id,
            "save_status": save_status,
            "role": final_dataset_role,
            "user_type": ctx['mapped_type_name'],
            "interest_tree": merged_tree,
            "profile": merged_profile,
            "persona_summary": updated_persona
        }
